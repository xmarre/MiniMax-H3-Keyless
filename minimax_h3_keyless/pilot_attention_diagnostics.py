from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from .activation_capture import CapturedPilotCase
from .activation_metrics import tensor_error_metrics
from .attention import KeylessAttentionTrain
from .ops import normalized_positioned


STAGE_A_DIAGNOSTIC_MAX_QUERY_ROWS = 32
STAGE_A_DIAGNOSTIC_HEAD_CHUNK = 8
STAGE_A_DIAGNOSTIC_KEY_CHUNK = 512
_NATIVE_SEGMENT_KINDS = (
    "text",
    "cond",
    "cond_audio",
    "ref_img",
    "ref_audio",
    "audio",
    "video",
)
_CAPTURE_CONTEXT_KEY = "minimax_h3_keyless_live_capture_v1"


@dataclass(frozen=True)
class PilotAttentionDiagnostic:
    """Bounded complete-key comparison for one held-out Stage-A capture.

    Query rows are deterministically sampled while every key/value row is streamed.
    Heads are processed in fixed-size chunks and reassembled only at the small sampled
    query set so pre/post-out-projection metrics preserve the real all-head output.
    Modality masses are aligned to ``modality_kinds``.
    """

    case_id: str
    sigma: float | None
    modality_label: str | None
    sampled_query_rows: tuple[int, ...]
    head_chunk_size: int
    key_chunk_size: int
    modality_kinds: tuple[str, ...]
    mean_centered_logit_nrmse: float
    mean_teacher_to_student_softmax_kl: float
    pre_out_normalized_rmse: float
    pre_out_cosine: float
    post_out_normalized_rmse: float
    post_out_cosine: float
    teacher_modality_mass: tuple[float, ...]
    student_modality_mass: tuple[float, ...]


def _capture_layout(record: CapturedPilotCase) -> tuple[int, tuple[tuple[int, int, str], ...]]:
    context = record.case.context
    if not isinstance(context, Mapping):
        raise RuntimeError("Stage-A attention diagnostics require capture context metadata")
    capture = context.get(_CAPTURE_CONTEXT_KEY)
    if not isinstance(capture, Mapping):
        raise RuntimeError("Stage-A attention diagnostics require live-capture layout metadata")
    layout = capture.get("layout")
    if not isinstance(layout, Mapping):
        raise RuntimeError("Stage-A attention diagnostics require captured layout metadata")
    seq_len = layout.get("seq_len")
    if isinstance(seq_len, bool) or not isinstance(seq_len, int) or seq_len <= 0:
        raise RuntimeError("captured Stage-A layout has an invalid sequence length")
    raw_segments = layout.get("segments")
    if not isinstance(raw_segments, (tuple, list)) or not raw_segments:
        raise RuntimeError("captured Stage-A layout is missing exact contiguous segments")

    segments: list[tuple[int, int, str]] = []
    previous = 0
    for index, raw in enumerate(raw_segments):
        if not isinstance(raw, (tuple, list)) or len(raw) != 3:
            raise RuntimeError(f"captured Stage-A layout segment {index} is malformed")
        start, stop, kind = raw
        if (
            isinstance(start, bool)
            or not isinstance(start, int)
            or isinstance(stop, bool)
            or not isinstance(stop, int)
            or not isinstance(kind, str)
        ):
            raise RuntimeError(f"captured Stage-A layout segment {index} has invalid types")
        if start != previous or stop <= start or stop > seq_len:
            raise RuntimeError("captured Stage-A layout segments are not one contiguous row partition")
        if kind not in _NATIVE_SEGMENT_KINDS:
            raise RuntimeError(f"captured Stage-A layout has unknown native segment kind {kind!r}")
        segments.append((start, stop, kind))
        previous = stop
    if previous != seq_len:
        raise RuntimeError("captured Stage-A layout segments do not cover the complete key domain")
    if record.attention_input.shape[0] != seq_len:
        raise RuntimeError(
            "captured Stage-A attention-input rows do not match the captured layout sequence length"
        )
    return seq_len, tuple(segments)


def deterministic_query_rows(
    seq_len: int,
    segments: Sequence[tuple[int, int, str]],
    *,
    max_query_rows: int = STAGE_A_DIAGNOSTIC_MAX_QUERY_ROWS,
) -> tuple[int, ...]:
    """Choose a deterministic bounded query set that represents every packed segment."""
    if isinstance(max_query_rows, bool) or not isinstance(max_query_rows, int) or max_query_rows <= 0:
        raise ValueError("max_query_rows must be a positive integer")
    if len(segments) > max_query_rows:
        raise RuntimeError(
            "Stage-A packed layout has more segments than the bounded diagnostic query budget"
        )
    selected: set[int] = set()
    # One midpoint per segment guarantees that text/conditioning/reference/audio/video
    # segments present in this case are all represented before any global fill rows.
    for start, stop, _ in segments:
        selected.add(start + (stop - start - 1) // 2)

    # Add segment boundaries round-robin; these catch local timeline/domain transitions.
    for position in (0, -1):
        for start, stop, _ in segments:
            if len(selected) >= max_query_rows:
                break
            selected.add(start if position == 0 else stop - 1)
        if len(selected) >= max_query_rows:
            break

    # Fill remaining capacity evenly over the complete packed sequence without RNG state.
    remaining = max_query_rows - len(selected)
    if remaining > 0 and len(selected) < seq_len:
        count = min(remaining, seq_len - len(selected))
        if count == 1:
            candidates = [seq_len // 2]
        else:
            candidates = [round(i * (seq_len - 1) / (count - 1)) for i in range(count)]
        for row in candidates:
            selected.add(int(row))
        # Rounding/deduplication can leave unused capacity on small/irregular domains.
        if len(selected) < min(max_query_rows, seq_len):
            stride = max(1, seq_len // max_query_rows)
            for row in range(0, seq_len, stride):
                selected.add(row)
                if len(selected) >= min(max_query_rows, seq_len):
                    break
    return tuple(sorted(selected))


def _modality_ids(
    seq_len: int,
    segments: Sequence[tuple[int, int, str]],
) -> tuple[torch.Tensor, tuple[str, ...]]:
    kinds: list[str] = []
    kind_to_index: dict[str, int] = {}
    ids = torch.empty(seq_len, dtype=torch.long)
    for start, stop, kind in segments:
        if kind not in kind_to_index:
            kind_to_index[kind] = len(kinds)
            kinds.append(kind)
        ids[start:stop] = kind_to_index[kind]
    return ids, tuple(kinds)


def _slice_rope(
    rope_freqs: torch.Tensor | None,
    rows: slice | torch.Tensor,
) -> torch.Tensor | None:
    if rope_freqs is None:
        return None
    if rope_freqs.ndim != 6:
        raise RuntimeError("captured Stage-A RoPE table has an unexpected rank")
    return rope_freqs[:, rows]


def _bias_free_weight(module: nn.Module, name: str) -> torch.Tensor:
    weight = getattr(module, "weight", None)
    if not torch.is_tensor(weight) or weight.ndim != 2 or getattr(weight, "is_meta", False):
        raise RuntimeError(f"Stage-A diagnostics require a materialized {name} weight")
    if getattr(module, "bias", None) is not None:
        raise RuntimeError(f"Stage-A diagnostics require bias-free {name}")
    return weight


def _online_weights(
    logits: torch.Tensor,
    old_max: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    chunk_max = logits.max(dim=-1).values
    new_max = torch.maximum(old_max, chunk_max)
    if not torch.isfinite(new_max).all():
        raise RuntimeError("Stage-A attention diagnostics observed non-finite logits")
    old_scale = torch.where(
        torch.isfinite(old_max),
        torch.exp(old_max - new_max),
        torch.zeros_like(new_max),
    )
    weights = torch.exp(logits - new_max.unsqueeze(-1))
    return new_max, old_scale, weights


def _head_chunk_comparison(
    *,
    hidden_cpu: torch.Tensor,
    query_rows: torch.Tensor,
    modality_ids: torch.Tensor,
    modality_count: int,
    rope_cpu: torch.Tensor | None,
    teacher_attention: nn.Module,
    student_attention: KeylessAttentionTrain,
    head_start: int,
    head_stop: int,
    key_chunk_size: int,
) -> tuple[torch.Tensor, ...]:
    heads = int(teacher_attention.heads)
    head_dim = int(teacher_attention.head_dim)
    inner = heads * head_dim
    hs = slice(head_start * head_dim, head_stop * head_dim)
    teacher_qkv = _bias_free_weight(teacher_attention.qkv_proj, "teacher qkv projection")
    student_q = _bias_free_weight(student_attention.q_proj, "student q projection")
    student_v = _bias_free_weight(student_attention.v_proj, "student v projection")
    device = teacher_qkv.device
    head_count = head_stop - head_start
    queries = int(query_rows.numel())
    scale = head_dim ** -0.5

    xq_t = hidden_cpu.index_select(0, query_rows).to(device=device, dtype=teacher_qkv.dtype)
    tq_raw = F.linear(xq_t, teacher_qkv[hs]).view(queries, head_count, head_dim)
    rope_q = _slice_rope(rope_cpu, query_rows)
    tq = normalized_positioned(tq_raw, teacher_attention.q_norm.weight, teacher_attention.q_norm.eps, rope_q)

    xq_s = hidden_cpu.index_select(0, query_rows).to(device=device, dtype=student_q.dtype)
    sq_raw = F.linear(xq_s, student_q[hs]).view(queries, head_count, head_dim)
    route_weight = student_attention.query_route.weight[head_start:head_stop]
    sq_raw = torch.einsum("qhd,hed->qhe", sq_raw, route_weight)
    sq = normalized_positioned(sq_raw, student_attention.q_norm.weight, student_attention.q_norm.eps, rope_q)

    tqh = tq.float().permute(1, 0, 2)
    sqh = sq.float().permute(1, 0, 2)
    mt = torch.full((head_count, queries), float("-inf"), dtype=torch.float32, device=device)
    ms = mt.clone()
    zt = torch.zeros((head_count, queries), dtype=torch.float32, device=device)
    zs = zt.clone()
    kl_num = zt.clone()
    out_t_num = torch.zeros((head_count, queries, head_dim), dtype=torch.float32, device=device)
    out_s_num = torch.zeros_like(out_t_num)
    count = torch.zeros((head_count, queries), dtype=torch.float64, device=device)
    sum_t = torch.zeros_like(count)
    sum_t2 = torch.zeros_like(count)
    sum_err = torch.zeros_like(count)
    sum_err2 = torch.zeros_like(count)
    mass_t = torch.zeros((head_count, queries, modality_count), dtype=torch.float32, device=device)
    mass_s = torch.zeros_like(mass_t)

    tk_rows = slice(inner + head_start * head_dim, inner + head_stop * head_dim)
    tv_rows = slice(2 * inner + head_start * head_dim, 2 * inner + head_stop * head_dim)
    sv_rows = slice(head_start * head_dim, head_stop * head_dim)
    for start in range(0, hidden_cpu.shape[0], key_chunk_size):
        stop = min(start + key_chunk_size, hidden_cpu.shape[0])
        xk_t = hidden_cpu[start:stop].to(device=device, dtype=teacher_qkv.dtype)
        tk_raw = F.linear(xk_t, teacher_qkv[tk_rows]).view(-1, head_count, head_dim)
        tv = F.linear(xk_t, teacher_qkv[tv_rows]).view(-1, head_count, head_dim)
        rope_k = _slice_rope(rope_cpu, slice(start, stop))
        tk = normalized_positioned(
            tk_raw,
            teacher_attention.k_norm.weight,
            teacher_attention.k_norm.eps,
            rope_k,
        )

        xk_s = hidden_cpu[start:stop].to(device=device, dtype=student_v.dtype)
        sv = F.linear(xk_s, student_v[sv_rows]).view(-1, head_count, head_dim)
        sr = normalized_positioned(
            sv,
            student_attention.route_norm.weight,
            student_attention.route_norm.eps,
            rope_k,
        )
        if not all(torch.isfinite(value).all() for value in (tk, tv, sr, sv)):
            raise RuntimeError("Stage-A attention diagnostics observed non-finite projected K/route/V")

        tkh = tk.float().permute(1, 0, 2)
        srh = sr.float().permute(1, 0, 2)
        tvh = tv.float().permute(1, 0, 2)
        svh = sv.float().permute(1, 0, 2)
        lt = torch.einsum("hqd,hkd->hqk", tqh, tkh) * scale
        ls = torch.einsum("hqd,hkd->hqk", sqh, srh) * scale
        if not torch.isfinite(lt).all() or not torch.isfinite(ls).all():
            raise RuntimeError("Stage-A attention diagnostics observed non-finite attention logits")

        ltd = lt.double()
        errd = (ls - lt).double()
        width = stop - start
        count += width
        sum_t += ltd.sum(dim=-1)
        sum_t2 += (ltd * ltd).sum(dim=-1)
        sum_err += errd.sum(dim=-1)
        sum_err2 += (errd * errd).sum(dim=-1)

        new_mt, scale_t, wt = _online_weights(lt, mt)
        new_ms, scale_s, ws = _online_weights(ls, ms)
        zt = zt * scale_t + wt.sum(dim=-1)
        zs = zs * scale_s + ws.sum(dim=-1)
        kl_num = kl_num * scale_t + (wt * (lt - ls)).sum(dim=-1)
        out_t_num = out_t_num * scale_t.unsqueeze(-1) + torch.einsum("hqk,hkd->hqd", wt, tvh)
        out_s_num = out_s_num * scale_s.unsqueeze(-1) + torch.einsum("hqk,hkd->hqd", ws, svh)
        one_hot = F.one_hot(
            modality_ids[start:stop].to(device=device),
            num_classes=modality_count,
        ).to(dtype=torch.float32)
        mass_t = mass_t * scale_t.unsqueeze(-1) + torch.einsum("hqk,km->hqm", wt, one_hot)
        mass_s = mass_s * scale_s.unsqueeze(-1) + torch.einsum("hqk,km->hqm", ws, one_hot)
        mt, ms = new_mt, new_ms

    if (count <= 0).any() or (zt <= 0).any() or (zs <= 0).any():
        raise RuntimeError("Stage-A attention diagnostics produced an invalid online softmax state")
    mean_t = sum_t / count
    mean_err = sum_err / count
    teacher_var = (sum_t2 / count - mean_t.square()).clamp_min(0.0)
    error_var = (sum_err2 / count - mean_err.square()).clamp_min(0.0)
    teacher_rms = teacher_var.sqrt()
    error_rms = error_var.sqrt()
    centered = torch.where(
        teacher_rms > 0,
        error_rms / teacher_rms,
        torch.where(error_rms == 0, torch.zeros_like(error_rms), torch.full_like(error_rms, float("inf"))),
    )
    logzt = mt + torch.log(zt)
    logzs = ms + torch.log(zs)
    kl = kl_num / zt + logzs - logzt
    teacher_out = out_t_num / zt.unsqueeze(-1)
    student_out = out_s_num / zs.unsqueeze(-1)
    teacher_mass = mass_t / zt.unsqueeze(-1)
    student_mass = mass_s / zs.unsqueeze(-1)
    return (
        centered.permute(1, 0),
        kl.permute(1, 0),
        teacher_out.permute(1, 0, 2),
        student_out.permute(1, 0, 2),
        teacher_mass.permute(1, 0, 2),
        student_mass.permute(1, 0, 2),
    )


def compare_captured_native_keyless_attention(
    teacher_attention: nn.Module,
    student_attention: KeylessAttentionTrain,
    record: CapturedPilotCase,
    *,
    max_query_rows: int = STAGE_A_DIAGNOSTIC_MAX_QUERY_ROWS,
    head_chunk_size: int = STAGE_A_DIAGNOSTIC_HEAD_CHUNK,
    key_chunk_size: int = STAGE_A_DIAGNOSTIC_KEY_CHUNK,
) -> PilotAttentionDiagnostic:
    """Compare teacher K-attention and Keyless value-routing on one exact capture.

    No QxK matrix or complete K/V projection is retained. The only full-sequence tensor
    consumed is the immutable captured post-AdaLN input already required by Stage A.
    """
    if not isinstance(student_attention, KeylessAttentionTrain):
        raise TypeError("Stage-A attention diagnostics require KeylessAttentionTrain")
    if isinstance(head_chunk_size, bool) or not isinstance(head_chunk_size, int) or head_chunk_size <= 0:
        raise ValueError("head_chunk_size must be a positive integer")
    if isinstance(key_chunk_size, bool) or not isinstance(key_chunk_size, int) or key_chunk_size <= 0:
        raise ValueError("key_chunk_size must be a positive integer")
    if int(teacher_attention.heads) != student_attention.heads or int(teacher_attention.head_dim) != student_attention.head_dim:
        raise RuntimeError("teacher/student Stage-A attention geometry differs")
    teacher_qkv = _bias_free_weight(teacher_attention.qkv_proj, "teacher qkv projection")
    if student_attention.q_proj.weight.device != teacher_qkv.device or student_attention.v_proj.weight.device != teacher_qkv.device:
        raise RuntimeError("teacher/student Stage-A attention weights must share one device")

    seq_len, segments = _capture_layout(record)
    query_tuple = deterministic_query_rows(seq_len, segments, max_query_rows=max_query_rows)
    query_rows = torch.tensor(query_tuple, dtype=torch.long)
    modality_ids, modality_kinds = _modality_ids(seq_len, segments)
    rope_cpu = record.case.rope_freqs
    if rope_cpu is not None and rope_cpu.shape[1] != seq_len:
        raise RuntimeError("captured Stage-A RoPE rows do not match the complete key domain")
    hidden_cpu = record.attention_input
    if hidden_cpu.device.type != "cpu":
        raise RuntimeError("canonical Stage-A attention diagnostics require CPU-resident captures")

    centered_parts = []
    kl_parts = []
    teacher_out_parts = []
    student_out_parts = []
    teacher_mass_parts = []
    student_mass_parts = []
    heads = int(teacher_attention.heads)
    with torch.no_grad():
        for head_start in range(0, heads, head_chunk_size):
            head_stop = min(head_start + head_chunk_size, heads)
            values = _head_chunk_comparison(
                hidden_cpu=hidden_cpu,
                query_rows=query_rows,
                modality_ids=modality_ids,
                modality_count=len(modality_kinds),
                rope_cpu=rope_cpu,
                teacher_attention=teacher_attention,
                student_attention=student_attention,
                head_start=head_start,
                head_stop=head_stop,
                key_chunk_size=key_chunk_size,
            )
            centered, kl, teacher_out, student_out, teacher_mass, student_mass = values
            centered_parts.append(centered.cpu())
            kl_parts.append(kl.cpu())
            teacher_out_parts.append(teacher_out.cpu())
            student_out_parts.append(student_out.cpu())
            teacher_mass_parts.append(teacher_mass.cpu())
            student_mass_parts.append(student_mass.cpu())

    centered = torch.cat(centered_parts, dim=1)
    kl = torch.cat(kl_parts, dim=1)
    teacher_out = torch.cat(teacher_out_parts, dim=1)
    student_out = torch.cat(student_out_parts, dim=1)
    teacher_mass = torch.cat(teacher_mass_parts, dim=1)
    student_mass = torch.cat(student_mass_parts, dim=1)
    if not all(torch.isfinite(value).all() for value in (centered, kl, teacher_out, student_out, teacher_mass, student_mass)):
        raise RuntimeError("Stage-A attention diagnostics produced non-finite comparison evidence")

    queries = len(query_tuple)
    teacher_flat = teacher_out.reshape(queries, -1)
    student_flat = student_out.reshape(queries, -1)
    pre_nrmse, pre_cosine = tensor_error_metrics(teacher_flat, student_flat)
    teacher_out_weight = _bias_free_weight(teacher_attention.out_proj, "teacher out projection")
    student_out_weight = _bias_free_weight(student_attention.out_proj, "student out projection")
    teacher_projected = F.linear(
        teacher_flat.to(device=teacher_out_weight.device, dtype=teacher_out_weight.dtype),
        teacher_out_weight,
    ).float().cpu()
    student_projected = F.linear(
        student_flat.to(device=student_out_weight.device, dtype=student_out_weight.dtype),
        student_out_weight,
    ).float().cpu()
    post_nrmse, post_cosine = tensor_error_metrics(teacher_projected, student_projected)

    teacher_modality = teacher_mass.mean(dim=(0, 1))
    student_modality = student_mass.mean(dim=(0, 1))
    scalars = (centered.mean(), kl.mean(), pre_nrmse, pre_cosine, post_nrmse, post_cosine)
    if not all(torch.isfinite(value) for value in scalars):
        raise RuntimeError("Stage-A attention diagnostics produced non-finite aggregate metrics")
    return PilotAttentionDiagnostic(
        case_id=record.case.case_id,
        sigma=record.case.sigma,
        modality_label=record.case.modality_label,
        sampled_query_rows=query_tuple,
        head_chunk_size=int(head_chunk_size),
        key_chunk_size=int(key_chunk_size),
        modality_kinds=modality_kinds,
        mean_centered_logit_nrmse=float(centered.mean().item()),
        mean_teacher_to_student_softmax_kl=float(kl.mean().item()),
        pre_out_normalized_rmse=float(pre_nrmse.item()),
        pre_out_cosine=float(pre_cosine.item()),
        post_out_normalized_rmse=float(post_nrmse.item()),
        post_out_cosine=float(post_cosine.item()),
        teacher_modality_mass=tuple(float(value) for value in teacher_modality.tolist()),
        student_modality_mass=tuple(float(value) for value in student_modality.tolist()),
    )
