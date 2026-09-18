from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class StreamedAttentionComparison:
    """Per sampled query/head diagnostics without persistent Q×K score storage.

    Per-query fields use H3 row-major shape ``[query_rows, heads]``. Attention
    outputs are ``[query_rows, heads, head_dim]``. Modality masses, when requested,
    are ``[query_rows, heads, modalities]`` in ``modality_values`` order.
    """

    centered_logit_nrmse: torch.Tensor
    softmax_kl_teacher_student: torch.Tensor
    output_nrmse: torch.Tensor
    output_cosine: torch.Tensor
    teacher_output: torch.Tensor
    student_output: torch.Tensor
    modality_values: tuple[int, ...]
    teacher_modality_mass: torch.Tensor | None
    student_modality_mass: torch.Tensor | None


def tensor_error_metrics(
    teacher: torch.Tensor, student: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return normalized RMS error and mean row cosine for pre/post projection tensors."""
    if teacher.shape != student.shape:
        raise ValueError("teacher/student output shapes must match")
    t = teacher.float()
    s = student.float()
    error_rms = (s - t).square().mean().sqrt()
    teacher_rms = t.square().mean().sqrt()
    nrmse = torch.where(
        teacher_rms > 0,
        error_rms / teacher_rms,
        torch.where(error_rms == 0, torch.zeros_like(error_rms), torch.full_like(error_rms, float("inf"))),
    )
    cosine = F.cosine_similarity(t.reshape(-1, t.shape[-1]), s.reshape(-1, s.shape[-1]), dim=-1).mean()
    return nrmse, cosine


def _mask_chunk(
    mask: torch.Tensor | None,
    *,
    heads: int,
    queries: int,
    start: int,
    stop: int,
    device: torch.device,
) -> torch.Tensor | None:
    if mask is None:
        return None
    part = mask[..., start:stop].to(device=device)
    if part.ndim == 2:
        if part.shape[0] != queries:
            raise ValueError("2D mask must have shape [query_rows, key_rows]")
        part = part.unsqueeze(0)
    elif part.ndim == 3:
        if part.shape[-2] != queries or part.shape[0] not in (1, heads):
            raise ValueError("3D mask must broadcast as [heads, query_rows, key_rows]")
    elif part.ndim == 4:
        if part.shape[0] != 1 or part.shape[1] not in (1, heads) or part.shape[2] != queries:
            raise ValueError("4D mask must broadcast as [1, heads, query_rows, key_rows]")
        part = part.squeeze(0)
    else:
        raise ValueError("mask must have rank 2, 3, or 4")
    return part.expand(heads, queries, stop - start)


def _effective_logits(
    logits: torch.Tensor,
    mask_chunk: torch.Tensor | None,
    log_measure_chunk: torch.Tensor | None,
) -> torch.Tensor:
    if mask_chunk is not None:
        if mask_chunk.dtype == torch.bool:
            logits = logits.masked_fill(~mask_chunk, float("-inf"))
        else:
            logits = logits + mask_chunk.float()
    if log_measure_chunk is not None:
        logits = logits + log_measure_chunk.view(1, 1, -1)
    if torch.isnan(logits).any() or torch.isposinf(logits).any():
        raise ValueError("attention diagnostics received NaN/+inf effective logits")
    return logits


def _online_weights(
    logits: torch.Tensor,
    old_max: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    chunk_max = logits.max(dim=-1).values
    new_max = torch.maximum(old_max, chunk_max)
    finite_new = torch.isfinite(new_max)
    old_scale = torch.where(
        torch.isfinite(old_max) & finite_new,
        torch.exp(old_max - new_max),
        torch.zeros_like(new_max),
    )
    shifted = logits - new_max.unsqueeze(-1)
    weights = torch.where(
        torch.isfinite(logits) & finite_new.unsqueeze(-1),
        torch.exp(shifted),
        torch.zeros_like(logits),
    )
    return new_max, old_scale, weights


def streamed_attention_comparison(
    teacher_q: torch.Tensor,
    teacher_k: torch.Tensor,
    teacher_v: torch.Tensor,
    student_q: torch.Tensor,
    student_route: torch.Tensor,
    student_v: torch.Tensor,
    *,
    scale: float | None = None,
    mask: torch.Tensor | None = None,
    log_measure: torch.Tensor | None = None,
    modality_ids: torch.Tensor | None = None,
    key_chunk_size: int = 1024,
) -> StreamedAttentionComparison:
    """Compare complete-key teacher/student attention in bounded key chunks.

    This routine is intended for the sampled-query diagnostics in the design. It never
    materializes a persistent ``query_rows × heads × key_rows`` tensor. The caller is
    still responsible for bounding the number of sampled heads/query rows.
    """
    if key_chunk_size <= 0:
        raise ValueError("key_chunk_size must be positive")
    if teacher_q.shape != student_q.shape or teacher_q.ndim != 3:
        raise ValueError("teacher/student Q must share [query_rows, heads, head_dim]")
    if teacher_k.shape != teacher_v.shape or student_route.shape != student_v.shape:
        raise ValueError("each routing operand must align with its retrieval value")
    if teacher_k.shape != student_route.shape or teacher_k.ndim != 3:
        raise ValueError("teacher/student key domains and head geometry must match")
    if teacher_q.shape[1:] != teacher_k.shape[1:]:
        raise ValueError("query/key head geometry must match")

    queries, heads, head_dim = teacher_q.shape
    keys = teacher_k.shape[0]
    if keys == 0:
        raise ValueError("attention key domain must be non-empty")
    if log_measure is not None and (log_measure.ndim != 1 or log_measure.shape[0] != keys):
        raise ValueError("log_measure must align with the complete key domain")
    if modality_ids is not None and (modality_ids.ndim != 1 or modality_ids.shape[0] != keys):
        raise ValueError("modality_ids must align with the complete key domain")

    device = teacher_q.device
    tensors = (teacher_k, teacher_v, student_q, student_route, student_v)
    if any(t.device != device for t in tensors):
        raise ValueError("all attention diagnostic tensors must share a device")
    scale = head_dim ** -0.5 if scale is None else float(scale)

    tq = teacher_q.float().permute(1, 0, 2)
    sq = student_q.float().permute(1, 0, 2)
    neg_inf = torch.full((heads, queries), float("-inf"), device=device)
    mt = neg_inf.clone()
    ms = neg_inf.clone()
    zt = torch.zeros((heads, queries), dtype=torch.float32, device=device)
    zs = zt.clone()
    kl_num = zt.clone()
    out_t_num = torch.zeros((heads, queries, head_dim), dtype=torch.float32, device=device)
    out_s_num = torch.zeros_like(out_t_num)

    count = torch.zeros((heads, queries), dtype=torch.float64, device=device)
    sum_t = torch.zeros_like(count)
    sum_t2 = torch.zeros_like(count)
    sum_err = torch.zeros_like(count)
    sum_err2 = torch.zeros_like(count)

    modality_values: tuple[int, ...] = ()
    mass_t = mass_s = None
    if modality_ids is not None:
        modality_values = tuple(int(x) for x in torch.unique(modality_ids.detach().cpu(), sorted=True).tolist())
        mass_t = torch.zeros((heads, queries, len(modality_values)), dtype=torch.float32, device=device)
        mass_s = torch.zeros_like(mass_t)
        modality_index = {value: i for i, value in enumerate(modality_values)}
    else:
        modality_index = {}

    for start in range(0, keys, key_chunk_size):
        stop = min(start + key_chunk_size, keys)
        tk = teacher_k[start:stop].float().permute(1, 0, 2)
        sk = student_route[start:stop].float().permute(1, 0, 2)
        tv = teacher_v[start:stop].float().permute(1, 0, 2)
        sv = student_v[start:stop].float().permute(1, 0, 2)
        lt = torch.einsum("hqd,hkd->hqk", tq, tk) * scale
        ls = torch.einsum("hqd,hkd->hqk", sq, sk) * scale
        mask_part = _mask_chunk(
            mask,
            heads=heads,
            queries=queries,
            start=start,
            stop=stop,
            device=device,
        )
        measure_part = None if log_measure is None else log_measure[start:stop].to(device=device, dtype=torch.float32)
        lt = _effective_logits(lt, mask_part, measure_part)
        ls = _effective_logits(ls, mask_part, measure_part)

        finite_t = torch.isfinite(lt)
        finite_s = torch.isfinite(ls)
        if not torch.equal(finite_t, finite_s):
            raise ValueError("teacher/student effective logit support differs")
        valid = finite_t
        ltd = torch.where(valid, lt, torch.zeros_like(lt)).double()
        lsd = torch.where(valid, ls, torch.zeros_like(ls)).double()
        errd = lsd - ltd
        count += valid.sum(dim=-1).double()
        sum_t += ltd.sum(dim=-1)
        sum_t2 += (ltd * ltd).sum(dim=-1)
        sum_err += errd.sum(dim=-1)
        sum_err2 += (errd * errd).sum(dim=-1)

        new_mt, scale_t, wt = _online_weights(lt, mt)
        new_ms, scale_s, ws = _online_weights(ls, ms)
        zt = zt * scale_t + wt.sum(dim=-1)
        zs = zs * scale_s + ws.sum(dim=-1)
        safe_diff = torch.where(finite_t, lt - ls, torch.zeros_like(lt))
        kl_num = kl_num * scale_t + (wt * safe_diff).sum(dim=-1)
        out_t_num = out_t_num * scale_t.unsqueeze(-1) + torch.einsum("hqk,hkd->hqd", wt, tv)
        out_s_num = out_s_num * scale_s.unsqueeze(-1) + torch.einsum("hqk,hkd->hqd", ws, sv)

        if modality_ids is not None:
            ids = modality_ids[start:stop].detach().cpu().tolist()
            one_hot = torch.zeros((stop - start, len(modality_values)), dtype=torch.float32, device=device)
            for row, value in enumerate(ids):
                one_hot[row, modality_index[int(value)]] = 1.0
            assert mass_t is not None and mass_s is not None
            mass_t = mass_t * scale_t.unsqueeze(-1) + torch.einsum("hqk,km->hqm", wt, one_hot)
            mass_s = mass_s * scale_s.unsqueeze(-1) + torch.einsum("hqk,km->hqm", ws, one_hot)

        mt, ms = new_mt, new_ms

    if (count == 0).any() or (zt <= 0).any() or (zs <= 0).any():
        raise ValueError("attention diagnostics encountered an all-masked query/head row")

    mean_t = sum_t / count
    mean_err = sum_err / count
    teacher_var = (sum_t2 / count - mean_t.square()).clamp_min(0.0)
    error_var = (sum_err2 / count - mean_err.square()).clamp_min(0.0)
    teacher_rms = teacher_var.sqrt()
    error_rms = error_var.sqrt()
    centered_nrmse = torch.where(
        teacher_rms > 0,
        error_rms / teacher_rms,
        torch.where(error_rms == 0, torch.zeros_like(error_rms), torch.full_like(error_rms, float("inf"))),
    )

    logzt = mt + torch.log(zt)
    logzs = ms + torch.log(zs)
    kl = kl_num / zt + logzs - logzt
    teacher_out = out_t_num / zt.unsqueeze(-1)
    student_out = out_s_num / zs.unsqueeze(-1)
    out_error_rms = (student_out - teacher_out).square().mean(dim=-1).sqrt()
    out_teacher_rms = teacher_out.square().mean(dim=-1).sqrt()
    out_nrmse = torch.where(
        out_teacher_rms > 0,
        out_error_rms / out_teacher_rms,
        torch.where(
            out_error_rms == 0,
            torch.zeros_like(out_error_rms),
            torch.full_like(out_error_rms, float("inf")),
        ),
    )
    out_cosine = F.cosine_similarity(teacher_out, student_out, dim=-1)

    teacher_mass = student_mass = None
    if mass_t is not None and mass_s is not None:
        teacher_mass = (mass_t / zt.unsqueeze(-1)).permute(1, 0, 2)
        student_mass = (mass_s / zs.unsqueeze(-1)).permute(1, 0, 2)

    return StreamedAttentionComparison(
        centered_logit_nrmse=centered_nrmse.permute(1, 0),
        softmax_kl_teacher_student=kl.permute(1, 0),
        output_nrmse=out_nrmse.permute(1, 0),
        output_cosine=out_cosine.permute(1, 0),
        teacher_output=teacher_out.permute(1, 0, 2),
        student_output=student_out.permute(1, 0, 2),
        modality_values=modality_values,
        teacher_modality_mass=teacher_mass,
        student_modality_mass=student_mass,
    )
