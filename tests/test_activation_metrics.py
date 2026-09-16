from __future__ import annotations

import torch
import torch.nn.functional as F

from minimax_h3_keyless.activation_metrics import (
    streamed_attention_comparison,
    tensor_error_metrics,
)


def _direct(q, k, v, *, scale, mask, log_measure):
    logits = torch.einsum("qhd,khd->hqk", q.float(), k.float()) * scale
    if mask is not None:
        if mask.dtype == torch.bool:
            logits = logits.masked_fill(~mask.unsqueeze(0), float("-inf"))
        else:
            logits = logits + mask.unsqueeze(0)
    if log_measure is not None:
        logits = logits + log_measure.view(1, 1, -1)
    probs = torch.softmax(logits, dim=-1)
    out = torch.einsum("hqk,khd->qhd", probs, v.float())
    return logits, probs, out


def test_streamed_metrics_match_dense_complete_key_calculation() -> None:
    torch.manual_seed(41)
    q_rows, k_rows, heads, dim = 3, 7, 2, 4
    tq = torch.randn(q_rows, heads, dim)
    tk = torch.randn(k_rows, heads, dim)
    tv = torch.randn(k_rows, heads, dim)
    sq = tq + 0.1 * torch.randn_like(tq)
    sk = tk + 0.1 * torch.randn_like(tk)
    sv = tv + 0.1 * torch.randn_like(tv)
    mask = torch.tensor(
        [
            [True, True, False, True, True, True, True],
            [True, False, True, True, True, True, True],
            [True, True, True, True, False, True, True],
        ]
    )
    log_measure = torch.log(torch.tensor([1.0, 2.0, 0.5, 1.5, 3.0, 1.0, 0.75]))
    modality_ids = torch.tensor([0, 0, 1, 1, 2, 2, 2])
    scale = 0.31

    result = streamed_attention_comparison(
        tq,
        tk,
        tv,
        sq,
        sk,
        sv,
        scale=scale,
        mask=mask,
        log_measure=log_measure,
        modality_ids=modality_ids,
        key_chunk_size=2,
    )
    lt, pt, ot = _direct(tq, tk, tv, scale=scale, mask=mask, log_measure=log_measure)
    ls, ps, os = _direct(sq, sk, sv, scale=scale, mask=mask, log_measure=log_measure)
    valid = torch.isfinite(lt)
    centered_t = torch.where(valid, lt, torch.zeros_like(lt))
    centered_s = torch.where(valid, ls, torch.zeros_like(ls))
    counts = valid.sum(-1)
    mt = centered_t.sum(-1) / counts
    ms = centered_s.sum(-1) / counts
    centered_t = torch.where(valid, lt - mt.unsqueeze(-1), torch.zeros_like(lt))
    centered_s = torch.where(valid, ls - ms.unsqueeze(-1), torch.zeros_like(ls))
    expected_logit = (
        ((centered_s - centered_t).square().sum(-1) / counts).sqrt()
        / (centered_t.square().sum(-1) / counts).sqrt()
    ).permute(1, 0)
    safe_log_pt = torch.where(pt > 0, torch.log(pt), torch.zeros_like(pt))
    safe_log_ps = torch.where(pt > 0, torch.log(ps), torch.zeros_like(ps))
    expected_kl = (pt * (safe_log_pt - safe_log_ps)).sum(-1).permute(1, 0)
    expected_nrmse = (
        (os - ot).square().mean(-1).sqrt() / ot.square().mean(-1).sqrt()
    )
    expected_cos = F.cosine_similarity(ot, os, dim=-1)

    torch.testing.assert_close(result.centered_logit_nrmse, expected_logit.double(), atol=2e-5, rtol=2e-5)
    torch.testing.assert_close(result.softmax_kl_teacher_student, expected_kl, atol=2e-5, rtol=2e-5)
    torch.testing.assert_close(result.teacher_output, ot, atol=2e-5, rtol=2e-5)
    torch.testing.assert_close(result.student_output, os, atol=2e-5, rtol=2e-5)
    torch.testing.assert_close(result.output_nrmse, expected_nrmse, atol=2e-5, rtol=2e-5)
    torch.testing.assert_close(result.output_cosine, expected_cos, atol=2e-5, rtol=2e-5)

    assert result.modality_values == (0, 1, 2)
    expected_teacher_mass = []
    expected_student_mass = []
    for value in result.modality_values:
        selection = modality_ids == value
        expected_teacher_mass.append(pt[..., selection].sum(-1).permute(1, 0))
        expected_student_mass.append(ps[..., selection].sum(-1).permute(1, 0))
    expected_teacher_mass = torch.stack(expected_teacher_mass, dim=-1)
    expected_student_mass = torch.stack(expected_student_mass, dim=-1)
    torch.testing.assert_close(result.teacher_modality_mass, expected_teacher_mass, atol=2e-5, rtol=2e-5)
    torch.testing.assert_close(result.student_modality_mass, expected_student_mass, atol=2e-5, rtol=2e-5)


def test_streamed_metrics_reject_all_masked_query() -> None:
    q = torch.randn(1, 1, 2)
    k = torch.randn(2, 1, 2)
    v = torch.randn(2, 1, 2)
    try:
        streamed_attention_comparison(q, k, v, q, k, v, mask=torch.zeros(1, 2, dtype=torch.bool))
    except ValueError as exc:
        assert "all-masked" in str(exc)
    else:
        raise AssertionError("all-masked rows must be rejected")


def test_tensor_error_metrics_identity() -> None:
    x = torch.randn(4, 3, 5)
    nrmse, cosine = tensor_error_metrics(x, x.clone())
    torch.testing.assert_close(nrmse, torch.tensor(0.0))
    torch.testing.assert_close(cosine, torch.tensor(1.0), atol=1e-6, rtol=0.0)
