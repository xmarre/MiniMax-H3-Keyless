from __future__ import annotations

from dataclasses import replace

import torch
import torch.nn as nn

from minimax_h3_keyless.activation_capture import CapturedPilotCase
from minimax_h3_keyless.pilot import PilotCase
from minimax_h3_keyless.route_fit import (
    collect_route_activation_statistics,
    solve_route_activation_fit,
)


class TinyNativeAttention(nn.Module):
    def __init__(self, hidden: int = 6, heads: int = 2, head_dim: int = 2):
        super().__init__()
        self.heads = heads
        self.head_dim = head_dim
        self.qkv_proj = nn.Linear(hidden, 3 * heads * head_dim, bias=False, dtype=torch.bfloat16)


def _records(inputs: tuple[torch.Tensor, ...]) -> tuple[CapturedPilotCase, ...]:
    rows = []
    for index, attention_input in enumerate(inputs):
        case = PilotCase(
            x=attention_input.clone(),
            t_emb=torch.zeros(1, 1),
            mod_segments=((0, attention_input.shape[0], 0),),
            rope_freqs=None,
            transformer_options={},
            case_id=f"train-{index}",
            sigma=0.25 + 0.25 * index,
            modality_label="video",
            context={"stage_a_split": "train"},
        )
        rows.append(
            CapturedPilotCase(
                block_index=0,
                case=case,
                attention_input=attention_input.clone(),
                captured_bytes=attention_input.numel() * attention_input.element_size(),
            )
        )
    return tuple(rows)


def test_activation_ls_recovers_per_head_storage_route_from_train_rows() -> None:
    torch.manual_seed(701)
    hidden, heads, head_dim = 6, 2, 2
    inner = heads * head_dim
    attention = TinyNativeAttention(hidden, heads, head_dim)

    q = torch.randn(inner, hidden)
    v = torch.randn(inner, hidden)
    expected = torch.tensor(
        [
            [[1.25, -0.5], [0.2, 0.8]],
            [[0.6, 0.3], [-0.4, 1.1]],
        ],
        dtype=torch.float32,
    )
    k_heads = []
    for head in range(heads):
        a, b = head * head_dim, (head + 1) * head_dim
        # F.linear(x, v[a:b]) gives V.  K = V @ expected[head], so the
        # storage projection rows are expected.T @ v_rows.
        k_heads.append(expected[head].T @ v[a:b])
    k = torch.cat(k_heads, dim=0)
    with torch.no_grad():
        attention.qkv_proj.weight.copy_(torch.cat((q, k, v), dim=0).to(torch.bfloat16))

    inputs = (
        torch.randn(9, hidden, dtype=torch.bfloat16),
        torch.randn(7, hidden, dtype=torch.bfloat16),
    )
    stats = collect_route_activation_statistics(attention, _records(inputs), chunk_rows=4)
    fit = solve_route_activation_fit(stats, lambda_relative=0.0)

    assert stats.rows == 16
    assert stats.numerical_rank == (head_dim, head_dim)
    assert fit.diagnostics.lambda_actual == (0.0, 0.0)
    torch.testing.assert_close(
        fit.storage_weight.float(),
        expected,
        atol=2e-2,
        rtol=2e-2,
    )


def test_activation_route_statistics_depend_on_attention_input_not_block_input() -> None:
    torch.manual_seed(702)
    attention = TinyNativeAttention()
    records = _records((torch.randn(8, 6, dtype=torch.bfloat16),))
    first = collect_route_activation_statistics(attention, records, chunk_rows=3)

    changed_case_x = tuple(
        replace(record, case=replace(record.case, x=record.case.x + 1000.0))
        for record in records
    )
    second = collect_route_activation_statistics(attention, changed_case_x, chunk_rows=5)

    torch.testing.assert_close(first.gram, second.gram)
    torch.testing.assert_close(first.cross, second.cross)


def test_regularized_activation_fit_records_per_head_lambda_and_spectrum() -> None:
    torch.manual_seed(703)
    attention = TinyNativeAttention()
    records = _records((torch.randn(11, 6, dtype=torch.bfloat16),))
    stats = collect_route_activation_statistics(attention, records)
    fit = solve_route_activation_fit(stats, lambda_relative=1e-2)

    assert fit.diagnostics.rows == 11
    assert fit.diagnostics.lambda_relative == 1e-2
    assert len(fit.diagnostics.lambda_actual) == attention.heads
    assert all(value > 0.0 for value in fit.diagnostics.lambda_actual)
    assert len(fit.diagnostics.smallest_singular_value) == attention.heads
    assert len(fit.diagnostics.largest_singular_value) == attention.heads
    assert len(fit.diagnostics.numerical_rank) == attention.heads
    assert torch.isfinite(fit.storage_weight).all()
