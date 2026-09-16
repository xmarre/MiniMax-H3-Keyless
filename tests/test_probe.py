from __future__ import annotations

from pathlib import Path

import torch
from safetensors.torch import save_file

from minimax_h3_keyless.probe import (
    NativeHeadSlices,
    probe_native_head,
    read_native_qkv_head_slices,
    run_teacher_probe,
)


def test_head_slice_reader_uses_exact_q_k_v_row_ranges(tmp_path: Path) -> None:
    hidden, heads, head_dim = 5, 2, 3
    inner = heads * head_dim
    weight = torch.arange(3 * inner * hidden, dtype=torch.float32).reshape(3 * inner, hidden)
    path = tmp_path / "teacher.safetensors"
    save_file({"blocks.4.attn.qkv_proj.weight": weight}, str(path))
    slices = read_native_qkv_head_slices(
        path,
        layer=4,
        head=1,
        hidden_size=hidden,
        heads=heads,
        head_dim=head_dim,
    )
    offset = head_dim
    torch.testing.assert_close(slices.q_math, weight[offset:offset + head_dim].T)
    torch.testing.assert_close(slices.k_math, weight[inner + offset:inner + offset + head_dim].T)
    torch.testing.assert_close(slices.v_math, weight[2 * inner + offset:2 * inner + offset + head_dim].T)
    assert len(slices.q_sha256) == len(slices.k_sha256) == len(slices.v_sha256) == 64


def test_probe_reports_zero_residual_when_k_and_v_match() -> None:
    torch.manual_seed(51)
    q = torch.randn(7, 3)
    v = torch.randn(7, 3)
    slices = NativeHeadSlices(q, v, v.clone(), "q", "k", "v")
    result = probe_native_head(slices, layer=0, head=0)
    assert result.fixed_value_bilinear_relative_residual < 1e-10
    assert result.key_projection_relative_residual < 1e-10
    assert max(result.principal_angles_degrees_k_v) < 1e-5


def test_teacher_probe_can_explicitly_record_unverified_development_run(tmp_path: Path) -> None:
    # Tiny noncanonical file only exercises provenance behavior; canonical geometry is
    # deliberately tested by read_native_qkv_head_slices and production defaults.
    path = tmp_path / "empty.safetensors"
    save_file({"x": torch.ones(1)}, str(path))
    result = run_teacher_probe(
        path,
        layers=(),
        heads=(),
        expected_sha256="0" * 64,
        verify_full_sha256=False,
    )
    assert result["actual_sha256"] is None
    assert result["full_sha256_verified"] is False
    assert result["scope"] == "raw_weight_subspace_diagnostic_only"
