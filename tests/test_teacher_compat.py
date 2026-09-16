from __future__ import annotations

import pytest

from minimax_h3_keyless.checkpoint import TensorSignature
from minimax_h3_keyless.teacher_compat import (
    TeacherCompatibilityError,
    exact_copy_shared_keys,
    mutable_core_shared_keys,
    validate_deploy_signatures_against_teacher,
)


def _teacher(core_blocks=2, hidden=12, inner=8, head_dim=4):
    tensors = {
        "condition_proj.weight": TensorSignature((hidden, 6), "BF16"),
        "rope.inv_freq": TensorSignature((16,), "F32"),
        "final_layer.video_out.weight": TensorSignature((7, hidden), "BF16"),
    }
    for i in range(core_blocks):
        p = f"blocks.{i}.attn."
        tensors[p + "qkv_proj.weight"] = TensorSignature((3 * inner, hidden), "BF16")
        tensors[p + "q_norm.weight"] = TensorSignature((head_dim,), "BF16")
        tensors[p + "k_norm.weight"] = TensorSignature((head_dim,), "BF16")
        tensors[p + "out_proj.weight"] = TensorSignature((hidden, inner), "BF16")
        tensors[p + "to_gate_compress.weight"] = TensorSignature((inner, hidden), "BF16")
        tensors[f"blocks.{i}.mlp.fc1.weight"] = TensorSignature((4 * inner, hidden), "BF16")
    return tensors


def _deploy_from_teacher(teacher, core_blocks=2, hidden=12, inner=8, head_dim=4):
    deploy = dict(teacher)
    for i in range(core_blocks):
        p = f"blocks.{i}.attn."
        deploy.pop(p + "qkv_proj.weight")
        deploy.pop(p + "k_norm.weight")
        deploy[p + "qv_proj.weight"] = TensorSignature((2 * inner, hidden), "BF16")
        deploy[p + "route_norm.weight"] = TensorSignature((head_dim,), "BF16")
    return deploy


def test_complete_signature_transform_accepts_only_qkv_k_to_qv_route_change() -> None:
    teacher = _teacher()
    deploy = _deploy_from_teacher(teacher)
    exact, mutable = validate_deploy_signatures_against_teacher(
        teacher,
        deploy,
        core_blocks=2,
        hidden_size=12,
        inner_dim=8,
        head_dim=4,
    )
    assert "condition_proj.weight" in exact
    assert "blocks.0.mlp.fc1.weight" in exact
    assert "blocks.0.attn.to_gate_compress.weight" in exact
    assert "blocks.0.attn.q_norm.weight" in mutable
    assert "blocks.0.attn.out_proj.weight" in mutable
    assert "blocks.0.attn.q_norm.weight" not in exact


def test_complete_signature_transform_rejects_unrelated_shape_drift() -> None:
    teacher = _teacher()
    deploy = _deploy_from_teacher(teacher)
    deploy["condition_proj.weight"] = TensorSignature((11, 6), "BF16")
    with pytest.raises(TeacherCompatibilityError, match="shared tensor signature changed"):
        validate_deploy_signatures_against_teacher(
            teacher,
            deploy,
            core_blocks=2,
            hidden_size=12,
            inner_dim=8,
            head_dim=4,
        )


def test_complete_signature_transform_rejects_stray_tensor() -> None:
    teacher = _teacher()
    deploy = _deploy_from_teacher(teacher)
    deploy["blocks.0.attn.fake_k.weight"] = TensorSignature((8, 12), "BF16")
    with pytest.raises(TeacherCompatibilityError, match="exact core50 native->Keyless transformation"):
        validate_deploy_signatures_against_teacher(
            teacher,
            deploy,
            core_blocks=2,
            hidden_size=12,
            inner_dim=8,
            head_dim=4,
        )


def test_exact_copy_set_excludes_only_removed_and_design_mutable_core_tensors() -> None:
    teacher = _teacher()
    exact = exact_copy_shared_keys(set(teacher), core_blocks=2)
    mutable = mutable_core_shared_keys(2)
    assert not any(key.endswith("qkv_proj.weight") for key in exact)
    assert not any(key.endswith("k_norm.weight") for key in exact)
    assert exact.isdisjoint(mutable)
    assert "final_layer.video_out.weight" in exact
