from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest
import torch
import torch.nn as nn

from minimax_h3_keyless.attention import KeylessAttentionTrain
from minimax_h3_keyless.contracts import TARGET_MODEL_REVISION, TEACHER_SHA256
from minimax_h3_keyless.initialization import initialize_training_attention_from_native
from minimax_h3_keyless.ops import normalized_positioned, torch_sdpa_attention
from minimax_h3_keyless.pilot import PilotCase, set_pilot_block_stage
from minimax_h3_keyless.pilot_campaign import (
    DATASET_SCHEMA,
    GATE_SCHEMA,
    PILOT_LS_LAMBDAS,
    PilotRunIdentity,
    canonical_json_sha256,
    evaluate_pilot_cases,
    load_pilot_resume_checkpoint,
    save_pilot_resume_checkpoint,
    train_pilot_stage,
    validate_optimizer_matches_trainable,
    validate_pilot_dataset_manifest,
    validate_pilot_gate_manifest,
    write_json_atomic,
)


class TinyNativeAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.heads = 2
        self.head_dim = 2
        self.qkv_proj = nn.Linear(4, 12, bias=False)
        self.q_norm = nn.RMSNorm(2, eps=1e-5)
        self.k_norm = nn.RMSNorm(2, eps=1e-5)
        self.out_proj = nn.Linear(4, 4, bias=False)
        self.to_gate_compress = None

    def forward(self, x, rope_freqs=None, transformer_options=None):
        q, k, v = self.qkv_proj(x).split(4, dim=-1)
        q = normalized_positioned(q.view(-1, 2, 2), self.q_norm.weight, self.q_norm.eps, rope_freqs)
        k = normalized_positioned(k.view(-1, 2, 2), self.k_norm.weight, self.k_norm.eps, rope_freqs)
        v = v.view(-1, 2, 2)
        out = torch_sdpa_attention(q, k, v, scale=2 ** -0.5).reshape(-1, 4)
        return self.out_proj(out)


class TinyBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.norm1 = nn.RMSNorm(4, eps=1e-5)
        self.attn = TinyNativeAttention()
        self.mlp = nn.Linear(4, 4, bias=False)

    def forward(self, x, t_emb, mod_segments, rope_freqs, transformer_options=None):
        h = self.norm1(x)
        x.add_(self.attn(h, rope_freqs=rope_freqs, transformer_options=transformer_options))
        return x + 0.01 * self.mlp(x)


def _pair() -> tuple[TinyBlock, TinyBlock]:
    torch.manual_seed(401)
    teacher = TinyBlock()
    student = copy.deepcopy(teacher)
    native = teacher.attn
    keyless = KeylessAttentionTrain(4, 2, 2, 1e-5, block_index=0, dtype=torch.float32)
    initialize_training_attention_from_native(
        keyless,
        qkv_weight=native.qkv_proj.weight,
        q_norm_weight=native.q_norm.weight,
        k_norm_weight=native.k_norm.weight,
        out_proj_weight=native.out_proj.weight,
    )
    student.attn = keyless
    set_pilot_block_stage(student, "route")
    return teacher, student


def _cases() -> list[PilotCase]:
    torch.manual_seed(402)
    return [
        PilotCase(
            x=torch.randn(5, 4),
            t_emb=torch.randn(1, 2),
            mod_segments=((0, 5, 0),),
            rope_freqs=None,
            case_id="case-a",
            sigma=0.2,
            modality_label="video",
        ),
        PilotCase(
            x=torch.randn(6, 4),
            t_emb=torch.randn(1, 2),
            mod_segments=((0, 6, 0),),
            rope_freqs=None,
            case_id="case-b",
            sigma=0.8,
            modality_label="audio-video",
        ),
    ]


def _dataset_manifest() -> dict:
    cases = []
    for i in range(16):
        cases.append(
            {
                "case_id": f"case-{i}",
                "split": "train" if i < 12 else "holdout",
                "prompt": f"prompt {i}",
                "seed": 1000 + i,
                "schedule": {"name": "fixed-test"},
                "modality_label": "t2v" if i % 2 == 0 else "av",
                "resolution": [480, 672] if i < 8 else [704, 960],
                "duration_seconds": 5.0 + i,
                "sigmas": [j / 7 for j in range(8)],
                "coverage_tags": ["native", "low" if i < 8 else "high"],
                "assets": [] if i % 2 == 0 else [
                    {"path_or_uri": "asset.wav", "sha256": "1" * 64}
                ],
            }
        )
    return {"schema": DATASET_SCHEMA, "cases": cases}


def _identity() -> PilotRunIdentity:
    return PilotRunIdentity(
        run_id="resume-test",
        code_commit="deadbeef",
        dataset_manifest_sha256="a" * 64,
        gate_manifest_sha256="b" * 64,
        block_index=0,
        route_mode="identity",
        lambda_relative=0.0,
    )


def test_dataset_manifest_freezes_complete_case_split_and_sigma_coverage() -> None:
    manifest = _dataset_manifest()
    identity = validate_pilot_dataset_manifest(
        manifest, required_coverage_tags=("native", "low", "high")
    )
    assert identity == canonical_json_sha256(manifest)

    bad = json.loads(json.dumps(manifest))
    bad["cases"][-1]["case_id"] = bad["cases"][0]["case_id"]
    with pytest.raises(ValueError, match="duplicate"):
        validate_pilot_dataset_manifest(bad)

    bad = json.loads(json.dumps(manifest))
    for case in bad["cases"]:
        case["split"] = "train"
    with pytest.raises(ValueError, match="train and holdout"):
        validate_pilot_dataset_manifest(bad)


def test_gate_manifest_requires_predeclared_thresholds_and_calibration() -> None:
    manifest = {
        "schema": GATE_SCHEMA,
        "thresholds": {"block_output_warning": 0.01, "block_output_hard": 0.02},
        "calibration_evidence": {"teacher_repeatability": "pending-fixed-suite"},
    }
    assert validate_pilot_gate_manifest(manifest) == canonical_json_sha256(manifest)
    with pytest.raises(ValueError, match="predeclared thresholds"):
        validate_pilot_gate_manifest({"schema": GATE_SCHEMA, "thresholds": {}})


def test_pilot_run_identity_is_pinned_to_teacher_depth_and_ls_grid() -> None:
    identity = PilotRunIdentity(
        run_id="pilot-0-ls",
        code_commit="deadbeef",
        dataset_manifest_sha256="a" * 64,
        gate_manifest_sha256="b" * 64,
        block_index=25,
        route_mode="least_squares",
        lambda_relative=PILOT_LS_LAMBDAS[1],
    )
    assert identity.teacher_model_revision == TARGET_MODEL_REVISION
    assert identity.teacher_model_sha256 == TEACHER_SHA256
    with pytest.raises(ValueError, match="one of"):
        PilotRunIdentity(
            run_id="bad",
            code_commit="deadbeef",
            dataset_manifest_sha256="a" * 64,
            gate_manifest_sha256="b" * 64,
            block_index=25,
            route_mode="least_squares",
            lambda_relative=0.5,
        )


def test_holdout_evaluation_is_case_level_and_restores_student_mode() -> None:
    teacher, student = _pair()
    student.train(True)
    report = evaluate_pilot_cases(teacher, student, _cases())
    assert report.case_count == 2
    assert set(report.by_modality) == {"audio-video", "video"}
    assert report.by_modality["video"]["case_count"] == 1
    assert len(report.cases) == 2
    assert report.worst_block_normalized_mse >= report.mean_block_normalized_mse
    assert student.training is True


def test_optimizer_must_be_rebuilt_after_stage_transition() -> None:
    teacher, student = _pair()
    route_optimizer = torch.optim.AdamW(
        [p for p in student.parameters() if p.requires_grad], lr=1e-3
    )
    validate_optimizer_matches_trainable(student, route_optimizer)
    set_pilot_block_stage(student, "value")
    with pytest.raises(RuntimeError, match="Rebuild the optimizer"):
        validate_optimizer_matches_trainable(student, route_optimizer)

    value_optimizer = torch.optim.AdamW(
        [p for p in student.parameters() if p.requires_grad], lr=1e-3
    )
    events = train_pilot_stage(
        teacher,
        student,
        _cases()[:1],
        value_optimizer,
        stage="value",
        epochs=1,
        max_grad_norm=10.0,
    )
    assert len(events) == 1
    assert events[0].stage == "value"
    assert events[0].case_id == "case-a"
    assert events[0].report.gradient_l2_norm > 0


def test_stage_argument_cannot_lie_about_optimizer_or_resume_artifact(tmp_path: Path) -> None:
    _, student = _pair()
    route_optimizer = torch.optim.AdamW(
        [p for p in student.parameters() if p.requires_grad], lr=1e-3
    )
    with pytest.raises(RuntimeError, match="Rebuild the optimizer"):
        save_pilot_resume_checkpoint(
            tmp_path / "bad-stage.pt",
            student_block=student,
            optimizer=route_optimizer,
            identity=_identity(),
            stage="value",
            step=0,
        )
    assert not (tmp_path / "bad-stage.pt").exists()


def test_resume_checkpoint_binds_run_identity_optimizer_stage_and_rng(tmp_path: Path) -> None:
    _, student = _pair()
    optimizer = torch.optim.AdamW(
        [p for p in student.parameters() if p.requires_grad], lr=1e-3
    )
    identity = _identity()
    path = tmp_path / "resume.pt"
    before = student.attn.query_route.weight.detach().clone()
    checkpoint_sha = save_pilot_resume_checkpoint(
        path,
        student_block=student,
        optimizer=optimizer,
        identity=identity,
        stage="route",
        step=7,
        extra={"note": "bounded test"},
    )
    with pytest.raises(RuntimeError, match="does not match requested stage"):
        load_pilot_resume_checkpoint(
            path,
            student_block=student,
            optimizer=optimizer,
            expected_identity=identity,
            expected_stage="value",
            restore_rng=False,
        )
    with torch.no_grad():
        student.attn.query_route.weight.add_(1.0)
    loaded = load_pilot_resume_checkpoint(
        path,
        student_block=student,
        optimizer=optimizer,
        expected_identity=identity,
        expected_stage="route",
        restore_rng=False,
    )
    torch.testing.assert_close(student.attn.query_route.weight, before)
    assert loaded["step"] == 7
    assert loaded["stage"] == "route"
    assert loaded["extra"] == {"note": "bounded test"}
    assert loaded["checkpoint_sha256"] == checkpoint_sha


def test_atomic_json_writer_records_a_hashable_audit_artifact(tmp_path: Path) -> None:
    path = tmp_path / "metrics.json"
    digest = write_json_atomic(path, {"schema": "test", "value": 3})
    assert len(digest) == 64
    assert json.loads(path.read_text(encoding="utf-8")) == {"schema": "test", "value": 3}
