# Stage-A Native-BF16 Capture and Pilot Runbook

This runbook covers the first empirical gate in `KEYLESS_H3_IMPLEMENTATION_DESIGN.md`: native MiniMax H3 block-local Keyless pilots at diffusion blocks 0, 25 and 49. It does **not** authorize the 50-block progressive sweep, BF16 release export, INT8 ConvRot release export, or any quality/speed claim. Those remain gated on empirical Stage-A evidence and the later validation stages in the design.

## 1. Fixed inputs

Prepare these inputs before capturing or training:

- the exact pinned BF16 teacher checkpoint identified by `minimax_h3_keyless.contracts.TEACHER_SHA256`;
- a fixed Stage-A dataset manifest accepted by `validate_pilot_dataset_manifest`;
- a gate manifest whose Stage-A thresholds were chosen from calibration evidence before the pilot result exists;
- a fixed Stage-A train-plan JSON accepted by `load_stage_a_run_plan`.

The production Stage-A dataset validator requires at least 16 cases, at least 8 distinct sigma strata across the corpus, both `train` and `holdout` cases, and coverage tags `short`, `long`, `reference`, `audio`, and `mixed-grid`. The manifest is an experiment definition, not a convenience list: prompt, seed, schedule, modality, resolution, duration, sigma strata and assets are part of the fixed evidence identity.

Train/holdout isolation is by complete case **and asset content**. Case IDs must be unique. An asset SHA-256 may be reused by multiple cases inside the same split, but the same asset bytes may not appear in both `train` and `holdout`, even under different paths or URIs. This prevents reference/audio/video assets from leaking across the empirical gate under renamed case IDs.

Do not change dataset, gate, or train-plan inputs after seeing pilot outcomes and continue under the same run identity. Use a new experiment/run identity instead.

## 2. Capture under plain native H3

Use the ComfyUI node **MiniMax H3 Stage-A BF16 Teacher Loader**. It accepts only the exact pinned BF16 native H3 checkpoint. The loader verifies the full file SHA-256 and native model topology and rejects quantized, Keyless, mixed-QKV/QV and non-BF16 teachers.

Route its `MODEL` output through **MiniMax H3 Stage-A Capture** before the normal sampler. Configure that capture node with:

- `dataset_manifest_path`: the fixed dataset manifest;
- `case_id`: the exact case currently being executed;
- `target_sigma`: one sigma explicitly declared for that case;
- `output_subdir`: a relative path below the Comfy output directory;
- `max_capture_mib`: an explicit per-forward CPU capture budget.

Run the real workflow for that exact manifest case. The workflow itself is responsible for reproducing the manifest prompt, seed, schedule, media/reference inputs, duration and resolution. The capture node does not synthesize those inputs from the manifest.

At the target video sigma, the capture wrapper records the live inputs required to replay blocks 0, 25 and 49. It records the block input, actual post-AdaLN attention input, timestep embedding/modulation state, RoPE/layout state and audit context. Capture tensors are copied to CPU and bounded by `max_capture_mib`.

### Fail-closed capture conditions

A Stage-A capture is rejected rather than silently accepted when any of the following is detected:

- the model did not originate from the strict Stage-A BF16 teacher loader;
- the live native H3 topology no longer matches the pinned teacher contract;
- model patches, object patches, weight wrappers, injections, hooks or callbacks are active;
- another `DIFFUSION_MODEL` wrapper is installed;
- a Keyless provider or optimized-attention override is active;
- the target H3 forward is re-entrant or executes more than once for the same capture controller;
- the observed runtime video sigma does not correspond to the requested manifest sigma;
- the capture byte budget is exceeded;
- the target bundle or receipt path is already occupied.

Spectrum, SOL, VDN, Flow and other execution-changing integrations are compatibility distributions for later validation. They are not the Stage-A native-teacher definition.

### Source provenance

Live capture requires clean Git working trees for both MiniMax-H3-Keyless and ComfyUI. Their full commit IDs are written into the capture provenance. A dirty tracked working tree is rejected.

All captures used in one registry must have the same teacher identity, MiniMax-H3-Keyless capture commit, ComfyUI commit and execution descriptor.

## 3. Capture every manifest case/sigma execution

Create one immutable capture bundle/receipt pair for every `(case_id, sigma)` declared by the fixed dataset manifest. Each bundle must contain exactly the ordered Stage-A blocks `(0, 25, 49)`.

A capture node only writes evidence if the sampler actually reaches `target_sigma`. If no bundle appears, first verify that the workflow schedule reaches the declared sigma; do not relabel a nearby executed sigma as the requested stratum.

Do not overwrite a capture to “retry” it under the same artifact identity. Keep failed/obsolete evidence separate and produce a new path/experiment identity.

## 4. Build the immutable capture registry

After all bundle/receipt pairs exist, build the registry:

```bash
python tools/build_stage_a_capture_registry.py \
  --dataset-manifest /path/to/stage_a_dataset.json \
  --capture-dir /path/to/comfy/output/keyless_stage_a \
  --output /path/to/stage_a_capture_registry.json
```

The builder recursively finds `*.capture.pt.receipt.json` by default. It hash-checks every receipt and bundle, validates serialized capture records one artifact at a time, checks exact block/case/sigma/modality provenance, rejects mixed source provenance, and requires exact coverage of the manifest case×sigma corpus. It does not retain the complete activation corpus in memory while building the registry.

The emitted registry stores relative bundle/receipt paths plus each immutable receipt SHA-256. Moving the registry together with its capture tree preserves those relative references.

## 5. Run the Stage-A block pilots

Run the campaign from the clean MiniMax-H3-Keyless revision you intend to record:

```bash
python tools/run_stage_a_pilots.py \
  --teacher /path/to/pinned_bf16_teacher.safetensors \
  --dataset-manifest /path/to/stage_a_dataset.json \
  --capture-registry /path/to/stage_a_capture_registry.json \
  --gate-manifest /path/to/stage_a_gate_manifest.json \
  --train-plan /path/to/stage_a_train_plan.json \
  --output-dir /path/to/stage_a_results \
  --run-id stage-a-001 \
  --code-commit "$(git rev-parse HEAD)" \
  --device cuda:0
```

The runner requires the claimed `--code-commit` to equal the clean MiniMax-H3-Keyless source actually executing the campaign. It also requires the current clean ComfyUI revision to equal the ComfyUI revision that produced the capture corpus.

The runner indexes and hash-validates the full capture corpus without loading all activation tensors. For each pilot depth it materializes only that block's train/holdout records, then releases them when the block run ends. This avoids keeping captures for blocks 0, 25 and 49 resident simultaneously.

For each block, the Stage-A v3 runner:

1. replays every train and holdout capture through the frozen native teacher block and proves the recorded post-AdaLN attention input is reproduced within the predeclared tolerance;
2. derives the regularized-LS route fits from **training captures only**, evaluates identity initialization plus the fixed LS lambda-relative grid `0`, `1e-4`, `1e-2` on complete **training cases**, and records the bounded attention diagnostics for those training cases;
3. selects the candidate initialization and the best LS lambda from that training evidence only, with deterministic tie breaking;
4. evaluates the untrained identity baseline and the **train-selected LS baseline** on the untouched complete-case holdout;
5. trains only the stages present in the fixed train plan, in monotonic `route -> query -> value -> norm_out` order, using training cases only;
6. evaluates the trained candidate and its attention diagnostics on that same untouched holdout;
7. applies the predeclared Stage-A numerical gate to the held-out candidate versus the two held-out untrained baselines;
8. writes hash-bound v3 block result/resume evidence, including `selection_split=train_complete_cases` and the train-selected LS lambda, without relabeling a failed gate as success.

The same-input local objective uses the actual copied H3 block. Non-attention block weights remain frozen. The native teacher block executes under `no_grad`; the Keyless student receives the same block input and must reproduce the same post-AdaLN attention input before its loss is accepted.

The holdout is an exit-gate dataset, not a hyperparameter-selection dataset. Do not inspect it to choose identity versus LS, choose the LS lambda, change loss weights, change the train plan, or tune thresholds under the same experiment identity.

## 6. Resume an interrupted campaign

If one or more complete block artifacts already exist for the same immutable experiment context, rerun with:

```bash
python tools/run_stage_a_pilots.py \
  ...same arguments... \
  --resume-completed
```

A completed block is reused only after its checkpoint/result hashes, experiment context, v3 selection-split marker, initialization selection, train-selected LS lambda, numerical evidence and recomputed gate validate. The v3 loader also requires the initialization rows to describe one training case set and rejects overlap between those training-selection case IDs and the held-out candidate case IDs. A partial checkpoint/result pair is an error. Changing dataset, registry, gate manifest, train-plan identity or capture provenance changes the experiment context and prevents silent reuse.

Trusted-local `torch.save` resume checkpoints contain optimizer/RNG state and are loaded with `weights_only=False`; do not use untrusted resume files.

## 7. Interpreting the exit

A campaign passes Stage A only when the predeclared gate passes for all three prescribed depths: 0, 25 and 49. CPU unit tests and structural CI establish implementation contracts only; they do not establish H3 output parity.

If Stage A fails, retain the immutable evidence and diagnose the failure. Do not reuse the holdout to pick a different initialization or LS lambda, do not loosen gate thresholds after observing the result, and do not proceed to the 50-block progressive conversion while calling the failed pilot accepted. Any changed dataset, training recipe, architecture escalation, or thresholds require a new experiment identity and fresh evidence.

If Stage A passes, the next design stage is the progressive core50 sweep on live student inputs. That stage must locally compare each frozen original QKV block against the Keyless replacement on the same current hidden input and must retain rollback/checkpoint granularity. The Stage-A capture corpus is not a substitute for that live-input progressive procedure.
