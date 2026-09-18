# Stage-B Progressive Core50 Runbook

This runbook covers Phase 4 of `KEYLESS_H3_IMPLEMENTATION_DESIGN.md`: early-to-late conversion of the 50 MiniMax H3 core transformer blocks after a passed Stage-A campaign. It does **not** establish generation-quality parity, production-kernel compatibility, INT8 ConvRot quality or speedup. Those remain later empirical gates.

Stage B is deliberately iterative. For target block `i`, the capture corpus must come from the exact accepted prefix `0..i-1`; the original native QKV block `i` is then trained/evaluated against the Keyless candidate on those same live hidden inputs. A failed candidate is persisted as evidence but does not advance the prefix.

Stage-B initialization/model-selection evidence and the Stage-B numerical exit gate are separate datasets within the fixed corpus. Identity-versus-LS initialization and the LS regularization choice are selected on complete **training** cases only. The fixed holdout is then used only to evaluate the two train-selected untrained baselines and the trained candidate for the frozen exit gate. Do not use the holdout to choose initialization, LS lambda, train stages or architecture escalation.

## 1. Freeze the complete policy before Stage A

The gate manifest used for Stage A is also the hash-bound Stage-B policy. In addition to all required `stage_a_*` thresholds, it must contain these predeclared non-negative finite thresholds before Stage-A evidence is produced:

```json
{
  "progressive_fold_atol": 0.0,
  "progressive_fold_rtol": 0.0
}
```

The values above are only syntax examples, not recommended tolerances. Choose them from calibration evidence before observing the full progressive sweep. Stage-B authorization rejects a gate manifest that does not already contain both values. Do not add or loosen them after Stage A under the same experiment identity: changing the gate manifest changes its hash and invalidates the authorized prefix.

Stage B also requires the exact Stage-A train-plan file. Both its canonical semantic identity and full file SHA-256 are checked against the passed Stage-A campaign result. Canonical v3 plans contain `route`, optional `query`, and optional `value` in monotonic order. They do **not** contain `norm_out`; copied `q_norm`/`out_proj` escalation remains deferred until a separately predeclared train-only calibration/plateau contract exists. The Stage-B exit holdout may not authorize that escalation either.

## 2. Authorize the empty progressive prefix

After Stage A passes, create the initial prefix from the same dataset and gate policy:

```bash
python tools/authorize_progressive_sweep.py \
  --stage-a-result /path/to/stage_a_results/stage-a-001.campaign.result.json \
  --dataset-manifest /path/to/stage_a_dataset.json \
  --gate-manifest /path/to/stage_a_gate_manifest.json \
  --output-dir /path/to/progressive_artifacts \
  --sweep-id keyless-core50-001
```

Authorization records the clean MiniMax-H3-Keyless Git revision as the fixed sweep source. The initial manifest is `keyless-core50-001.prefix-00.json`; it contains no accepted blocks. All later capture and training steps require the runtime repository to remain at that clean revision.

## 3. Capture the next block from the current accepted prefix

For every case and sigma in the fixed dataset, run the real H3 workflow through these ComfyUI nodes:

1. **MiniMax H3 Stage-A BF16 Teacher Loader** — loads only the pinned native BF16 teacher.
2. **MiniMax H3 Progressive Prefix Overlay** — reconstructs the accepted Keyless prefix from the current prefix manifest and prior progressive artifacts without mutating the shared native teacher.
3. **MiniMax H3 Progressive Capture** — captures the input to exactly `prefix.next_block` at the declared case/sigma execution.

The overlay artifact root must contain the immutable result/resume artifacts referenced by the accepted prefix. For prefix 00 there are none; after each accepted block, keep using the same progressive artifact directory.

The workflow itself must reproduce the fixed manifest prompt, seed, sampler schedule, reference/media inputs, resolution and duration. The capture node does not synthesize those inputs. Run one capture for every manifest `(case_id, sigma)` execution.

A progressive capture is fail-closed when, among other conditions, the accepted overlay is missing or inconsistent, the target is not the next native block, runtime mutation/wrappers violate the plain-capture contract, source trees are dirty or mismatched, the requested sigma is not reached, the forward is re-entrant, or an immutable output path already exists.

Do not reuse Stage-A block-0/25/49 activations here. Stage-B inputs must be regenerated after every accepted prefix advancement because earlier Keyless replacements change the hidden states seen by later blocks.

## 4. Build the immutable capture registry

After all executions for the current prefix exist:

```bash
python tools/build_progressive_capture_registry.py \
  --dataset-manifest /path/to/stage_a_dataset.json \
  --prefix-manifest /path/to/progressive_artifacts/keyless-core50-001.prefix-00.json \
  --capture-dir /path/to/comfy/output/keyless_progressive_block00 \
  --output /path/to/progressive_artifacts/block00.capture-registry.json
```

The builder requires exact case×sigma coverage. It validates every receipt and bundle one at a time, rejects mixed prefix/source/Comfy/teacher provenance, and binds the registry to both the current prefix identity and the full prefix-manifest SHA-256.

The production training path uses `load_progressive_block_capture_set_lazy`: indexing validates every artifact while retaining no activation tensors. Repeated replay/evaluation/training accesses reload one capture at a time. Device cases are also moved lazily, so the runner does not materialize the entire corpus on CUDA.

## 5. Train and conditionally accept exactly one block

Run the current target block:

```bash
python tools/run_progressive_block.py \
  --teacher /path/to/pinned_bf16_teacher.safetensors \
  --stage-a-result /path/to/stage_a_results/stage-a-001.campaign.result.json \
  --prefix-manifest /path/to/progressive_artifacts/keyless-core50-001.prefix-00.json \
  --capture-registry /path/to/progressive_artifacts/block00.capture-registry.json \
  --dataset-manifest /path/to/stage_a_dataset.json \
  --gate-manifest /path/to/stage_a_gate_manifest.json \
  --train-plan /path/to/stage_a_train_plan.json \
  --output-dir /path/to/progressive_artifacts \
  --device cuda:0
```

Before allocating the H3 model, the runner binds all immutable identities: passed Stage-A campaign, dataset, gate policy, Stage-A-authorized train plan, current prefix manifest, capture registry, next block and fixed source revision. At runtime it also requires clean MiniMax-H3-Keyless and ComfyUI revisions matching the progressive capture provenance.

The runner then:

1. loads the exact pinned native BF16 teacher;
2. reconstructs the already accepted prefix from immutable progressive artifacts;
3. leaves the current target block in its original native QKV form;
4. verifies native same-input replay for every current-prefix train and holdout capture;
5. computes bounded activation-derived least-squares route statistics on train captures only;
6. evaluates the fixed identity/least-squares initialization grid on complete **training** captures only, records `selection_split=train_complete_cases`, and selects identity versus LS plus the LS lambda from that training evidence;
7. evaluates the untrained identity baseline and the **train-selected LS baseline** on the untouched complete-case holdout;
8. trains the fixed canonical plan (`route -> query -> value`, limited to stages declared by the Stage-A v3 plan) on training captures only;
9. atomically replaces a mutable crash-recovery checkpoint after every **complete epoch** with q/R/v student state, current-stage optimizer state, Python/NumPy/Torch RNG state and prior training events;
10. evaluates the trained candidate on the same untouched holdout and recomputes the frozen numerical gate against the held-out identity and train-selected-LS baselines;
11. persists the immutable candidate result/resume pair regardless of pass/fail, including the train-only selection marker and train-selected LS-baseline lambda;
12. removes the mutable crash-recovery file only after that immutable candidate transaction succeeds;
13. only if the gate passes, independently revalidates the train-selection/holdout case separation, folds the training route into deploy Q, verifies fold parity using the predeclared `progressive_fold_atol/rtol`, installs the target block and atomically publishes the next hash-chained prefix manifest.

The current immutable Stage-B result/resume evidence schemas are v2. The mutable training-recovery schema is also v2. These versions deliberately reject pre-isolation v1 artifacts/resume files, because v1 did not attest that initialization/LS-lambda selection was isolated from the exit holdout. No v1 artifact should be promoted into an accepted v2 prefix by relabeling or editing it.

The default recovery path is:

```text
<output-dir>/<sweep-id>.blockNN.training-resume.pt
```

A fresh run refuses to overwrite an existing recovery checkpoint. After interruption, rerun the exact same command with `--resume`; all prefix-manifest, capture-registry, train-plan, train-selected-initialization and source identities must still match. An alternate scratch path may be declared explicitly with `--resume-path`. Recovery resumes only from completed epochs. If interruption happens partway through an epoch, that epoch is replayed from the preceding completed-epoch checkpoint rather than treating a partial optimizer sequence as complete evidence. Resume validation also requires the stored `(stage, epoch, case_id)` event sequence to match the exact current training-capture traversal; an event list with the right count but different order or case identity is rejected.

The holdout is an exit-gate dataset. Do not inspect it to choose identity versus LS, choose the LS regularization value, change the train plan, decide whether to unfreeze copied `q_norm`/`out_proj`, or tune thresholds under the same sweep identity. Any such change requires a new predeclared experiment/evidence contract; the current holdout cannot be recycled as the selector.

A failed numerical gate returns exit code `2`. Its candidate artifacts remain immutable evidence, but the current prefix manifest and live accepted model are unchanged. A fold/parity/publication failure raises and the live target block is restored; persisted candidate evidence still does not become accepted without the next prefix manifest.

## 6. Export and run periodic folded BF16 snapshots

The authoritative design requires a second Stage-B checkpoint form in addition to q/R/v training recovery: a deployable folded BF16 snapshot for periodic full-model testing. Do not create one implicitly after every block; choose the testing points before or during the fixed experimental procedure according to the intended full-model validation cadence, because each snapshot is a full H3 checkpoint.

For an accepted prefix, export the complete mixed-topology model:

```bash
python tools/export_progressive_snapshot.py \
  --teacher /path/to/pinned_bf16_teacher.safetensors \
  --prefix-manifest /path/to/progressive_artifacts/keyless-core50-001.prefix-10.json \
  --artifact-dir /path/to/progressive_artifacts \
  --output /path/to/progressive_snapshots/keyless-core50-001.prefix-10.safetensors
```

The exporter revalidates the immutable prefix chain, reconstructs accepted attentions from their checkpoint/result hashes, folds each accepted q/R/v attention into deploy QV form, leaves all later core blocks and both token refiners native QKV, moves the model to CPU before serialization, and writes the safetensors plus a hash-bound sidecar without replacing an existing path.

These snapshots use the distinct architecture identity `h3_keyless_progressive_core50_v1` and are explicitly marked `canonical_release=false`. They are **not** valid `h3_keyless_core50_v1` release artifacts and must not pass through the final all-Keyless loader. The snapshot validator requires the exact accepted-QV/native-QKV boundary, forbids training-only `q_proj`/`query_route`/`v_proj` tensors, preserves the pinned teacher tensor count, and binds the artifact to the prefix manifest, Stage-A campaign, dataset, gate policy, source revision and pinned teacher lineage.

Before using a snapshot for a periodic full-model run, validate that its bytes can reconstruct the mixed runtime topology in a fresh pinned native model:

```bash
python tools/verify_progressive_snapshot.py \
  --teacher /path/to/pinned_bf16_teacher.safetensors \
  --snapshot /path/to/progressive_snapshots/keyless-core50-001.prefix-10.safetensors \
  --snapshot-manifest /path/to/progressive_snapshots/keyless-core50-001.prefix-10.safetensors.manifest.json \
  --prefix-manifest /path/to/progressive_artifacts/keyless-core50-001.prefix-10.json
```

The verifier uses bounded tensor-at-a-time safetensors reads after validating the complete key/shape/dtype map, rather than materializing another full snapshot state dict beside the H3 model. The same path is exposed in ComfyUI as **MiniMax H3 Progressive Snapshot Loader**. Supply the pinned teacher model, snapshot, snapshot receipt and prefix manifest; the node reconstructs the exact accepted-QV/native-QKV topology in a fresh teacher and invalidates Comfy's cached model-size value so subsequent scheduling measures the smaller reconstructed module tree.

Both the CLI verifier and Comfy node establish structural topology/state reload only. The required periodic full-denoiser comparison still has to execute the reloaded model on the fixed validation suite; successful serialization/reload is not numerical or media parity evidence.

## 7. Advance early-to-late

After block 0 succeeds, the emitted manifest is `keyless-core50-001.prefix-01.json`. Repeat Sections 3–5 using that manifest. The next capture corpus must be newly generated through accepted block 0, and the training target is block 1.

Continue strictly in order until the accepted prefix contains blocks `0..49`. Never skip a block, train a later block against an older prefix, combine captures from different prefix identities, or manually edit a prefix manifest. The prefix chain and every accepted block reference immutable checkpoint/result SHA-256 identities.

If a candidate fails, diagnose it under a new experiment/artifact identity. The current artifact naming intentionally refuses overwrite; do not delete or replace failed evidence merely to rerun different hyperparameters under the same identity. A changed train plan or gate policy is a new experiment and requires matching authorization rather than silent continuation.

## 8. Stage-B exit and canonical BF16 handoff

Structural unit tests establish serialization, provenance, rollback, lazy-memory, crash-recovery, mixed-snapshot, train/holdout isolation and orchestration contracts. They do not establish that H3 outputs remain visually/audibly acceptable.

Phase 4 exits only after all 50 core blocks have been empirically accepted under the fixed sweep and the required periodic full-model checks have been run. The deploy form contains QV plus the folded route; the training-only `query_route` must not remain in deployable artifacts. Generation-level validation, ecosystem reference compatibility, INT8 ConvRot validation and measured performance remain later phases of the authoritative design.

The canonical all-Keyless BF16 **format/export path** is already available so a completed `prefix-50` can be handed into the later full-model and deployment phases without inventing an aggregate q/R/v checkpoint:

```bash
python tools/export_progressive_bf16.py \
  --teacher /path/to/pinned_bf16_teacher.safetensors \
  --prefix-manifest /path/to/progressive_artifacts/keyless-core50-001.prefix-50.json \
  --artifact-dir /path/to/progressive_artifacts \
  --output /path/to/keyless-core50-bf16.safetensors
```

This command rejects an incomplete prefix, reconstructs all 50 accepted blocks from their immutable checkpoint/result hashes on CPU, exports their already-folded QV state directly as canonical `h3_keyless_core50_v1`, refuses to replace an existing artifact/receipt pair, and reopens the written bytes for both strict deploy validation and pinned-teacher compatibility validation. It records the completed prefix payload and prefix-manifest SHA-256 in the export receipt.

The Stage-B training revision and canonical export revision are separate provenance facts: the prefix retains the frozen training/capture `code_commit`, while the clean source revision that performs export is recorded as `export_commit`. Export hardening after a sweep therefore does not rewrite the training identity.

Producing this canonical-format file does **not** mean the BF16 model passed Phase 5, ecosystem gates, or release gates. Do not feed it into production INT8/release claims until the required fixed full-denoiser and decoded-media evidence has passed.