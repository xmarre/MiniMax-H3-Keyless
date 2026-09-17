# Stage-B Progressive Core50 Runbook

This runbook covers Phase 4 of `KEYLESS_H3_IMPLEMENTATION_DESIGN.md`: early-to-late conversion of the 50 MiniMax H3 core transformer blocks after a passed Stage-A campaign. It does **not** establish generation-quality parity, production-kernel compatibility, INT8 ConvRot quality or speedup. Those remain later empirical gates.

Stage B is deliberately iterative. For target block `i`, the capture corpus must come from the exact accepted prefix `0..i-1`; the original native QKV block `i` is then trained/evaluated against the Keyless candidate on those same live hidden inputs. A failed candidate is persisted as evidence but does not advance the prefix.

## 1. Freeze the complete policy before Stage A

The gate manifest used for Stage A is also the hash-bound Stage-B policy. In addition to all required `stage_a_*` thresholds, it must contain these predeclared non-negative finite thresholds before Stage-A evidence is produced:

```json
{
  "progressive_fold_atol": 0.0,
  "progressive_fold_rtol": 0.0
}
```

The values above are only syntax examples, not recommended tolerances. Choose them from calibration evidence before observing the full progressive sweep. Stage-B authorization rejects a gate manifest that does not already contain both values. Do not add or loosen them after Stage A under the same experiment identity: changing the gate manifest changes its hash and invalidates the authorized prefix.

Stage B also requires the exact Stage-A train-plan file. Both its canonical semantic identity and full file SHA-256 are checked against the passed Stage-A campaign result.

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
4. verifies native same-input replay for every current-prefix capture;
5. computes bounded activation-derived least-squares route statistics on train captures;
6. evaluates the fixed identity/least-squares initialization grid on holdout captures;
7. trains the fixed monotonic plan (`route -> query -> value -> norm_out`, limited to the stages declared by the Stage-A plan);
8. atomically replaces a mutable crash-recovery checkpoint after every **complete epoch** with q/R/v student state, current-stage optimizer state, Python/NumPy/Torch RNG state and prior training events;
9. recomputes the frozen numerical gate on holdout evidence;
10. persists the immutable candidate result/resume pair regardless of pass/fail;
11. removes the mutable crash-recovery file only after that immutable candidate transaction succeeds;
12. only if the gate passes, folds the training route into deploy Q, verifies fold parity using the predeclared `progressive_fold_atol/rtol`, installs the target block and atomically publishes the next hash-chained prefix manifest.

The default recovery path is:

```text
<output-dir>/<sweep-id>.blockNN.training-resume.pt
```

A fresh run refuses to overwrite an existing recovery checkpoint. After interruption, rerun the exact same command with `--resume`; all prefix-manifest, capture-registry, train-plan, selected-initialization and source identities must still match. An alternate scratch path may be declared explicitly with `--resume-path`. Recovery resumes only from completed epochs. If interruption happens partway through an epoch, that epoch is replayed from the preceding completed-epoch checkpoint rather than treating a partial optimizer sequence as complete evidence.

A failed numerical gate returns exit code `2`. Its candidate artifacts remain immutable evidence, but the current prefix manifest and live accepted model are unchanged. A fold/parity/publication failure raises and the live target block is restored; persisted candidate evidence still does not become accepted without the next prefix manifest.

## 6. Export periodic folded BF16 snapshots

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

That verifier establishes structural topology/state reload only. The required periodic full-denoiser comparison still has to execute the reloaded model on the fixed validation suite; successful serialization/reload is not numerical or media parity evidence.

## 7. Advance early-to-late

After block 0 succeeds, the emitted manifest is `keyless-core50-001.prefix-01.json`. Repeat Sections 3–5 using that manifest. The next capture corpus must be newly generated through accepted block 0, and the training target is block 1.

Continue strictly in order until the accepted prefix contains blocks `0..49`. Never skip a block, train a later block against an older prefix, combine captures from different prefix identities, or manually edit a prefix manifest. The prefix chain and every accepted block reference immutable checkpoint/result SHA-256 identities.

If a candidate fails, diagnose it under a new experiment/artifact identity. The current artifact naming intentionally refuses overwrite; do not delete or replace failed evidence merely to rerun different hyperparameters under the same identity. A changed train plan or gate policy is a new experiment and requires matching authorization rather than silent continuation.

## 8. Stage-B exit and remaining gates

Structural unit tests establish serialization, provenance, rollback, lazy-memory, crash-recovery, mixed-snapshot and orchestration contracts. They do not establish that H3 outputs remain visually/audibly acceptable.

Phase 4 exits only after all 50 core blocks have been empirically accepted under the fixed sweep and the required periodic full-model checks have been run. The deploy form contains QV plus the folded route; the training-only `query_route` must not remain in deployable artifacts. Final canonical BF16 export, generation-level validation, INT8 ConvRot export/validation, native/SOL/VDN/Flow interoperability and measured performance remain later phases of the authoritative design.