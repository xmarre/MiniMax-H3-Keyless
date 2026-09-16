# MiniMax-H3 Keyless implementation design

Status: **authoritative implementation specification with empirical gates**. No trained Keyless model, production CUDA/CuTe implementation, generation-quality parity result, or speedup is established by this document. Audit date: 2026-09-16. `docs/evidence/source_manifest.json` is the initial evidence snapshot; `docs/evidence/completion_delta_20260916.json` records the completion-time moving-PR delta.

## 1. Product and decisions

Build a value-routed attention derivative of `xmarre/MiniMax-H3-Pruned-Ref-Delta-Fused-r1024-ComfyUI`. Preserve its pruned timestep conditioning, fused reference behavior, audio/video schedules, packing, and existing conditioning interfaces. Train against the corresponding BF16 checkpoint; export both BF16 and native ComfyUI INT8 ConvRot checkpoints. The deployment baseline is `MiniMax-H3-Pruned-Ref-Delta-Fused-r1024-comfy-int8-convrot.safetensors`, including quantized `fc2`.

The chosen first production architecture replaces all **50 diffusion-block attention modules**, retains the **two original QKV token-refiner blocks**, and declares this scope explicitly as `h3_keyless_core50_v1`. The refiner processes text and can run in `preprocess_text_embeds`; its cost is not proportional to the large audiovisual sequence at each transformer evaluation. Replacing it immediately risks conditioning drift upstream of every block for only two more removed projections. It remains real, live QKV attention, not a fake compatibility module. A later all-52 variant requires a separately identified checkpoint and conditioning-quality validation; it is not a prerequisite for core50 release. The project must never describe core50 as containing no K weights anywhere.

Use a packed QV projection and a **normalized, position-transformed routing view of V**, while retrieving the unmodified V. Learn a per-head query factorization before query normalization so the factors can be folded at export. Do not retain a dead K projection, recover K through a full cross-head transform, or silently route through the original QKV teacher at runtime.

A materialized routing tensor is permitted in the reference/compatibility backend. The intended optimized path derives routing tiles from the selected V domain inside the kernel. Compatibility output equivalence and native-kernel memory efficiency are separate gates.

Distillation is mandatory. Bounded measurements already contradict exact fixed-V raw linear conversion in sampled teacher heads. Whether this derivative can meet audiovisual quality parity remains an empirical research risk. The implementation phases below prescribe the response to failure without pretending that a training recipe guarantees success.

## 2. Evidence and provenance

`evidence/source_manifest.json` records exact source heads, active PR heads/bases, both complete safetensors headers, model metadata, and supplied-paper hashes from the initial audit. Headers were obtained by bounded HTTP byte ranges at a pinned model revision. They establish storage topology and dtype, not a full-file checksum verification. Sampled BF16 weight probes are recorded separately. No local CUDA runtime or installed user ComfyUI environment was inspected; repository source is not installed-kernel/compiler provenance.

### 2.1 Target model

Hugging Face revision: `f26363f0d42fbd46ef59008fd5e4d946ea0e9426` of [the target repository](https://huggingface.co/xmarre/MiniMax-H3-Pruned-Ref-Delta-Fused-r1024-ComfyUI/tree/f26363f0d42fbd46ef59008fd5e4d946ea0e9426).

Immediate source lineage: `diffusers-modular/MiniMax-H3-Pruned-Ref-Delta-Fused-r1024`, revision `c7d26373ecb070f1f1dc8811609d56d10d45d690`, as recorded by the conversion provenance. Native BF16 file: `MiniMax-H3-Pruned-Ref-Delta-Fused-r1024-comfy.safetensors`. Published full-file hashes, to verify on actual acquisition:

- BF16: `78b88298e241231b3bd95d752abde711efc9dd6517669a8a934faeb70baf6a98`.
- INT8 ConvRot: `00be5b0f995cc5a628921790f69cb22e138776c1e12235e3eab521941bb4b8c2`.

Do not substitute a generic H3 checkpoint or the old `fc2bf16` compatibility artifact. Preserve original license/NOTICE and exact model provenance when distributing derivatives; the source's custom model license is not replaced by a software license.

Verified header facts:

| Property | Value |
|---|---|
| Hidden width D | 5,376 |
| Heads H; head width d | 56; 128 |
| Attention inner width I | 7,168, not D |
| Main blocks; refiner blocks | 50; 2 |
| Each QKV weight, PyTorch order | `[21504, 5376]`, Q then K then V |
| Each output projection | `[5376, 7168]` |
| Q/K norm weight | `[128]`, shared across heads by broadcasting within each module |
| MLP fc1; fc2 | `[28672, 5376]`; `[5376, 14336]` |
| Pruned time table | `adaln_t_table [1025,8]`, F32 |
| Main AdaLN weight; bias | `[96768,8]` BF16; `[96768]` F32 |
| Final AdaLN weight; bias | `[10752,8]` BF16; `[10752]` F32 |
| RoPE frequencies | `[16]` F32 |
| BF16 container | 534 tensors, plus metadata entry |
| INT8 ConvRot container | 932 tensors, plus metadata entry; 200 I8 weights and 200 U8 quant descriptors |

Native conversion combines 52 Q/K/V groups and restores native SwiGLU ordering. The conversion source `tools/convert_h3_diffusers_to_comfy_v3.py` and `PROVENANCE.md` were inspected. Keep all 51 folded AdaLN biases. Do not reconstruct the absent full timestep MLP or introduce SiLU into the pruned curve projection. BF16 includes `adaln_basis` and `adaln_mean` auxiliaries unused by native inference; quantized derivatives omit them. Preserve these in provenance, not as newly trainable model behavior.

### 2.2 Audited source heads

| Repository | Audited default branch SHA |
|---|---|
| Comfy-Org/ComfyUI, **master** | `7a0b5eede3f9721c8faab290689893f36edc6d66` |
| xmarre/ComfyUI-Spectrum-MiniMax-H3 | `120d72e2f48b781235b34149e39bbdf0f1317d82` |
| xmarre/ComfyUI-Sol-H3 | `f82ff2693be37dbad3438a30eb389d77136c0276` |
| xmarre/ComfyUI-VDN-H3-Plus | `76b31323f9e09019b435237dcd8bad1e05476ce1` |
| xmarre/MiniMax-H3-Flow-Aligned-Regenerate | `970396db839ae7ab431b9718859f6d48a2e5019b` |
| xmarre/ComfyUI-H3-Continuum-Plus | `a5b8943844594545301b20d01af5d9e3fa38ae29` |
| xmarre/ComfyUI-Untwisting-RoPE | `63c8d8df5e08dfdad4b397480d966b90f9b5a6c6` |
| kijai/ComfyUI-KJNodes | `d3cfe21625e5170126ce06fbfcfe1d88108688c3` |
| Comfy-Org/comfy-kitchen | `6c7f08bf8d8c6d039892b311d40c976c26ce979f` |

The old `xmarre/ComfyUI-H3-Continuum` URL redirects to `ComfyUI-H3-Continuum-Plus`. Follow the repository identity, not the obsolete name. ComfyUI does not have a `main` branch in this snapshot.

Source/worktree heads actually inspected during the initial design pass include Spectrum #110 `78a9a5b36c185a55af59a44c073bc7f8e534bc97`; Sol #14 `8b049e39d000b0d283f2f01e8477f74cf5c2d608`; VDN #18 `333d63f81d33fe29dc1f1f637c5f4a4396880f99`; Flow #30 `f6a940fdc2fb6d31249a487aff5e4a1297151e9c`; Flow #46 at its initial audited head `5970116ef5a04c405f0f2a4c873585a7097b15df`; and Continuum #24 `4d9c5f4a01d35f49ef2890cf67653adde59ce571`.

Preserve existing topology. Sol #11 → #12 → #13, VDN #15 → #16 → #17, and Flow #43 → #44 → #45 are distinct W/E/M diagnostic stacks. Production Sol #14 branches from #9's weighted-measure head; VDN #18 branches from #8's audio-fidelity head; Flow #46 branches from #41. Do not rewrite, repurpose, merge, or subsume diagnostic evidence into Keyless work. Use new implementation mirrors for any later companion changes. Source manifests include actual base SHAs and ref names.

Completion-time metadata re-fetch found Sol #14 still at `8b049e39d000b0d283f2f01e8477f74cf5c2d608` and VDN #18 still at `333d63f81d33fe29dc1f1f637c5f4a4396880f99`. Flow #46 advanced after the initial source snapshot to `c67025e7456d54b33b032ab4f2eebba2966aca39`; its PR body now reports hosted CI green, including Ruff/format and source-contract lanes. The completion pass did **not** re-audit every changed source byte at that new Flow head, so implementation must re-fetch it rather than treating the new metadata as source review. See `evidence/completion_delta_20260916.json`.

The active production PRs remain structural evidence only. Sol #14 and VDN #18 report hosted contract CI green but still require real SM120/performance/media gates. Re-fetch reviews and CI before implementation. ComfyUI #16239 (key-measure execution), #16344 (H3 BSA composition), #16245 (compiler opt-out) and #16362 (draft sparse attention) are moving relevant interfaces; do not presume their proposals are merged or required.

### 2.3 Research consequences

| Supplied primary paper | Evidence relevant to this design | Limit on inference |
|---|---|---|
| [Keyless, 2606.21848v3](https://arxiv.org/abs/2606.21848v3), §§2.1–2.4, Appendix A/B | Query-side value-space routing; foldable linear factors; per-head subspace condition; language-model training experiments including RoPE architectures | Does not establish exact conversion of H3, H3 quality, or a fused normalized-V SM120 kernel. Linear equivalence proof omits H3 normalization/position operators. |
| [QKV variants, 2606.04032v3](https://arxiv.org/abs/2606.04032v3) | Projection sharing is task dependent; separate Q and shared K/V is a useful ablation | Language/vision results cannot determine audiovisual diffusion parity. Symmetric QK logits do not imply a symmetric row-softmax matrix. |
| [QV May Be Enough, 2603.15665v1](https://arxiv.org/abs/2603.15665v1), §§4–7 | QV attention baseline; positional interference deserves explicit testing | Reduced-layer Transformer-BIG translation experiments do not validate H3. Linguistic interpretations are hypotheses, not model constraints. |
| [Slim Attention, 2503.05840v2](https://arxiv.org/abs/2503.05840v2), §1 and non-square discussion | Recovering values from concatenated keys can be algebraically exact under rank assumptions, with additional compute | This uses cross-head/global reconstruction; it does not prove per-head QV equivalence. Normalization and RoPE must be accounted for. |
| [DeepSeek-V2, 2405.04434v5](https://arxiv.org/abs/2405.04434v5), §2.1/Appendix C | MLA compression and explicit decoupled RoPE address an absorption obstacle | A separate positional key would change this project's architecture; MLA's autoregressive cache savings are not H3 performance predictions. |

Do not adopt the Keyless paper's statement that standard routing gradients are fully independent of V as a literal backpropagation claim: the upstream gradient through softmax depends on the values. The architectural parameter tying is real; the stronger independence interpretation is unnecessary here.

## 3. Teacher math and conversion limits

Use row-vector notation in this document; transpose stored PyTorch weights. For one head, A= WQ, B=WK, C=WV have shape `[5376,128]`. Different heads have independent projections; norm gains `[128]` are shared over the 56 heads of a module. Let

`N_g(z) = (z / sqrt(mean(z²) + 1e-5)) ⊙ g`.

The main attention input X is already normed and modulated by the block's AdaLN. Native H3 then computes, for positions i,j:

`q_i = P_i(N_gq(x_i A))`

`k_j = P_j(N_gk(x_j B))`

`v_j = x_j C`

`o_i = Σ_j softmax_j(q_i · k_j / sqrt(128) + mask_ij + log_mass_j) v_j`.

The native unpatched dense model has no causal mask and unit mass. Mask/restricted domains and log measure describe explicit ecosystem interventions; they are not present by default. Concatenate heads and apply `[5376,7168]` output projection, then the existing gated residual/MLP path.

`P_i` rotates the first 96 channels, using 16 frequencies on each of time/height/width; it pairs `[0:48]` with `[48:96]` in split-half order, leaving `[96:128]` unchanged. Main packed order is text, conditioning/reference rows, audio, video. Audio is stereo channel-major. Position IDs and modality segments come from native PackedLayout, not an inferred video-only grid. Token refiners call Attention without RoPE.

Source: ComfyUI `comfy/ldm/minimax/model.py`: `Attention`, `rope_rotation_table`, `rope_freqs`, `RefinerBlock`, `DiTBlock`, `PackedLayout`, `_forward`; `comfy/model_detection.py` detects the pruned model and infers heads from QKV first dimension. Current main uses `AttentionTensorContainer` and fused kitchen RMSNorm/RoPE; inference transforms Q/K in their packed backing buffer, whereas its training branch selects the out-of-place operation. This buffer mutation must never be applied to the student's retrieval V.

Without normalization/positions the teacher form is Ω=A Bᵀ. Fixed-C Keyless can reproduce it iff `Ω = Ω P_C`, where `P_C=C C⁺`. For full-column-rank C, the least-squares query is

`A_eff = A Bᵀ C (Cᵀ C)⁻¹`.

Implement via SVD/QR or a regularized solve, never an explicit inverse. If A has full column rank, exactness implies `col(B) ⊆ col(C)`. H3's concatenated inner width 7,168 being larger than hidden width does not remove this **per-head** requirement. Recovering raw K through all concatenated V may be possible at sufficient rank but requires cross-head reconstruction and different cost/layout; it is not the chosen mechanism.

Even zero raw residual would not prove normalized, position-aware equivalence. Normalization introduces input-dependent denominators; arbitrary transforms do not commute with RoPE. A query factor placed after RMSNorm or RoPE generally cannot be absorbed into the original input projection. Equality of softmax distributions is weaker than raw-logit equality because row-constant offsets cancel; output equality is weaker again. Consequently, raw residuals diagnose initialization geometry, not audiovisual quality or impossibility of all alternative architectures.

### 3.1 Bounded diagnostics

Measure each head using float64 decompositions of BF16 source weights promoted to float64. Record ranks with explicit singular-value tolerance, singular spectra/conditioning, principal angles between col(B) and col(C), `||B-P_C B||F/||B||F`, and `||Ω(I-P_C)||F/||Ω||F`. Avoid forming 5376×5376 Ω: use trace products of 128×128 Gram matrices. Record source slice hashes and offset/range validation. The accompanying probe samples head 0 of layers 0,25,49; it is deliberately not an all-head claim. Whole-model diagnostics are phase-1 work only if they influence initialization choices; one failed head already disproves universal exact fixed-V raw conversion.

The interrupted investigation reported approximately **96.3% relative raw bilinear residual for its first sampled BF16 head** under the best fixed-V linear reconstruction. That number is evidence against universal exact conversion for that sampled head, not a generation-quality result and not a proof that distillation cannot succeed. Reproduce the probe from the pinned BF16 checkpoint before using the number quantitatively in implementation decisions.

Activation diagnostics must collect **actual block input after AdaLN**, position IDs, modality/conditioning labels, sigma, and the effective provider/domain/mass/preprocessor configuration. First use unpatched BF16 native attention. For at least 16 fixed cases with short/long, reference, audio and mixed-grid coverage, sample eight sigma strata across [0,1] including endpoints when valid. Hold out input cases, not merely tokens from the same case.

For sampled query rows compute centered-logit NRMSE (subtract each row mean), teacher-to-student softmax KL, output NRMSE/cosine before and after out_proj, and per-modality attention mass. Use complete key domains with streamed log-sum-exp for selected query rows; a 128-query × 56-head × N-key dense FP32 tensor can itself be large. Process a few heads and queries at a time. Key subsampling changes softmax and must be labeled a diagnostic approximation. No persistent N² capture, whole-layer GPU hooks retained across steps, or unbounded activation archive.

## 4. Chosen student architecture

For each converted head:

`u_i = x_i A_h R_h`

`q_i = P_i(N_gq(u_i))`

`v_j = x_j C_h`

`route_j = U_j(P_j(N_gr(v_j)))`

`o_i = Σ_j softmax_j(q_i · route_j / sqrt(128) + mask_ij + log_mass_j) v_j`.

A has shape `[5376,128]`, R `[128,128]`, C `[5376,128]`. U is identity unless an explicitly supported routing preprocessor such as current Untwist is active. Retrieval values never receive routing RMSNorm, RoPE, or Untwist. `gq` and `gr` are learned `[128]` module gains shared over heads, preserving H3's norm parameter granularity. Keep epsilon=1e-5, 56 heads and scale=128^-0.5. No learned routing-side matrix is applied to V in the chosen production core.

This is an H3-specific normalized Keyless architecture. The raw paper formula is recovered with identity normalization/position transforms, but its empirical results are not proof for this variant. Normalizing V for routing preserves the teacher's bounded routing geometry while keeping value amplitudes available to retrieval. Moving R **before** q_norm is intentional: the two input-side linear operations remain foldable. It is not an exact rewrite of a teacher norm.

### 4.1 Parameter/checkpoint forms

| Form | Stored core attention tensors | Execution |
|---|---|---|
| Training/resume | `q_proj.weight [7168,5376]`, `query_route.weight [56,128,128]`, `v_proj.weight [7168,5376]`, q_norm, route_norm, out_proj | Linear Q → per-head linear R → q_norm → partial RoPE; independent V used by route and retrieval; autograd through both V uses |
| Canonical deployable BF16 | `qv_proj.weight [14336,5376]` ordered `[Q_effective;V]`, q_norm, route_norm, out_proj | One packed GEMM; split at 7168 |
| Optimized BF16 runtime | Same weights | Q normalization/positioning plus fused value-route attention; no global routing tensor |
| INT8 ConvRot | Same QV naming/shapes, I8 qv weight plus per-output-row F32 scale and U8 `.comfy_quant` | Existing ConvRot linear layout; outputs in model compute dtype; attention still consumes BF16 |

In PyTorch storage orientation, `Wq_eff[h] = Wr[h] @ Wq[h]`. Compute export products in FP32 (float64 diagnostic oracle if needed), round once to BF16, then quantize that BF16 artifact. No norm or positional operator lies between these factors. The real-arithmetic fold is exact; BF16 reassociation/rounding is not bit-exact and must pass an export parity gate. No `query_route` tensor remains in deployment checkpoints. Preserve the unfused trainable checkpoint separately with optimizer/RNG state for resumption; it is not loadable as a production model by accidental key matching.

Metadata must include `architecture=h3_keyless_core50_v1`, format version 1, QV order, norm epsilon, RoPE policy `h3_split_half_96_v1`, hidden/head geometry, core50/refiner-QKV scope, parent model revision/hash, training run and export provenance, and quantization recipe if present. Put large provenance in a sidecar manifest hashed from metadata. Metadata is not enough for detection: verify expected tensor signatures/counts, absence of core qkv weights, retained refiner signatures and full parameter compatibility. Reject contradictory or mixed core QKV/QV checkpoints rather than partially loading them.

### 4.2 Initialization choice

Copy C, out_proj, gains, all non-attention weights and both token-refiner blocks from the exact BF16 teacher. Initialize route_norm from teacher k_norm as a scale prior; it is not a claim that V has become K. Initialize A from teacher Q; compare identity R and regularized least-squares R=`Bᵀ C (CᵀC + λI)⁻¹` on captured activations. Use λ relative to the average diagonal of CᵀC (pilot 0, 1e-4, 1e-2 times that scale), with singular-value diagnostics. Select per layer by held-out actual attention-output error, not raw residual alone. A rank-reducing LS initialization can perform poorly after q_norm; do not discard the identity baseline.

Initially freeze C to isolate routing calibration, then unfreeze it because fixed per-head value subspaces are demonstrably restrictive. Pure orthogonal reparameterization of C cannot change its column space. Gradient learning of C is required to change that space. R is a training parameterization choice, not additional inference expressivity beyond the freely trained effective Q projection. Include a directly learned Q-effective ablation only if factorized training fails or is materially slower; document a change of training form without changing deployed math.

## 5. Efficiency budget

All numbers below are calculated from inspected headers and the core50 scope. They are reductions in specific terms, not speedup predictions.

| Term | Core50 reduction |
|---|---:|
| K weights per block | 38,535,168 parameters |
| K weights total | 1,926,758,400 parameters |
| BF16 weight payload | 3,853,516,800 bytes = 3.5888671875 GiB |
| INT8 weight payload | 1,926,758,400 bytes = 1.79443359375 GiB |
| INT8 output-row scales | 50×7168×4 = 1,433,600 bytes |
| Projection arithmetic | 3,853,516,800 FLOPs per token across 50 blocks, counting multiply-add as 2 |
| Training-only R parameters | 45,875,200, removed on export |

Per block QKV→QV reduces input-projection FLOPs by one third. Including out_proj, the four D×I attention linears become three, a 25% reduction of that linear arithmetic. MLP arithmetic and softmax matrix products are unchanged. Standard dense score+retrieval products remain about `4 H Nq Nv d` FLOPs. Sparse SOL still needs routing selection and approximate/exact contributions; a smaller projection is not proof of faster attention.

An eliminated BF16 K-sized activation is `2 N I` bytes: at N=65,536, 896 MiB. This is a **per-live-buffer** quantity, not 50 times less peak inference VRAM. Packed QKV lifetime, aliases, workspace, provider copies and allocator reuse determine the peak. The materialized-route bridge still holds Q, route and V, so it does not establish this activation saving even though it removes K weights/GEMM. Training saves fewer activations because backward requires intermediates.

The native kernel should eliminate the global K write and full-size K reads, deriving routing from V tiles and retaining separate shared-memory/register representations for routing and retrieval. Repeated value loads across CTAs, summary passes, norms, positional metadata and occupancy may reduce or erase bandwidth savings. Measure HBM traffic where supported, as well as peak allocation and launch counts. Do not assume a 50% reduction in total HBM traffic. H3 has no autoregressive KV cache to halve.

The BF16 container has approximately 20.11 billion stored elements; core50 deployable student approximately 18.18 billion (includes small non-trainable/auxiliary items). BF16 teacher+student weights alone approach 71 GiB. Full student mixed-precision Adam at an illustrative 16 bytes/parameter is approximately 271 GiB before teacher, activations or workspaces. A 96 GB GPU cannot hold that training state. Use block-local training first; global training requires explicit offload/sharding or parameter-efficient updates. At N=65,536, one BF16 hidden stream is 672 MiB. Recomputing activations is essential; sequence/window slicing that changes attention is not an equivalent memory optimization.

Measure transformer-only and full generation separately. If a measured removable projection share is f and its isolated speed ratio is r, Amdahl's expression `1 / ((1-f)+f/r)` is an estimate under unchanged other costs. VAE, text conditioning, offload and sampler behavior reduce end-to-end benefit. No numeric speed claim is authorized before measured generation tests.

## 6. Runtime model, loading and semantic contracts

### 6.1 Model ownership and loader

The repository owns a Keyless H3 model implementation and canonical loader. Do not require a fake `qkv_proj` tensor or synthesize a zero/dead K slice just to enter ComfyUI's current generic H3 detection path. At the audited ComfyUI head, `model_detection.py` derives H3 head count from `blocks.0.attn.qkv_proj.weight`, so an exported QV checkpoint is not natively discoverable by that path without a core change.

The first implementation should therefore provide a strict Keyless loader that:

1. reads safetensors metadata and tensor signatures before allocating the model;
2. validates `architecture=h3_keyless_core50_v1`, QV shape/order, retained token-refiner QKV tensors, pruned AdaLN signatures and parent provenance;
3. constructs the same MiniMax-H3 model/base-model/latent semantics used by native ComfyUI while replacing only the 50 main attention modules with Keyless attention;
4. loads quantized tensors through Comfy's current quantized-operations machinery rather than eagerly dequantizing the complete checkpoint;
5. returns an ordinary Comfy `MODEL`/ModelPatcher so conditioning, samplers and model-patch APIs continue to work;
6. refuses generic H3, mixed QKV/QV or unknown format versions instead of guessing.

Prefer subclassing/composition around current `MiniMaxH3Model` and preserving its packing/forward implementation over copying the whole H3 model, provided the live source still permits a narrow replacement. Constructor-time native QKV modules may be created on meta/offload storage and replaced before weight materialization if that avoids a maintained model fork. If current Comfy source makes this unsafe, implement the smallest source-derived Keyless model class and pin the compatibility boundary. Do not monkeypatch the global native H3 class.

Core upstream support may later teach `model_detection.py`/`supported_models.py` about the QV signature, but the repository must remain able to load its own canonical checkpoints without waiting for an upstream merge. Any KJNodes loader/optimization support is a compatibility lane, not the owner of the checkpoint format.

### 6.2 Static attention capability

Expose a versioned semantic capability on the diffusion model instead of requiring other nodes to infer architecture from attribute names. The implementation name may be a dataclass/protocol, but its public identity is `minimax_h3_keyless_contract_v1` and it must include at least:

- `api = 1`;
- `architecture = "h3_keyless_core50_v1"`;
- `core_blocks = 50`, `token_refiner = "native_qkv"`;
- `heads = 56`, `head_dim = 128`, `inner_dim = 7168`;
- `routing_source = "value"`;
- `retrieval_source = "raw_projected_value"`;
- `routing_norm = "rmsnorm"`, epsilon `1e-5`;
- `rope_policy = "h3_split_half_96_v1"`;
- QV projection order and a stable way to obtain projection dtype/device/weight identity without dereferencing `qkv_proj`;
- deployed-checkpoint format version and provenance identity.

Consumers must include architecture/format identity in caches and history. QKV history, calibration or provider receipts must never be reused merely because sequence geometry matches.

### 6.3 Lazy routing/provider contract

A second contract is required for numerical attention backends. Use a distinct Keyless contract; do **not** call it VDN provider v5 or otherwise overload existing VDN/Sol provider-version meaning.

The semantic provider key is `minimax_h3_keyless_provider_v1`. The exact Python packaging may evolve, but the call contract must give a provider:

- the already normalized/positioned query tensor Q;
- raw projected retrieval V;
- head count and scale;
- an immutable routing specification containing route-norm weight/epsilon, the exact RoPE rows or position mapping for the current V domain, block/layout identity, and any logical routing-only preprocessors;
- mask/key-log-measure/exact-block semantics when present;
- query/value row-domain metadata where a consumer has restricted or remapped rows;
- an explicit dense/reference fallback.

The routing specification must support selecting/gathering a value domain while preserving the exact aligned route positions and measure. A provider must not gather a pseudo-K independently from V.

If no Keyless-aware provider is installed, the canonical compatibility fallback may materialize

`route = routing_spec.materialize(V)`

then call current Comfy `optimized_attention(Q, route, V, ...)`. Existing QKV attention preprocessors may be applied to `(Q, route, V)` exactly once **only when their semantics are valid for the logical routing operand**. This fallback establishes functional composition; it does not establish the intended activation-memory saving.

The production optimized path consumes Q + V + routing specification and derives route tiles from selected V inside the backend. Telemetry must distinguish `materialized_route` from `fused_value_route`; release/performance claims may use only the latter.

## 7. Ecosystem compatibility requirements

### 7.1 Spectrum

Current Spectrum source contains direct `attn.qkv_proj` assumptions in Core-BSA execution identity/device/dtype eligibility and model-aware tensor selection. Those paths must gain explicit Keyless handling through `minimax_h3_keyless_contract_v1`; do not create an alias named `qkv_proj`.

Required changes/tests:

- projection identity/device/dtype reads use the semantic contract and QV projection for Keyless blocks;
- model-aware profiling samples the appropriate QV/out/MLP tensors and records the architecture identity, rather than looking for `blocks.<n>.attn.qkv_proj.weight` only;
- Core-BSA/source-audit history identity includes Keyless architecture, routing policy, provider and preprocessing identity so old QKV actual history cannot prime a Keyless forecast;
- actual/forecast schedules and history ownership remain unchanged unless measured evidence requires a policy change;
- unsupported Core-BSA Keyless composition fails closed to an actual/reference route rather than silently interpreting V as ordinary K.

Spectrum forecasts model outputs, not attention internals; no retraining of Spectrum itself is implied. Nevertheless structural forecast correctness does not prove generated-media parity after the base model architecture changes.

### 7.2 Sol-H3

Current Sol-H3 validates explicit Q/K/V tensors and its sparse SM120 kernel consumes all three. Existing exact mode and native provenance also assume ordinary H3 QKV arithmetic. Preserve that QKV path unchanged.

Implement a separate Keyless route in stages:

1. **Reference bridge:** materialize route(V) and feed `(Q, route, V)` through a Keyless-labeled Sol bridge. Arithmetic is checked against dense Keyless attention. This may reuse the current sparse kernel where its numerical assumptions remain valid, but receipts/calibration keys must identify Keyless semantics.
2. **Native Keyless kernel:** accept Q + V + route specification. Load each V tile once where practical, derive route RMSNorm/RoPE/preprocessing in registers/shared memory, score with the route view and accumulate the raw V view. Do not write a global route tensor.

Existing sink rows, weighted key measure, exact K-block ranges and mapped-neighbor descriptors become routing/value-row semantics over the same physical selected V rows. Query-position provider v4 remains an orthogonal VDN-owned mapping contract; do not repurpose its API number for Keyless.

The native Keyless kernel gets its own arithmetic gate against a materialized dense Keyless oracle. Current Sol thresholds may be used as an initial calibration target only after verifying that the same error metrics are meaningful; do not inherit an old QKV pass result. Calibration identity must include route-norm/RoPE/preprocessor policy and the selected value-domain mapping.

Until a Keyless exact executor is independently verified, `exact=True` must fail closed or explicitly use a Keyless reference block. It must never fall through to the original QKV teacher and report that as Keyless exactness.

### 7.3 VDN-H3-Plus

VDN is the largest compatibility change. At the audited production branch, `hybrid.py` directly binds `attn.qkv_proj`; keeps `q_raw`, `k_raw` and `v`; and the learned linear branch calls `branch.readout(...)` with raw Q/K/V plus optional text K/V state. Therefore changing only the softmax provider interface does **not** preserve released VDN behavior.

Split the work explicitly:

- **Softmax/window branch:** use Keyless Q, route(V) and raw V. VDN continues to own grouped/restricted row selection and query-position maps. It selects the exact V domain once; routing is derived from those selected V rows with aligned positions.
- **Learned linear branch:** define a Keyless-specific checkpoint/schema and distill/retrain it. Replace dependence on teacher `k_raw` with a value-derived feature family tested on paired teacher activations (candidate inputs include raw V and/or the normalized pre/post-RoPE routing view). Preserve text-state semantics with the same rule. Do not assume a mechanical K→V substitution preserves the learned branch.
- **Compatibility:** existing VDN checkpoints are not silently declared Keyless-compatible. A diagnostic bridge may temporarily load the teacher K projection to quantify the learned-branch gap, but a release path that keeps persistent teacher K weights/compute is not a production Keyless solution.
- **Metadata:** a released Keyless VDN adapter must identify its parent Keyless checkpoint/architecture and branch-input schema. Mismatched QKV VDN weights fail closed.

Preserve current global/audio context controls, learned softmax gate, native out projection ownership and provider-v4 position maps unless paired evidence shows a necessary change. Re-run audio/reference/media validation; VDN structural tests alone are insufficient.

### 7.4 Flow-Aligned-Regenerate

Flow's current weighted-measure profile preserves all mixed Q/K/V rows and represents carrier density as key log measure. In Keyless that measure is attached to the corresponding routing/value row. Mixed-grid external-sequence geometry remains valid independently of whether the model is QKV or Keyless.

For any legacy path that physically reduces K/V rows, replace the semantic operation with one value-domain selection that returns selected V, aligned route positions and aligned measure. Route is derived after the selection. Do not maintain parallel independently selected pseudo-K state.

Flow must advertise geometry/measure only; the active Keyless attention owner remains authoritative. VDN API-2 geometry does not imply VDN is installed. Re-fetch PR #30 and the current #46 candidate stack before adapting it; #46 advanced beyond the initial audit snapshot during this design task.

### 7.5 Untwisting RoPE

Current Untwist `attention_preprocess_v1` performs reference-key scaling while preserving Q/K/V shape, dtype and device. For Keyless, its semantic target is the **logical routing representation**, not raw retrieval V.

Compatibility path: materialize route after route RMSNorm + partial RoPE, apply the current logical-key transform exactly once to route, leave raw V unchanged, then dispatch attention.

Native path: express the same routing-only scaling/ranges in the route specification so the fused backend applies it to route tiles without modifying retrieval V. Preserve current reference selection, including Continuum/native-RoPE exclusions. Double application remains an error.

### 7.6 Core BlockSparseAttention and other dense providers

Current Core BSA and its Spectrum audit contain native H3/QKV assumptions. Either add explicit Keyless support through the semantic/provider contracts or bypass/fail closed with clear telemetry. Do not spoof source identity or callable ownership to make a QKV-only block patch accept Keyless.

Generic dense providers such as SageAttention can remain usable through the materialized-route compatibility path if they accept ordinary `(Q, route, V)` geometry and do not inspect model weights. That path is correctness compatibility, not the production memory result.

### 7.7 KJNodes

KJNodes contains MiniMax-H3-specific attention optimizations and imports native H3 types. The canonical Keyless loader must not depend on KJNodes, but the user's production workflow requires a compatibility lane. Any KJ H3 patch that inspects `qkv_proj` or replaces native Attention must recognize/refuse Keyless explicitly. `sage_attention=auto` is acceptable only through a verified provider path; no patch may recreate or require teacher K silently.

### 7.8 Continuum

No Keyless-specific Continuum algorithm is justified by the audited architecture. Keep Continuum model-opaque and validate its normal latent/reference/audio contracts with a Keyless `MODEL`. Cover reference packing, progressive/refine paths, audio restore and boundary workflows. If live evidence shows a concrete attention-internal assumption, add the smallest companion change; do not invent one in advance.

## 8. Training and distillation plan

### 8.1 General rules

Use the exact BF16 parent as teacher. Never use the INT8 checkpoint as the canonical teacher or optimize the architecture against quantization noise. Training arithmetic is BF16/FP32 as appropriate; quantization happens only after a deployable BF16 student is frozen.

Do not try to keep teacher and a full Adam-trained student resident with all optimizer state. The default strategy is **block-local/progressive distillation with one teacher block duplicate**, followed by an optional parameter-efficient global correction only if the full-model gate requires it.

Training data need not contain ground-truth target videos for the first distillation stages, but it must exercise the real H3 input distribution. Record a manifest of prompts, reference/audio assets, resolutions/durations, seeds, schedules and hashes. Split holdout by complete case/asset, never by token rows from the same case.

Minimum modality coverage before full sweep:

- native T2V, short and long;
- I2V/keyframe paths;
- image reference and video/reference-capable paths used by the target checkpoint;
- synchronized audio/video conditioning and generation;
- low/high resolution and multiple sequence lengths;
- at least eight sigma strata across the valid sampling interval;
- separate compatibility cases for Flow mixed-grid, VDN and Continuum after the base native student is stable.

Do not train primarily on Flow/VDN/Spectrum-modified execution before native H3 parity; those systems are compatibility distributions, not a replacement definition of the base model.

### 8.2 Stage A: initialization/calibration pilot

Pilot blocks 0, 25 and 49 first. For each block, feed the exact same post-AdaLN hidden input, timestep state, layout and RoPE to frozen teacher attention/block and student attention/block.

Compare at least:

1. identity-R initialization;
2. regularized LS R at λ scales defined in §4.2;
3. directly learned effective-Q only if the factorized form fails to optimize.

Train in this order:

1. R + route_norm only, with V and copied q_norm/out_proj frozen;
2. unfreeze q projection/factorization as needed;
3. unfreeze V so its per-head subspace can move;
4. only then consider q_norm/out_proj if held-out block error plateaus materially above the best attainable student result.

Do not unfreeze copied MLP/AdaLN/non-attention weights in this stage.

Primary losses are normalized attention-output and block-output error on the same block input. Add sampled centered-logit KL/NRMSE and cosine terms as diagnostics/secondary objectives. Normalize each loss by teacher scale and fix the relative weights after the pilot; do not retune weights separately per block to hide bad layers. Track per-modality rows separately.

Pilot exit requires consistent held-out improvement over both untrained initializations, finite/stable gradients, no modality-specific collapse and evidence that the chosen architecture can fit all three depth samples. If a depth sample cannot be fitted after V is unfrozen, stop the full sweep and revisit the routing architecture rather than training 47 more blocks.

### 8.3 Stage B: progressive core50 sweep

Train one block or a small contiguous group at a time, checkpointing after each accepted group. The default order is early→late so later student blocks are calibrated on the distribution produced by already-converted earlier blocks.

For block i:

- run the current partially converted model to the block input;
- evaluate the frozen original QKV block i on **that same input** as the local teacher mapping;
- evaluate/train the Keyless replacement on the same input;
- preserve all other blocks frozen;
- accept the block only when train/holdout metrics meet the pilot-defined gate and real forward execution remains finite.

This avoids storing 50 layers of hidden activations and explicitly trains against distribution shift introduced by prior converted blocks. If small-group training is faster, group size must be bounded by measured VRAM and rollback granularity, not guessed from available VRAM.

Maintain two checkpoints:

- a resumable training checkpoint with q/R/v and optimizer/RNG state;
- a deployable folded BF16 snapshot for periodic full-model testing.

Do not overwrite the last accepted group with an experimental failure.

### 8.4 Stage C: full-model BF16 correction

After all 50 blocks pass local gates, run fixed full-denoiser comparisons and real generations. Local parity does not imply global parity.

If global drift is already within the predeclared acceptance envelope, do not add training complexity. If it is not, escalate in this order:

1. another progressive sweep using live student inputs;
2. low-rank/global correction adapters on Keyless Q/V/route/out projections, trained end-to-end and merged into BF16 weights afterward;
3. optimizer/teacher offload or sharded full attention-only tuning only if the measured gap justifies it.

Before any global end-to-end run, measure actual peak memory with teacher, student, activations and optimizer strategy. Do not infer feasibility from parameter count alone. Teacher targets may be computed in a separate no-grad pass if simultaneous residency is unsafe. Never silently switch the canonical teacher to INT8 to fit memory.

### 8.5 Predeclared numerical gates

Do not invent absolute parity thresholds after seeing the final result. Before the full sweep, measure on the fixed holdout suite:

- BF16 teacher repeatability across the selected deterministic backend;
- BF16 teacher vs published INT8 ConvRot baseline denoiser/output deltas;
- materialized-route reference numerical noise;
- any dense-provider backend variation that will be accepted in production.

Use those distributions to freeze implementation-stage warning/hard thresholds for block output, full denoiser output and quantization delta. Record them in the training-run manifest before large training. They are screening gates, not substitutes for decoded-media review.

## 9. Export, quantization and checkpoint contracts

### 9.1 BF16 export

For every converted head, fold `query_route @ q_proj` in FP32, assemble `[Q_effective; V]` as `qv_proj.weight [14336,5376]`, round once to BF16, and remove training-only R tensors. Preserve native token-refiner QKV weights untouched.

Before writing the canonical artifact, compare unfused training-form execution against folded deployment execution on held-out activations. The fold must be exact in real arithmetic and differ only by the declared finite-precision/export path. Any larger semantic discrepancy is an export bug.

The exporter must produce a deterministic tensor manifest, parent hash/revision, training checkpoint identity, code commit, tensor counts/shapes/dtypes and a full-file SHA-256 after writing. Never infer provenance from filename alone.

### 9.2 INT8 ConvRot export

Quantize **from the accepted folded BF16 Keyless artifact**, not from the training q/R representation. Retain the target model's full core-heavy policy: qv projection, out projection, fc1 and fc2 for each of 50 main blocks are the 200 heavy quantized linears; sensitive/non-core modules remain at their source precision unless a separately validated recipe changes that.

Use the live Comfy `TensorWiseINT8Layout` ConvRot implementation, per-channel scaling and groupsize 256 if the implementation-time audit confirms the same contract. QV input width 5,376 and output width 14,336 are both compatible with 256-wide grouping; this arithmetic fact does not remove the need to test the current kernel/layout implementation.

Give the Keyless quantization recipe a distinct identity such as `minimax_h3_keyless_core50_200_v1`. Update output-row scale/descriptor counts for the smaller QV projection. Do not reuse metadata claiming the parent QKV recipe unchanged.

Validation must compare:

- Keyless BF16 vs Keyless INT8 ConvRot tensor/denoiser behavior;
- original QKV BF16 vs original QKV INT8 ConvRot as the calibration baseline;
- actual decoded generation quality and audio/reference behavior;
- actual checkpoint bytes, resident model memory and peak runtime memory.

A Keyless BF16 result that is good but unusually fragile under ConvRot has not met the production target.

## 10. LoRA, DoRA and adapter policy

Existing adapters are not generically architecture-preserving.

- A packed QKV LoRA/DoRA may contain a nonzero K delta. There is no general exact destination for that delta in `h3_keyless_core50_v1`.
- Never silently split a QKV adapter, discard its K slice and call the result compatible.
- Adapters that provably touch only unchanged non-attention tensors may be supported after strict key/shape audit.
- Native Keyless adapters may target QV/out/MLP/etc. using a new format/metadata identity.
- To preserve the behavior of a desired legacy attention adapter, merge/apply it to the QKV teacher first and distill that merged teacher into a separately identified Keyless derivative.

Comfy's live LoRA key mapping must be re-audited because it currently has MiniMax-H3-specific handling. Incompatible adapter loads fail with a precise architecture error; they do not partially apply by default.

## 11. Validation and release evidence

### 11.1 Four-way baseline matrix

Every release candidate is evaluated against:

1. original target BF16 QKV;
2. original target INT8 ConvRot QKV;
3. Keyless BF16;
4. Keyless INT8 ConvRot.

Use identical prompts, reference assets, seeds, sampler schedules, latent geometry and decode path where a comparison is intended to be paired. Record model SHA, Comfy commit, all companion commits and runtime/kernel provenance.

### 11.2 Structural/numerical tests

Required automated coverage includes:

- strict checkpoint detection and contradictory-format rejection;
- core50 converted / refiner-QKV scope and tensor counts;
- QV split order;
- route RMSNorm epsilon and 96-channel split-half partial RoPE;
- retrieval V remains untouched by route norm/RoPE/Untwist;
- training-factor fold/export parity;
- dense materialized-route FP32/BF16 oracle checks;
- value-domain gather keeps V, positions, log measure and exact-block metadata aligned;
- provider/cache/history identities separate QKV and Keyless;
- BF16 and INT8 loader/offload/device movement;
- ConvRot metadata/dequant/reference linear tests;
- interruption/rollback/reentrancy and repeated model-patch application;
- unsupported ecosystem combinations fail closed without stale state.

Hosted CPU tests establish contracts only. At least one generic CUDA dense run and the production SM120 path must be tested separately.

### 11.3 Ecosystem integration matrix

Test the live production combinations actually used, not only isolated imports:

- native Keyless dense;
- Spectrum actual + forecast history;
- Untwist alone and composed with Spectrum/Sol;
- Sol materialized reference and native fused Keyless;
- VDN Keyless softmax + retrained linear branch, including audio controls;
- Flow Mixed-Grid weighted measure and relevant handoff path;
- Flow + VDN + Sol + Spectrum composition when the live branches support it;
- Continuum native/refine/reference/audio workflows;
- KJ/Sage provider path where used.

Preserve established logical/actual/forecast counts from each controlled baseline. Any extra hidden transformer evaluation is a regression unless deliberately designed and documented.

### 11.4 Real generation review

Numerical closeness is not sufficient. Decode and review matched media for at least:

- prompt/action adherence;
- object/attribute binding and text rendering where present;
- visual detail and artifacts;
- motion quality and fast-motion behavior;
- temporal continuity / boundary behavior;
- subject/reference identity and reference strength;
- framing/composition;
- audio content, speech/whisper behavior where applicable, stereo integrity and A/V synchronization;
- Continuum/Flow transition behavior.

Use a fixed regression set plus a held-out set. Keep raw/pre-guidance/final latents when an existing ecosystem diagnostic requires them. Do not claim better quality, prompt adherence or temporal consistency unless a separate controlled result establishes it; the release objective is parity plus efficiency.

### 11.5 Performance and VRAM

Measure cold and warmed runs separately on the same machine/runtime:

- actual checkpoint bytes;
- resident model allocation;
- peak allocated/reserved VRAM;
- QV projection time;
- route materialization/fused-route time;
- attention time;
- transformer-evaluation time;
- sampler time;
- end-to-end generation time;
- launch counts and HBM traffic where tooling supports them.

Compare materialized Keyless against fused Keyless as well as QKV. Native fused telemetry must prove that no full global routing tensor was materialized on the claimed path. Report median and dispersion over paired runs; expand repetitions only when variance prevents a decision. A theoretical FLOP reduction is not a benchmark result.

### 11.6 Release gates

A production release requires all of the following:

- BF16 Keyless passes native generation/media gates;
- INT8 ConvRot Keyless passes its own parity gates;
- strict loader/checkpoint provenance is complete;
- required ecosystem combinations pass structural and real-runtime tests;
- VDN uses a Keyless-specific learned branch rather than hidden teacher-K dependence;
- native optimized path passes arithmetic and memory-materialization gates;
- actual VRAM/checkpoint reductions are measured;
- performance is reported without unsupported extrapolation;
- unresolved limitations are documented in README/release notes.

A reference-only BF16 implementation may be published as experimental before the optimized kernel exists, but it must be labeled `materialized_route` and must not claim production activation-memory savings or full ecosystem parity.

## 12. Ordered implementation phases

### Phase 0 — live provenance and safe workspaces

**Objective:** establish implementation source of truth before editing.

Re-fetch Keyless `main`; target HF revision/metadata; Comfy `master`; Spectrum/Sol/VDN/Flow/Continuum/Untwist/KJ/comfy-kitchen heads; active PR bases/heads/reviews/CI; and local installed source/kernel/compiler provenance if real runtime tests will be called authoritative. Compare against both evidence manifests.

**Exit:** exact refs recorded; target full-file hashes verified after acquisition; separate implementation mirrors/branches created without disturbing diagnostic PR topology; GitHub checkpoint pushed.

### Phase 1 — diagnostics and model contract

**Objective:** reproduce weight-space probes and lock checkpoint/model semantics.

Implement bounded subspace diagnostics, checkpoint validator, static `minimax_h3_keyless_contract_v1`, tensor/provenance manifest and architecture tests. Reproduce the sampled residual result and, only if useful for initialization, extend diagnostics across all heads.

**Exit:** universal exact fixed-V conversion is either independently reproduced as false or new evidence forces design review; loader/model signatures are testable without training.

### Phase 2 — reference Keyless model and loader

**Objective:** execute untrained/calibrated `h3_keyless_core50_v1` in ComfyUI.

Implement KeylessAttention, training form, deployable QV form, route materialization, partial RoPE/RMSNorm, strict loader and dense reference backend. Retain two native QKV refiners. Add export-fold unit tests before training.

**Exit:** deterministic forward works on bounded inputs; reference attention agrees with explicit oracle; no fake K exists in core blocks; ordinary H3 conditioning/packing remains intact.

### Phase 3 — pilot block distillation

**Objective:** prove fit capacity before a long campaign.

Train blocks 0/25/49 through §8.2, compare identity/LS initialization and freeze/unfreeze stages, preserve metrics/artifacts.

**Exit:** all three depth pilots pass fixed held-out gates. Failure returns to architecture investigation; do not continue by weakening gates.

### Phase 4 — progressive core50 BF16 distillation

**Objective:** convert all 50 blocks with rollback-safe checkpoints.

Run progressive early→late block/group sweep using same-input QKV teacher blocks. Export folded BF16 checkpoints periodically and run fixed full-denoiser checks.

**Exit:** all blocks accepted; no training-only R appears in deployable artifact; full model is ready for media evaluation.

### Phase 5 — full-model parity / corrective training

**Objective:** establish native BF16 quality.

Run full fixed numerical and generation suite. If needed, execute the bounded escalation ladder in §8.4.

**Exit:** BF16 real-generation gate passes or project remains research-only with the failure documented. Do not proceed to production claims on structural success alone.

### Phase 6 — ecosystem reference compatibility

**Objective:** make the materialized-route Keyless model compose correctly before kernel optimization.

Implement/update semantic-contract consumers in Spectrum, Untwist, Sol reference bridge, Flow and KJ as required. Implement the VDN Keyless softmax path and design/train its learned-branch variant. Regression-test Continuum.

**Exit:** target compositions execute with correct ownership/history/row domains and decoded media; no consumer requires hidden persistent K in a production candidate.

### Phase 7 — BF16 deployment export and native optimized attention

**Objective:** remove route materialization from the production attention path.

Finalize deterministic BF16 exporter and implement Sol/SM120 fused value-route backend. Preserve weighted/mapped/sink semantics and routing-only Untwist. Add arithmetic, specialization, cache and HBM/materialization telemetry.

**Exit:** fused backend passes materialized-Keyless oracle and real SM120 gates and demonstrates zero global route materialization on the measured path.

### Phase 8 — INT8 ConvRot production artifact

**Objective:** produce the user's primary deployment format.

Quantize accepted folded BF16 using the audited Keyless recipe; implement/load through native quant ops; run BF16→INT8 numerical, media, VRAM and timing suite.

**Exit:** Keyless INT8 passes production quality/compatibility gates and exact artifact hash/provenance is recorded.

### Phase 9 — release hardening

**Objective:** make the repository independently usable.

Finalize README, install/loader documentation, checkpoint manifests, compatibility matrix, limitations, example workflows, tests/CI, licensing/NOTICE and release notes. Publish companion PRs without rewriting existing stacks.

**Exit:** a clean checkout plus published model artifacts can reproduce the documented reference and production paths; all remaining empirical limitations are explicit.

## 13. Rejected, falsified and deferred approaches

| Approach/hypothesis | Status | Evidence/scope |
|---|---|---|
| Existing H3 can always be converted exactly to fixed-V per-head Keyless by algebra alone | **Falsified for universal claim** | Multi-head subspace condition is not guaranteed; sampled H3 head shows very large raw bilinear residual. Scope: fixed-V per-head linear formulation before H3 nonlinear routing transforms. |
| Keyless gives H3 a literal 50% KV-cache reduction | **Not applicable** | H3 diffusion attention has no autoregressive KV cache. H3 benefits are weight/projection/transient-bandwidth reductions. |
| Keep a dead/fake K projection for compatibility | **Rejected design** | Erases weight benefit and lets consumers depend on the wrong semantics. Use explicit contracts/fallbacks instead. |
| Materialize a full route tensor and call the job memory-optimized | **Rejected as production claim** | Useful correctness fallback, but retains a K-sized routing activation. |
| Preserve VDN by changing only its softmax provider | **Falsified for behavior preservation** | Audited `hybrid.py` learned linear branch consumes raw Q/K/V and optional text K/V. It needs Keyless-specific adaptation/training. |
| Existing packed-QKV LoRA/DoRA is automatically Keyless-compatible | **Falsified in general** | Arbitrary ΔK has no exact generic Keyless destination. |
| Convert all 52 attention modules in the first release | **Deferred, not falsified** | Two refiners are cheap, text-side and upstream of all blocks; core50 captures nearly all target systems benefit with lower conditioning risk. |
| Cross-head Slim-style K recovery | **Not selected, not disproved** | May be algebraically possible under broader rank conditions but changes layout/compute and is not value-only per-head routing. |
| Train/distill primarily from INT8 ConvRot | **Rejected canonical path** | Quantization error would be baked into architecture conversion; BF16 is the authoritative teacher. INT8 remains a deployment/parity target. |
| Keyless inherently improves visual quality or prompt adherence | **Unsupported** | Supplied papers do not establish MiniMax-H3 audiovisual improvements. Release target is parity plus efficiency. |
| Current Sol QKV exact-mode gate proves Keyless exactness | **Falsified implication** | Different attention semantics and routing transforms require an independent Keyless oracle/gate. |
| Apply Untwist to raw V | **Rejected semantics** | Untwist modifies routing/key behavior; retrieval V must remain unchanged. |

## 14. Unresolved questions and mandatory re-checks

### 14.1 Genuine empirical unknowns

- Can normalized value-space routing reach native audiovisual quality parity on this target checkpoint?
- Is factorized Q training materially better/easier than learning the deployable Q-effective projection directly?
- Does route_norm initialized from k_norm remain optimal after V moves, or does training consistently learn a materially different scale?
- How much data/sweep repetition is required before block-local fit generalizes to held-out prompts/assets?
- Is a global correction stage necessary after progressive block distillation?
- What Keyless feature schema best preserves VDN's learned linear branch without teacher K?
- Does fused Q+V routing improve real SM120 transformer/end-to-end time after route normalization/RoPE/preprocessing costs?
- What are the actual peak-VRAM and HBM savings under real allocator/provider composition?
- Is an all-52 Keyless checkpoint worthwhile after core50 is stable?

These must be answered by implementation evidence, not filled in by prose.

### 14.2 Re-check before implementation

The implementer must re-fetch rather than trust these moving assumptions:

- ComfyUI default branch/source, H3 model/detection/quant APIs and current core BSA behavior;
- current target HF revision, metadata and published artifacts;
- active Spectrum #110, Sol #14, VDN #18, Flow #30/#46, Continuum #24 and any successor PRs;
- Sol provider-v4 and SM120 packaged-kernel provenance;
- VDN branch/readout/provider APIs;
- Flow measure/external-sequence contracts;
- Untwist preprocess/reference-selection contract;
- KJ H3/Sage patches;
- comfy-kitchen ConvRot and sparse-kernel APIs;
- local production `patcher/stack`, compiler/CUDA/CuTe provenance before matched runtime claims.

When source moved, preserve semantic invariants from this design rather than forcing stale function names. Any justified deviation must be written into the implementation commit/PR and design addendum with the evidence that required it.

## 15. Repository, branch and artifact discipline

`MiniMax-H3-Keyless` owns the base model, training/export tooling, reference runtime, tests and release documentation. Companion changes belong in their existing repositories on **new implementation mirrors/branches** based on the appropriate live production branch. Do not rewrite or repurpose the Sol #11/#12/#13, VDN #15/#16/#17 or Flow #43/#44/#45 diagnostic stacks.

Checkpoint to GitHub before destructive/hostile investigation, risky refactors, long training/test campaigns and after substantial progress. Training/model artifacts are too large for git; store manifests/hashes/configs/metrics in git and checkpoint model files in the designated model artifact store with immutable names.

For any important out-of-tree artifact record:

- exact path/URI;
- producing code commit;
- parent model/checkpoint hash;
- command/config;
- content hash;
- whether it is required for resume, audit, release or may be reproduced/discarded.

Do not inventory disposable caches.

No indispensable out-of-tree artifact is known from the interrupted design session. The authoritative evidence preserved by that session is committed under `docs/evidence/`. The sampled 96.3% statement must be reproduced in Phase 1 before being treated as a quantitative implementation gate.

## 16. Definition of completion

This design is complete as an implementation specification. The **project implementation is not complete** until the empirical release gates in §§11–12 are satisfied.

Verified design facts include the target lineage/topology, current QKV H3 math, conversion limitation, selected core50 architecture, checkpoint forms, quantified parameter/projection reductions and concrete ecosystem incompatibilities observed in audited source. Architectural conclusions include normalized value routing, foldable query-side training factors, packed deployable QV, explicit semantic/provider contracts, progressive block distillation, VDN-specific retraining and a fused value-route production kernel.

The major unresolved risks are empirical quality parity, VDN learned-branch parity, optimized-kernel speed/VRAM behavior and final INT8 ConvRot robustness. An implementation that passes unit tests but has not passed matched real-generation and runtime gates must remain explicitly experimental.
