# MiniMax-H3 Keyless implementation design

Status: architecture specification with empirical implementation gates. No trained Keyless model, CUDA implementation, generation-quality result, or speedup is established by this document. Audit date: 2026-09-16.

## 1. Product and decisions

Build a value-routed attention derivative of `xmarre/MiniMax-H3-Pruned-Ref-Delta-Fused-r1024-ComfyUI`. Preserve its pruned timestep conditioning, fused reference behavior, audio/video schedules, packing, and existing conditioning interfaces. Train against the corresponding BF16 checkpoint; export both BF16 and native ComfyUI INT8 ConvRot checkpoints. The deployment baseline is `MiniMax-H3-Pruned-Ref-Delta-Fused-r1024-comfy-int8-convrot.safetensors`, including quantized `fc2`.

The chosen first production architecture replaces all **50 diffusion-block attention modules**, retains the **two original QKV token-refiner blocks**, and declares this scope explicitly as `h3_keyless_core50_v1`. The refiner processes text and can run in `preprocess_text_embeds`; its cost is not proportional to the large audiovisual sequence at each transformer evaluation. Replacing it immediately risks conditioning drift upstream of every block for only two more removed projections. It remains real, live QKV attention, not a fake compatibility module. A later all-52 variant requires a separately identified checkpoint and conditioning-quality validation; it is not a prerequisite for core50 release. The project must never describe core50 as containing no K weights anywhere.

Use a packed QV projection and a **normalized, position-transformed routing view of V**, while retrieving the unmodified V. Learn a per-head query factorization before query normalization so the factors can be folded at export. Do not retain a dead K projection, recover K through a full cross-head transform, or silently route through the original QKV teacher at runtime.

A materialized routing tensor is permitted in the reference/compatibility backend. The intended optimized path derives routing tiles from the selected V domain inside the kernel. Compatibility output equivalence and native-kernel memory efficiency are separate gates.

Distillation is mandatory. Bounded measurements already contradict exact fixed-V raw linear conversion in sampled teacher heads. Whether this derivative can meet audiovisual quality parity remains an empirical research risk. The implementation phases below prescribe the response to failure without pretending that a training recipe guarantees success.

## 2. Evidence and provenance

`evidence/source_manifest.json` records exact source heads, active PR heads/bases, both complete safetensors headers, model metadata, and supplied-paper hashes. Headers were obtained by bounded HTTP byte ranges at a pinned model revision. They establish storage topology and dtype, not a full-file checksum verification. Sampled BF16 weight probes are recorded separately. No local CUDA runtime or installed user ComfyUI environment was inspected; repository source is not installed-kernel/compiler provenance.

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

Additional worktrees inspected: Spectrum #110 `78a9a5b36c185a55af59a44c073bc7f8e534bc97`; Sol #14 `8b049e39d000b0d283f2f01e8477f74cf5c2d608`; VDN #18 `333d63f81d33fe29dc1f1f637c5f4a4396880f99`; Flow #30 `f6a940fdc2fb6d31249a487aff5e4a1297151e9c` and #46 `5970116ef5a04c405f0f2a4c873585a7097b15df`; Continuum #24 `4d9c5f4a01d35f49ef2890cf67653adde59ce571`.

Preserve existing topology. Sol #11 → #12 → #13, VDN #15 → #16 → #17, and Flow #43 → #44 → #45 are distinct W/E/M diagnostic stacks. Production Sol #14 branches from #9's weighted-measure head; VDN #18 branches from #8's audio-fidelity head; Flow #46 branches from #41. Do not rewrite, repurpose, merge, or subsume diagnostic evidence into Keyless work. Use new implementation mirrors for any later companion changes. Source manifests include actual base SHAs and ref names.

At inspection, Sol #14 CPU-contract CI and VDN #18 CI were successful; Flow #46 CI failed at `ruff format --check .`, while its `source-contracts` job succeeded. These are upstream observations, not tests of Keyless. The three production PRs had no submitted review records or inline comments; VDN's bot conversation report described a review without actionable comments, while Sol/Flow bot reports said review skipped. Re-fetch reviews and CI before implementation. ComfyUI #16239 (key-measure execution), #16344 (H3 BSA composition), #16245 (compiler opt-out) and #16362 (draft sparse attention) are moving relevant interfaces; do not presume their proposals are merged or required.

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


Implementation contracts, training gates, validation phases and final audit are being completed in subsequent commits. This checkpoint is not the finished specification.
