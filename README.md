# MiniMax-H3-Keyless

Production design and implementation repository for a Keyless Attention derivative of `xmarre/MiniMax-H3-Pruned-Ref-Delta-Fused-r1024-ComfyUI`, with the BF16 checkpoint as the canonical teacher/trainable representation and INT8 ConvRot as a first-class deployment target.

The authoritative architecture, training, export, interoperability, validation and release specification is:

`docs/KEYLESS_H3_IMPLEMENTATION_DESIGN.md`

No trained Keyless checkpoint, production CUDA/CuTe kernel, generation-quality parity result or measured speedup is claimed until the empirical gates in that design are completed.
