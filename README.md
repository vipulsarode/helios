# Helios ☀️

**From-scratch 5D parallelism in PyTorch.**

Built to understand distributed training at the primitive level. Every collective, every schedule, every sharding decision implemented and validated before moving on.

---

## What's implemented

| Component | Status |
|---|---|
| Tensor Parallelism | ✅ Validated on CPU |
| Data Parallelism | ✅ Validated on CPU |
| Pipeline Parallelism | 🔄 In progress |
| Training and validating NanoGPT with (DP+TP+PP) | 🔄 In progress |

---

## Validation

Each parallelism strategy is validated against Karpathy's
[nanoGPT](https://github.com/karpathy/nanoGPT) with **0** modifications
to the model architecture. Only the training script is touched.

If the loss curve matches the baseline single-GPU run, the
implementation is correct.

---

## Roadmap
Pipeline Parallelism → Sequence Parallelism → Context Parallelism → Expert Parallelism → FSDP

## Contributions & ideas welcome

A few things I think would be genuinely useful to the community and aren't cleanly available elsewhere:

- **`MemoryTracker`** — Per-rank memory accounting across parameter, gradient, optimizer state, and activation boundaries. Makes ZeRO stage tradeoffs concrete and measurable rather than theoretical.
- **`BubbleProfiler`** — A pipeline schedule visualizer that renders the actual compute/idle timeline per rank, so the bubble ratio is empirical rather than derived from a formula.
- **`CommBenchmark`** — Bandwidth and latency benchmarks for each collective at varying tensor sizes and world sizes.
- **`GradFlowVerifier`** — A test harness that numerically validates gradient correctness against a single-GPU reference for every parallelism strategy. This catches the hard class of bugs where forward is right but backward silently drops or double-counts gradients.
- **Composability tests** — Correctness checks for combined strategies (TP+DP, PP+DP, TP+PP+DP). Composing primitives correctly is where most implementations quietly break.

If any of these interest you, feel free to open an issue or a PR.

---

## References

- [Megatron-LM](https://arxiv.org/abs/1909.08053)
- [ZeRO](https://arxiv.org/abs/1910.02054)
- [Ring Attention](https://arxiv.org/abs/2310.01889)
- [Ultra-Scale Playbook](https://huggingface.co/spaces/nanotron/ultrascale-playbook)
- [Picotron](https://github.com/huggingface/picotron)