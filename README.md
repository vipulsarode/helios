# Helios ☀️

**From-scratch 5D parallelism in PyTorch.**

Helios started as a learning project — a way to deeply understand distributed training by implementing it rather than reading about it. The goal is to eventually grow into something close to production-useful, but the priority right now is to get every primitive right and fully understood before moving on.

---

## Progress

### ✅ Done
- Synchronous collective operations

### 🔜 Up next
- Async collective operations
- Data Parallelism (DDP)
- Fully Sharded Data Parallelism (FSDP)
- Tensor Parallelism (TP)
- Sequence Parallelism (SP)
- Pipeline Parallelism (PP)
- Context Parallelism (CP)
- Expert Parallelism (EP)

---

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