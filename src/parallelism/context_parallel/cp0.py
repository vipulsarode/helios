"""
cp_v0.py  —  Context Parallelism, the "gather-everything" baseline (Stage 1).

GOAL: correctness, not efficiency. This is the ORACLE the ring (Stage 2) is
tested against. The only new idea is "shard the sequence"; attention is made
trivially correct by all_gathering the FULL K,V onto every rank and running
ordinary attention.

Forward :  all_gather(K), all_gather(V)  -> every rank has full KV -> plain attn
Backward:  the gather's dual, reduce_scatter, routes each K/V-shard gradient
           (summed over all ranks that used it) back to its owner.

The gather is wrapped in a custom autograd Function so backward fires that
reduce_scatter automatically. That Function is the heart of CP v0.
"""

import os
import torch
import torch.distributed as dist
import torch.nn.functional as F


# ==========================================================================
# 1. Differentiable all-gather along the SEQUENCE axis.
#    Forward:  gather every rank's shard -> full-sequence tensor.
#    Backward: reduce_scatter the incoming grad -> each rank gets the summed
#              gradient for ITS shard only.
# ==========================================================================
class GatherSeq(torch.autograd.Function):

    @staticmethod
    def forward(ctx, local_x, cp_group, cp_size, dim = 1):
        # local_x: [B, S_local, H]   (this rank's sequence shard)
        ctx.cp_group = cp_group
        ctx.cp_size = cp_size
        ctx.dim = dim
        ctx.cp_rank = dist.get_rank(cp_group)
        # BLANK 1 --------------------------------------------------------
        # all_gather local_x from all cp ranks and concatenate along the
        # SEQUENCE axis (dim=1) to form the full-sequence tensor.
        #   - make an output list of cp_size empty tensors shaped like local_x
        #   - dist.all_gather(list, local_x, group=cp_group)
        #   - torch.cat(...) along dim=1  -> full_x  [B, S_total, H]
        # (Note: dist.all_gather is NOT differentiable by itself — that's why
        #  we're inside a custom Function and must define backward ourselves.)
        seq = [torch.empty_like(local_x) for _ in range(ctx.cp_size)]
        dist.all_gather(seq, local_x, group=ctx.cp_group)
        full_x = torch.cat(seq, dim = ctx.dim)
        # ----------------------------------------------------------------
        return full_x

    @staticmethod
    def backward(ctx, grad_full):
        # grad_full: [B, S_total, H]  — gradient w.r.t. the FULL gathered tensor.
        # Each rank contributed its shard's queries, so grad_full holds, summed
        # implicitly by autograd across the batch of local computations? NO —
        # think: grad_full on THIS rank is this rank's contribution to the whole
        # sequence's grad. We need, per shard, the SUM across ranks, delivered
        # to that shard's owner. That is exactly reduce_scatter.

        # BLANK 2 --------------------------------------------------------
        # reduce_scatter grad_full (chunked along dim=1 into cp_size pieces)
        # so this rank receives the summed gradient for ITS shard only.
        #   - split grad_full into cp_size chunks along dim=1
        #   - allocate grad_local shaped like ONE chunk
        #   - dist.reduce_scatter(grad_local, list_of_chunks, group=cp_group)
        # chunks = list(torch.chunk(grad_full, chunks = ctx.cp_size, dim = ctx.dim))
        # grad_local = torch.empty_like(chunks[0])
        # dist.reduce_scatter(grad_local, chunks, group=ctx.cp_group)## does not work in glooo

        dist.all_reduce(grad_full, group=ctx.cp_group) # convert this to reduce_scatter once working with gloo
        chunks = list(torch.chunk(grad_full, chunks = ctx.cp_size, dim = ctx.dim))
        grad_local = chunks[ctx.cp_rank]
        # ----------------------------------------------------------------

        # Must return one grad per forward input (local_x, cp_group, cp_size).
        # Only local_x needs a gradient; the other two are non-tensors -> None.
        return grad_local, None, None, None


def gather_seq(local_x, cp_group, cp_size, dim):
    return GatherSeq.apply(local_x, cp_group, cp_size, dim)


# ==========================================================================
# 2. CP v0 attention: shard queries locally, attend against FULL gathered KV.
# ==========================================================================
def cp_v0_attention(q_local, k_local, v_local, cp_group, cp_size, causal=False):
    """
    q_local, k_local, v_local: [B, n_head, S_local, head_dim]
      (already projected + reshaped; each rank holds its own SEQUENCE shard)

    Returns o_local: [B, n_head, S_local, head_dim]  — attention output for
    THIS rank's queries only (output stays sequence-sharded).
    """
    # We gather along the sequence axis. Above, GatherSeq assumed dim=1 == seq.
    # Here seq is dim=2, so either (a) generalize GatherSeq to take a seq_dim
    # arg, or (b) transpose. Pick one and be consistent.
    #
    # BLANK 3 --------------------------------------------------------------
    # 1. gather k_local and v_local along the sequence axis -> k_full, v_full
    #    using the differentiable gather_seq (so backward routes grads home).
    #    q stays LOCAL — we only need this rank's queries.
    # 2. run ordinary attention: q_local (S_local queries) against k_full/v_full
    #    (S_total keys). Because KV is COMPLETE, plain softmax over the full key
    #    axis is exactly correct — no online-softmax, no merge.
    #    You may use F.scaled_dot_product_attention OR hand-rolled softmax(QK^T)@V.
    # 3. NOTE on causal: if causal=True, a naive full-KV mask is WRONG unless you
    #    offset it — this rank's queries are at GLOBAL positions
    #    [cp_rank*S_local : (cp_rank+1)*S_local], not [0:S_local]. Park causal
    #    for now (start with causal=False); we handle it deliberately later.
    
    k_full = gather_seq(k_local, cp_group, cp_size, dim = 2)
    v_full = gather_seq(v_local, cp_group, cp_size, dim = 2)

    
    
    
    o_local = F.scaled_dot_product_attention(q_local, k_full, v_full, is_causal=causal)
    # ----------------------------------------------------------------------
    return o_local


# ==========================================================================
# 3. Self-checking gloo test.
#    Strategy: build ONE full-sequence attention on a single reference, then
#    shard the SAME inputs across ranks, run CP v0, and assert the gathered
#    CP output matches the reference to fp tolerance. Same for gradients.
#
#    run:  torchrun --nproc_per_node=2 cp_v0.py
# ==========================================================================
def _run_test():
    dist.init_process_group("gloo")
    rank = dist.get_rank()
    world = dist.get_world_size()
    cp_group = dist.group.WORLD   # whole world IS the CP group for this test
    cp_size = world

    torch.manual_seed(0)  # SAME seed on every rank -> identical full tensors
    B, n_head, S_total, head_dim = 2, 4, 8, 16
    assert S_total % cp_size == 0
    S_local = S_total // cp_size

    # Full inputs, identical on every rank (same seed). requires_grad for gradcheck.
    q_full = torch.randn(B, n_head, S_total, head_dim, requires_grad=True)
    k_full = torch.randn(B, n_head, S_total, head_dim, requires_grad=True)
    v_full = torch.randn(B, n_head, S_total, head_dim, requires_grad=True)

    # ---- reference: single-process full attention (the oracle) ----
    ref = F.scaled_dot_product_attention(q_full, k_full, v_full, is_causal=False)
    ref.sum().backward()
    ref_grad_k = k_full.grad.clone()

    # ---- CP v0: shard the SAME inputs along seq (dim=2), run distributed ----
    lo, hi = rank * S_local, (rank + 1) * S_local
    q_loc = q_full.detach()[:, :, lo:hi, :].clone().requires_grad_(True)
    k_loc = k_full.detach()[:, :, lo:hi, :].clone().requires_grad_(True)
    v_loc = v_full.detach()[:, :, lo:hi, :].clone().requires_grad_(True)

    o_loc = cp_v0_attention(q_loc, k_loc, v_loc, cp_group, cp_size, causal=False)
    o_loc.sum().backward()

    # ---- check forward: gather CP outputs, compare to reference slice ----
    ref_slice = ref.detach()[:, :, lo:hi, :]
    fwd_ok = torch.allclose(o_loc.detach(), ref_slice, atol=1e-5)

    # ---- check backward: this rank's k grad should match ref's k-grad slice.
    #      (reduce_scatter should have routed the summed grad to the owner.) ----
    bwd_ok = torch.allclose(k_loc.grad, ref_grad_k[:, :, lo:hi, :], atol=1e-5)

    print(f"[rank {rank}] forward_ok={fwd_ok}  backward_ok={bwd_ok}")
    dist.destroy_process_group()


if __name__ == "__main__":
    _run_test()