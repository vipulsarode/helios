"""
cp_v1_ring_backward.py  —  Context Parallelism, ring backward (Stage 3).

WHAT THIS ADDS TO STAGE 2:
  Stage 2 ran forward under no_grad because dist.isend/irecv are invisible to
  autograd -- a bare backward through them would silently produce garbage.
  Stage 3 fixes that by putting the ENTIRE ring (forward comm + attention math
  + backward comm) inside ONE torch.autograd.Function. Once you hand-write
  .backward(), you own every derivative inside it, softmax included. That is
  why the FlashAttention backward math you just read shows up here and never
  did in DP/TP/PP: this is the first custom Function whose forward CONTAINS a
  softmax rather than merely moving tensors around one.

INVARIANT (do not break): rank i holds shard i, and only shard i, of every
  sequence-sharded tensor -- in forward (q,k,v,o) and in backward (dq,dk,dv).
  Backward is reduce_scatter semantics expressed as a ring, NOT all_reduce.

TEST TARGET: gradients from a single-process full-attention reference, sliced
  to this rank's shard. Same oracle strategy as cp_v0 -- build the truth on one
  process, shard the SAME seeded inputs, assert the CP grads match the slice.

  run:  torchrun --nproc_per_node=4 cp_v1_ring_backward.py
"""

import torch
import torch.distributed as dist
import torch.nn.functional as F
import math
from cp_v1 import _block_attention, _merge


# ==========================================================================
# 1. FORWARD, instrumented for backward.
#
#    Stage 2's forward returned only o_local and discarded the softmax stats.
#    Backward cannot run without them. This version must ADDITIONALLY produce,
#    for THIS rank's queries, the saved logsumexp L = m + log(l) -- one scalar
#    per query row, shape [B, nh, S_local, 1]. That single tensor is what lets
#    backward reconstruct P by  P = exp(S - L)  without ever having stored the
#    S_local x S_total probability matrix. This is the selective-checkpointing
#    trade at the heart of flash-style attention.
#
#    Everything else (the two-buffer rotation, the cp_size compute rounds /
#    cp_size-1 transfers, the online-softmax merge) is IDENTICAL to Stage 2.
#    Do not redesign it. Lift it.
# ==========================================================================
def _ring_forward(q_local, k_local, v_local, cp_group, cp_size, send_to, recv_from):
    """
    INPUTS : q_local, k_local, v_local  each [B, nh, S_local, hd]
    RETURNS: o_local [B, nh, S_local, hd]
             L_local [B, nh, S_local, 1]   (saved logsumexp for backward)

    This is your validated Stage 2 loop with two changes:
      - carry m and l to the end (you already do) and fold them into
        L = m + log(l) before returning.
      - return L alongside o.
    No other change. If you find yourself altering the rotation, stop.
    """
    B, nh, S_local, hd = q_local.shape

    o_run = torch.zeros_like(q_local)
    m_run = torch.full((B, nh, S_local, 1), -torch.inf)
    l_run = torch.zeros((B, nh, S_local, 1))
    
    cur_k = k_local.clone()
    cur_v = v_local.clone()

    recv_k = torch.empty_like(k_local)
    recv_v = torch.empty_like(v_local)

    for i in range(cp_size):

        if i < cp_size - 1:
            k_send_op = dist.P2POp(op=dist.isend, tensor=cur_k, peer=send_to, group=cp_group,tag=11)
            v_send_op = dist.P2POp(op=dist.isend, tensor=cur_v, peer=send_to, group = cp_group, tag=22)

            k_recv_op = dist.P2POp(op=dist.irecv, tensor=recv_k, peer=recv_from, group = cp_group, tag=11)
            v_recv_op = dist.P2POp(op=dist.irecv, tensor=recv_v, peer=recv_from, group = cp_group, tag=22) 

            reqs = dist.batch_isend_irecv([k_send_op, v_send_op, k_recv_op, v_recv_op]) 

        o_blk, m_blk, l_blk = _block_attention(q_local, cur_k, cur_v)
        o_new, m_new, l_new = _merge(o_run, m_run, l_run, o_blk, m_blk, l_blk)
        o_run, m_run, l_run = o_new, m_new, l_new

        if i < cp_size - 1:
            for req in reqs:
                req.wait()

            cur_k, recv_k = recv_k, cur_k
            cur_v, recv_v = recv_v, cur_v

    return o_run/l_run, m_run + torch.log(l_run)


    


# ==========================================================================
# 2. Per-block LOCAL backward  (the FlashAttention-2 backward math, one block).
#
#    Given THIS rank's fixed (q_local, o_local, dO_local, L_local) and ONE
#    visiting KV block (k_blk, v_blk), produce that block's contribution to
#    the three gradients:
#
#        recompute S = (q_local @ k_blk^T) * scale
#        recompute P = exp(S - L_local)                 # L makes this normalized
#        dV_blk = P^T @ dO_local
#        dP     = dO_local @ v_blk^T
#        D      = rowsum(dO_local * o_local)            # correction term
#        dS     = P * (dP - D)                          # softmax Jacobian, collapsed
#        dQ_contrib = dS @ k_blk
#        dK_blk = dS^T @ q_local
#
#    UNIT TEST you get for free: each row of dS must sum to ~0. Assert it here.
#    A nonzero row-sum means D is wrong, and you want to catch that locally,
#    not three hops downstream in a corrupted partial.
#
#    NOTE the split: dV_blk / dK_blk belong to k_blk/v_blk's OWNER (they must
#    travel). dQ_contrib belongs to THIS rank's queries (it stays). That split
#    is the whole reason backward has a rotating part and a local part.
# ==========================================================================
def _block_backward(q_local, o_local, dO_local, L_local, k_blk, v_blk, scale):
    """
    RETURNS: dq_contrib [B, nh, S_local, hd]   -> accumulate LOCALLY
             dk_blk     [B, nh, S_blk,   hd]   -> ride the rotating dK partial
             dv_blk     [B, nh, S_blk,   hd]   -> ride the rotating dV partial
    """


    S = q_local@k_blk.transpose(-2,-1) * scale
    P = torch.exp(S - L_local)
    dV_blk = P.transpose(-2,-1) @ dO_local
    dP = dO_local@v_blk.transpose(-2, -1)
    D = (dO_local * o_local).sum(dim=-1, keepdim=True)
    dS = P*(dP - D)
    dQ_contrib = (dS @ k_blk) * scale
    dK_blk = (dS.transpose(-2,-1) @ q_local) * scale

    # assert torch.allclose(dS.sum(dim=-1), torch.zeros_like(dS.sum(dim=-1)), atol=1e-5), "dS rows don't sum to zero"

    return dQ_contrib, dK_blk, dV_blk

# ==========================================================================
# 3. The BACKWARD ring.
#
#    STRUCTURE relative to forward -- read this before writing:
#
#    Forward rotated 2 tensors (K, V) with cp_size-1 transfers. Backward
#    rotates 4 tensors: the KV blocks AGAIN (you need them present to compute
#    each block's gradient), PLUS a dK partial and a dV partial that travel
#    WITH their blocks, accumulating one rank's contribution per hop.
#
#    THE ONE STRUCTURAL DIFFERENCE FROM FORWARD: cp_size transfers, not
#    cp_size-1. The final transfer carries no new computation -- it is pure
#    delivery, walking each completed partial the last hop home to its owner.
#    If you mirror the forward loop exactly you will drop that transfer, and
#    the signature is unmistakable: every rank's dK is missing exactly one
#    contribution -- its own.
#
#    WHAT ROTATES AND WHY IT LANDS HOME:
#      - A KV block and its dK/dV partials move together around the ring.
#      - At each hop, the current holder adds its _block_backward contribution
#        into the partials.
#      - After the block has visited all cp_size ranks, its partials hold the
#        full sum and sit on the owner. TRACE ONE BLOCK ON PAPER before coding:
#        follow B2's dK partial for cp_size steps and confirm it lands on rank 2.
#        That trace is what separates "works first try" from "spins forever."
#
#    WHAT STAYS LOCAL:
#      - dQ for this rank's queries. Every contribution to it is computed HERE
#        (the blocks come to us), so it accumulates in a local buffer and never
#        touches the wire.
#
#    BUFFERS: same two-role alternation as Stage 2, now for 4 rotating tensors
#    (KV block + dK partial + dV partial ... note K and V and their two partials
#    -> reason out the exact tensor count and say it out loud before allocating).
#    Ordering is still post -> compute -> wait -> swap, for the same overlap
#    reason. dQ buffer is NOT rotated; it is added into in place.
#
#    Return dq_local, dk_local, dv_local -- each [B, nh, S_local, hd], each the
#    COMPLETE gradient for THIS rank's shard.
# ==========================================================================
def _ring_backward(q_local, k_local, v_local, o_local, dO_local, L_local,
                   cp_group, cp_size, send_to, recv_from, scale):
    """
    RETURNS: dq_local, dk_local, dv_local   each [B, nh, S_local, hd]
    """
    
    if cp_size == 1:
        dq, dk, dv = _block_backward(q_local, o_local, dO_local, L_local, k_local, v_local, scale)
        return dq, dk, dv
        
    cur_k = k_local.clone()
    cur_v = v_local.clone()

    recv_k = torch.empty_like(k_local)
    recv_v = torch.empty_like(v_local)

    recv_dK = torch.empty_like(k_local)
    recv_dV = torch.empty_like(v_local)

    cur_dQ = torch.zeros_like(q_local)
    cur_dK = torch.zeros_like(k_local)
    cur_dV = torch.zeros_like(v_local)
    
    for step in range(cp_size):

        k_send = dist.P2POp(dist.isend, cur_k, peer=send_to, group=cp_group, tag = 11)
        v_send = dist.P2POp(dist.isend, cur_v, peer=send_to, group=cp_group, tag = 22)
        k_recv = dist.P2POp(dist.irecv, recv_k, peer=recv_from, group=cp_group, tag = 11)
        v_recv = dist.P2POp(dist.irecv, recv_v, peer=recv_from, group=cp_group, tag = 22)
        dK_recv = dist.P2POp(dist.irecv, recv_dK, peer=recv_from, group=cp_group, tag = 33)
        dV_recv = dist.P2POp(dist.irecv, recv_dV, peer=recv_from, group=cp_group, tag = 44)

        dQ, dK, dV = _block_backward(q_local, o_local, dO_local, L_local, cur_k, cur_v, scale)
        
        cur_dQ = cur_dQ + dQ.clone()
        cur_dK = cur_dK + dK.clone() 
        cur_dV = cur_dV + dV.clone()

        dK_send = dist.P2POp(dist.isend, cur_dK, peer=send_to, group=cp_group, tag = 33)
        dV_send = dist.P2POp(dist.isend, cur_dV, peer=send_to, group=cp_group, tag = 44)

        reqs = dist.batch_isend_irecv([k_send, k_recv, v_send, v_recv, dK_recv, dK_send, dV_recv, dV_send])

        for req in reqs:
            req.wait()

        cur_k, recv_k = recv_k, cur_k
        cur_v, recv_v = recv_v, cur_v
        cur_dK, recv_dK = recv_dK, cur_dK
        cur_dV, recv_dV = recv_dV, cur_dV

    return cur_dQ, cur_dK, cur_dV

# ==========================================================================
# 4. Autograd Function tying forward and backward together.
#
#    forward(ctx, ...):  run _ring_forward, stash for backward everything the
#      backward needs and cannot recompute -- q,k,v,o,L and the ring metadata
#      (group, size, neighbors, scale) -- via ctx.save_for_backward / ctx attrs.
#      Return o_local ONLY (L is an internal; the module above sees just o).
#
#    backward(ctx, grad_o):  grad_o IS dO_local. Pull the saved tensors, call
#      _ring_backward, return one gradient per forward input positionally, with
#      None for every non-tensor arg (cp_group, cp_size, ...). Getting the None
#      padding wrong is the classic first-run crash -- count the forward args.
# ==========================================================================
class RingAttention(torch.autograd.Function):

    @staticmethod
    def forward(ctx, q_local, k_local, v_local, cp_group, cp_size, send_to, recv_from):
        o_local, L_local = _ring_forward(q_local, k_local, v_local, cp_group, cp_size, send_to, recv_from)
        ctx.save_for_backward(q_local, k_local, v_local, o_local, L_local)
        ctx.cp_group = cp_group
        ctx.cp_size = cp_size
        ctx.send_to = send_to
        ctx.recv_from = recv_from
        ctx.scale = 1.0 / (q_local.shape[-1] ** 0.5)
        return o_local



    @staticmethod
    def backward(ctx, grad_o):
        q_local, k_local, v_local, o_local, L_local = ctx.saved_tensors
        dq_local, dk_local, dv_local = _ring_backward(q_local, k_local, v_local, o_local, grad_o, L_local,
                   ctx.cp_group, ctx.cp_size, ctx.send_to, ctx.recv_from, ctx.scale)
        return dq_local, dk_local, dv_local, None, None, None, None


def ring_attention(q_local, k_local, v_local, cp_group, cp_size, send_to, recv_from):
    return RingAttention.apply(q_local, k_local, v_local, cp_group, cp_size, send_to, recv_from)


# ==========================================================================
# 5. Self-checking gloo test.
#
#    Oracle: single-process full attention on the SAME seeded inputs. Run its
#    backward, keep dq_full/dk_full/dv_full. Then shard the inputs, run the CP
#    ring_attention forward+backward, and assert each CP grad matches the
#    reference sliced to [lo:hi]. Forward output check too, as a Stage 2 regress.
#
#    Checklist the test must cover (do not trust a single green line):
#      - forward o matches ref slice
#      - dq matches ref slice   (LOCAL path -- isolates the attention math)
#      - dk matches ref slice   (ROTATING path -- isolates the partial routing)
#      - dv matches ref slice   (ROTATING path, different tensor than dk)
#      dq green + dk red  => routing bug, math is fine.
#      dq red             => local backward math bug, look at _block_backward.
#
#    Run at cp_size in {1, 2, 4}. cp_size=1 exercises the no-comm degenerate
#    path (Stage 6 configures CP off through it). cp_size=2 hides ordering bugs
#    (swap is its own inverse) -- 4 is the honest test.
#
#    run:  torchrun --nproc_per_node=4 cp_v1_ring_backward.py
# ==========================================================================
def _run_test():
    dist.init_process_group("gloo")
    rank, world = dist.get_rank(), dist.get_world_size()
    cp_group, cp_size = dist.group.WORLD, world
    send_to = (rank + 1) % cp_size
    recv_from = (rank - 1) % cp_size

    torch.manual_seed(0)
    B, nh, S_total, hd = 2, 4, 16, 16
    assert S_total % cp_size == 0
    S_local = S_total // cp_size
    lo, hi = rank * S_local, (rank + 1) * S_local
    scale = 1.0 / (hd ** 0.5)

    q_full = torch.randn(B, nh, S_total, hd, requires_grad=True)
    k_full = torch.randn(B, nh, S_total, hd, requires_grad=True)
    v_full = torch.randn(B, nh, S_total, hd, requires_grad=True)

    # ---- oracle ----
    ref = F.scaled_dot_product_attention(q_full, k_full, v_full, is_causal=False)
    ref.sum().backward()
    ref_dq, ref_dk, ref_dv = q_full.grad.clone(), k_full.grad.clone(), v_full.grad.clone()

    # ---- CP ring: shard SAME inputs, run fwd+bwd ----
    q = q_full.detach()[:, :, lo:hi, :].clone().requires_grad_(True)
    k = k_full.detach()[:, :, lo:hi, :].clone().requires_grad_(True)
    v = v_full.detach()[:, :, lo:hi, :].clone().requires_grad_(True)

    o = ring_attention(q, k, v, cp_group, cp_size, send_to, recv_from)
    o.sum().backward()

    fwd_ok = torch.allclose(o.detach(), ref[:, :, lo:hi, :].detach(), atol=1e-5)
    dq_ok = torch.allclose(q.grad, ref_dq[:, :, lo:hi, :], atol=1e-5)
    dk_ok = torch.allclose(k.grad, ref_dk[:, :, lo:hi, :], atol=1e-5)
    dv_ok = torch.allclose(v.grad, ref_dv[:, :, lo:hi, :], atol=1e-5)

    print(f"[rank {rank}] fwd={fwd_ok} dq={dq_ok} dk={dk_ok} dv={dv_ok}")
    dist.destroy_process_group()


if __name__ == "__main__":
    _run_test()