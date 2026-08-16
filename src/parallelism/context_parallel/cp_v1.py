"""
cp_v1_ring.py  —  Context Parallelism, ring forward (Stage 2).

WHAT CHANGES FROM cp_v0:
  v0: all_gather K,V  -> every rank holds S_total keys -> one plain softmax.
      Memory O(cp_size) in KV. Traffic: everyone pulls the whole sequence.
  v1: each rank holds ONE KV block at a time and rotates it around the ring.
      Memory O(1) in KV. Per step, every rank sends exactly one block and
      receives exactly one -> every link busy every step.

The thing that buys the O(1) is the online-softmax accumulator (O, m, l).
A block arrives, is merged into the accumulator, is forwarded, and is then
garbage. Nothing about the sequence is retained except the accumulator.

TEST TARGET: cp_v0_attention, NOT F.scaled_dot_product_attention.
That is what Stage 1 was for.

  run:  torchrun --nproc_per_node=4 cp_v1_ring.py
"""

import torch
import torch.distributed as dist
import torch.nn.functional as F
import math
from cp0 import cp_v0_attention


# ==========================================================================
# 1. Block attention: q_local against ONE kv block.
#    Must return the softmax STATISTICS, not just the output — the merge
#    cannot be done from O alone.
# ==========================================================================
def _block_attention(q, k, v):
    """
    q: [B, nh, S_local, hd]      k, v: [B, nh, S_blk, hd]

    Returns (o_blk, m_blk, l_blk):
      m_blk: [B, nh, S_local, 1]  rowwise max of the scores
      l_blk: [B, nh, S_local, 1]  rowwise sum of exp(scores - m_blk)
      o_blk: [B, nh, S_local, hd]

    CONVENTION DECISION (state it, then never mix):
      (a) o_blk NORMALIZED   -> o_blk = softmax(s) @ v, i.e. already / l_blk
      (b) o_blk UNNORMALIZED -> o_blk = exp(s - m_blk) @ v, divide once at the end
    Your locked merge formula is written in convention (a). FlashAttention uses
    (b) — one fewer divide per step. Pick one. Mixing them is the classic bug
    and it shows up as an error that grows with cp_size.

    BLANK 1 ------------------------------------------------------------
    Note you cannot use F.scaled_dot_product_attention here: it returns the
    output only and throws away m and l. Hand-rolled.
      - scores = q @ k^T * scale        (don't lose 1/sqrt(hd))
      - m_blk   scores.max(dim=-1, keepdim=True)
      - l_blk = = exp(scores - m_blk).sum(dim=-1, keepdim=True)
      - o_blk  per your chosen convention
    """

    # --------------------------------------------------------------------
    
    scale = 1/math.sqrt(q.shape[-1])
    scores = q@k.transpose(-2, -1)*scale

    m_blk = scores.max(dim=-1, keepdim = True).values
    l_blk = torch.exp(scores - m_blk).sum(dim=-1, keepdim=True)
    o_blk = torch.exp(scores - m_blk) @ v
    
    return o_blk, m_blk, l_blk


# ==========================================================================
# 2. Online-softmax merge. You have this locked; write it from memory.
# ==========================================================================
def _merge(o_run, m_run, l_run, o_blk, m_blk, l_blk):
    """
    BLANK 2 ------------------------------------------------------------
    m_new = max(m_run, m_blk); rescale both sides by exp(. - m_new); combine.
    Init state is (-inf, 0, 0) — check your formula does the right thing on
    the FIRST merge, when l_run = 0 and m_run = -inf. exp(-inf - m_new) = 0
    is fine; -inf - (-inf) = nan is NOT. If your first step produces nan,
    this is why, and there are two standard fixes.
    """
    
    ## when causal attention happens, we might run into case where m_blk is also -inf
    ## when this happens, m_new will also be -inf, and in the factor, we do
    ## m_run - m_new = -inf -(-inf) which is nana nd will poison eeverything. 
    # # when we reach causal attention, we need to fix this by enforcing a rule. 
    
    
    m_new = torch.maximum(m_run, m_blk) # as both of these are a tensor

    o_new = o_run * torch.exp(m_run - m_new) + o_blk * torch.exp(m_blk - m_new)
    l_new = l_run * torch.exp(m_run - m_new) + l_blk * torch.exp(m_blk - m_new)

    
    return o_new, m_new, l_new


# ==========================================================================
# 3. The ring.
# ==========================================================================
def cp_v1_ring_attention(q_local, k_local, v_local, cp_group, cp_size, causal=False):
    """
    Returns o_local: [B, nh, S_local, hd] — same contract as cp_v0_attention.
    """
    assert not causal, "causal needs global-position offsets -> Stage 4"

    rank = dist.get_rank(cp_group)
    send_to = (rank + 1) % cp_size
    recv_from = (rank - 1) % cp_size

    # BLANK 3 ------------------------------------------------------------
    # Buffers. You reasoned to 2 slots. Decide, and write down:
    #   - 2 slots for K and 2 for V (4 tensors), or one fused KV buffer (2)?
    #     Fusing halves the message count. What does it cost you?
    #   - end-of-step swap: pointer swap, not a copy. If you find yourself
    #     writing .copy_() you have reintroduced a per-step memcpy and thrown
    #     away half of what the ring bought you.
    # --------------------------------------------------------------------

 

    # BLANK 4 ------------------------------------------------------------
    # The loop. cp_size - 1 transfers, cp_size compute rounds (you start
    # already holding your own block).
    #
    # ORDERING IS THE WHOLE POINT. If you write
    #     dist.send(...); dist.recv(...); compute(...)
    # you get two failures at once:
    #   (i)  every rank blocks in send simultaneously, nobody has posted a
    #        recv -> hang. Two standard fixes; one is strictly better because
    #        it also fixes (ii). (Gloo may buffer small messages and hide this
    #        at S_total=8. It will not hide it at real sizes. Predict, don't
    #        discover.)
    #   (ii) comm and compute are serialized. The ring's entire justification
    #        over all_gather is that the transfer of block i+1 hides under the
    #        attention math on block i. Serialize them and you have built
    #        something with all_gather's latency and none of its simplicity.
    #
    # So the step body is: post async comm for the NEXT block -> compute and
    # merge the CURRENT block -> wait -> swap. Write it that way.
    #
    # One trap: the tensor you pass to isend must not be mutated until the
    # wait returns. With a 2-slot scheme this is satisfied automatically —
    # confirm to yourself why before relying on it.
    o_run, m_run, l_run = 0, torch.tensor(-torch.inf), 0
    # --------------------------------------------------------------------


    
    cur_k = k_local.clone()
    cur_v = v_local.clone()

    recv_k = torch.empty_like(k_local)
    recv_v = torch.empty_like(v_local)

    for step in range(cp_size):
        
        if step < cp_size - 1: 
            k_send_op = dist.P2POp(op=dist.isend, tensor=cur_k, peer=send_to, tag=11)
            v_send_op = dist.P2POp(op=dist.isend, tensor=cur_v, peer=send_to, tag=22)

            k_recv_op = dist.P2POp(op=dist.irecv, tensor=recv_k, peer=recv_from, tag=11)
            v_recv_op = dist.P2POp(op=dist.irecv, tensor=recv_v, peer=recv_from, tag=22)  

            reqs = dist.batch_isend_irecv([k_send_op, v_send_op, k_recv_op, v_recv_op])   
        
        o_blk, m_blk, l_blk = _block_attention(q_local, cur_k, cur_v)
        o_new, m_new, l_new = _merge(o_run, m_run, l_run, o_blk, m_blk, l_blk)
        o_run, m_run, l_run = o_new, m_new, l_new 
        
        if step < cp_size - 1: 
            for req in reqs:
                req.wait()

            cur_k, recv_k = recv_k, cur_k
            cur_v, recv_v = recv_v, cur_v

    # BLANK 5: if you chose the unnormalized convention, divide by l_run here.
    return o_run / l_run 


# ==========================================================================
# 4. Test. Oracle is cp_v0.
# ==========================================================================
def _run_test():
    dist.init_process_group("gloo")
    rank, world = dist.get_rank(), dist.get_world_size()
    cp_group, cp_size = dist.group.WORLD, world

    torch.manual_seed(0)
    B, nh, S_total, hd = 2, 4, 16, 16
    assert S_total % cp_size == 0
    S_local = S_total // cp_size
    lo, hi = rank * S_local, (rank + 1) * S_local

    q_full = torch.randn(B, nh, S_total, hd)
    k_full = torch.randn(B, nh, S_total, hd)
    v_full = torch.randn(B, nh, S_total, hd)

    # DECISION (question 4 from before): Stage 3 does not exist yet, so
    # backward through this is NOT correct — dist.isend/irecv are not
    # differentiable and autograd will happily produce plausible garbage.
    # Run under no_grad so "forward is correct" is a claim you can test in
    # isolation. Do not leave a live autograd path you might trust later.
    with torch.no_grad():
        q = q_full[:, :, lo:hi, :].clone()
        k = k_full[:, :, lo:hi, :].clone()
        v = v_full[:, :, lo:hi, :].clone()

        o_ref = cp_v0_attention(q, k, v, cp_group, cp_size, causal=False)
        o_ring = cp_v1_ring_attention(q, k, v, cp_group, cp_size, causal=False)

        max_err = (o_ring - o_ref).abs().max().item()
        ok = torch.allclose(o_ring, o_ref, atol=1e-5)

    print(f"[rank {rank}] ring_matches_v0={ok}  max_err={max_err:.2e}")
    dist.destroy_process_group()


if __name__ == "__main__":
    _run_test()