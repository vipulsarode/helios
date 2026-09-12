"""
cp_v2_zigzag_backward.py  —  Zigzag causal ring backward (Stage 4, backward half).

WHAT YOU ALREADY HAVE, VALIDATED (do not re-derive):
  - Stage 3 ring backward: dK/dV partials rotate with their KV block and
    accumulate one rank's contribution per hop; dQ accumulates locally; a final
    delivery hop lands each partial on its owner. cp_size transfers.
  - Stage 4 forward: 2 owned chunks per rank (head r, tail 2n-1-r), the C/T/S
    classification by global chunk index, two independent accumulators.
  - The FlashAttention per-block backward (_block_backward) with the causal
    flag for the T case.

WHAT IS NEW IN THIS STAGE (small — it's a composition):
  1. Two query chunks -> TWO of everything on the local side: dq_head and
     dq_tail each accumulate locally (neither rotates; queries never move).
  2. Two KV chunks per rank -> the rotating partials are now per-chunk:
     a dK/dV partial for the visiting head chunk AND for the visiting tail
     chunk, each riding home to its own owner.
  3. The C/T/S gate on the gradient:
       S : this (query chunk, key chunk) pair contributed NOTHING in forward,
           so it contributes NOTHING to any gradient. Skip the _block_backward
           call entirely — but the partials still rotate onward (a skipped pair
           on THIS rank may be non-skipped on another).
       T : recompute with the causal mask (same mask as forward's T).
       C : recompute unmasked.
  4. Correctness subtlety: because forward used the saved logsumexp L per query
     chunk, backward must use the SAME per-chunk L (L_head for q_head, L_tail
     for q_tail). Mixing them silently corrupts the recomputed P.

  run:  torchrun --nproc_per_node=4 cp_v2_zigzag_backward.py
"""

import torch
import torch.distributed as dist
import torch.nn.functional as F
import math

# reuse from your validated files:
#   _owned_chunk_indices, _source_rank_at_step, _classify  (Stage 4 fwd)
#   _block_backward                                          (Stage 3)
#   _shard_zigzag, _zigzag_forward (instrumented to also return L_head, L_tail)

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

def _owned_chunk_indices(rank, cp_size):
    """
    The two GLOBAL chunk indices rank `rank` owns.
    BLANK A: return (head_idx, tail_idx) for this rank.
             head = rank ; tail = 2*cp_size - 1 - rank.
    """
    return rank, 2*cp_size - 1 - rank


def _source_rank_at_step(rank, step, cp_size):
    """
    Which rank originally owned the KV block currently visiting `rank` at `step`.
    BLANK B: standard ring receives from rank-1 each step, so the block in your
             buffer at step s started on rank (rank - step) mod cp_size.
    """
    return (rank - step) % cp_size


def _classify(q_idx, k_idx):
    """
    The C/T/S decision for one (query chunk, key chunk) pair, by global index.
    BLANK C: return one of the strings "C", "T", "S" per the rule above.
    """
    if q_idx > k_idx:
        return "C"
    elif q_idx < k_idx:
        return "S"
    else:
        return "T"
    
def _shard_zigzag(x_full, rank, cp_size):
    """
    x_full: [B, nh, S_total, hd], S_total divisible by 2*cp_size.
    Returns this rank's two chunks stacked/kept separate — YOUR CALL on layout,
    but be consistent everywhere downstream.

    BLANK D:
      - chunk_len = S_total // (2*cp_size)
      - head_idx, tail_idx = _owned_chunk_indices(rank, cp_size)
      - slice x_full along seq (dim=2) at those two chunk positions
      - return (x_head, x_tail), each [B, nh, chunk_len, hd]
    """
    chunk_len = x_full.shape[-2]//(2*cp_size)
    head_idx, tail_idx = _owned_chunk_indices(rank, cp_size)
    chunks = torch.split(x_full, chunk_len, dim=-2)
    return chunks[head_idx], chunks[tail_idx]

def _block_backward(q_local, o_local, dO_local, L_local, k_blk, v_blk, scale, causal):
    """
    RETURNS: dq_contrib [B, nh, S_local, hd]   -> accumulate LOCALLY
             dk_blk     [B, nh, S_blk,   hd]   -> ride the rotating dK partial
             dv_blk     [B, nh, S_blk,   hd]   -> ride the rotating dV partial
    """

    S = q_local@k_blk.transpose(-2,-1) * scale
    if causal:  
       mask = torch.tril(torch.ones_like(S ), diagonal=0)
       S  = S .masked_fill(mask == 0, float('-inf')) 
    P = torch.exp(S - L_local)
    dV_blk = P.transpose(-2,-1) @ dO_local
    dP = dO_local@v_blk.transpose(-2, -1)
    D = (dO_local * o_local).sum(dim=-1, keepdim=True)
    dS = P*(dP - D)
    dQ_contrib = (dS @ k_blk) * scale
    dK_blk = (dS.transpose(-2,-1) @ q_local) * scale

    # assert torch.allclose(dS.sum(dim=-1), torch.zeros_like(dS.sum(dim=-1)), atol=1e-5), "dS rows don't sum to zero"

    return dQ_contrib, dK_blk, dV_blk

def _block_attention(q, k, v, causal):
    """
    q,k,v: [B, nh, chunk_len, hd].  Returns (o_blk, m_blk, l_blk) with the
    convention you locked in Stage 2 (unnormalized O, divide once at the end).
    BLANK E (only if your Stage 3 version lacks a causal flag): add is_causal
    handling — scores get a lower-triangular mask (set upper to -inf) BEFORE
    the rowwise max, so the masked entries never enter m or l.
    """

    
    scale = 1/math.sqrt(q.shape[-1])
    scores = q@k.transpose(-2,-1)*scale
    
    if causal:  
       mask = torch.tril(torch.ones_like(scores), diagonal=0)
       scores = scores.masked_fill(mask == 0, float('-inf')) 
        
    m_blk = torch.max(scores, dim = -1, keepdim=True).values
    l_blk = torch.sum(torch.exp(scores - m_blk), dim=-1, keepdim=True)
    o_blk =  torch.exp(scores - m_blk)@v         

    return o_blk, m_blk, l_blk


# ==========================================================================
# 0. FORWARD, instrumented to save per-chunk logsumexp.
#
#    Your Stage 4 forward returns (o_head, o_tail). Backward also needs
#    (L_head, L_tail) — the per-query-chunk logsumexp = m + log(l), taken at the
#    END of the ring (normalized over every key that chunk actually attended).
#    Add these two returns; change nothing else. Same one-line fold as Stage 3.
# ==========================================================================
def _zigzag_forward_with_lse(q_head, q_tail, k_head, k_tail, v_head, v_tail,
                             cp_group, cp_size, send_to, recv_from, rank):
    """
    RETURNS: o_head, o_tail, L_head, L_tail
      o_*: [B, nh, chunk_len, hd]
      L_*: [B, nh, chunk_len, 1]   (m + log(l) for that query chunk)
    Lift your validated Stage 4 forward; add the two L returns.
    """
    
    o_head, m_head, l_head = 0, torch.tensor(-torch.inf), 0
    o_tail, m_tail, l_tail = 0, torch.tensor(-torch.inf), 0

    cur_k_head = k_head.clone()
    cur_k_tail = k_tail.clone()

    cur_v_head = v_head.clone()
    cur_v_tail = v_tail.clone()

    recv_k_head = torch.empty_like(k_head)
    recv_k_tail = torch.empty_like(k_tail)

    recv_v_head = torch.empty_like(v_head)
    recv_v_tail = torch.empty_like(v_tail)


    for step in range(cp_size):

        if step < cp_size-1:        
            k_head_send = dist.P2POp(dist.isend, cur_k_head, peer=send_to, group = cp_group, tag = 11)
            k_tail_send = dist.P2POp(dist.isend, cur_k_tail, peer=send_to, group = cp_group, tag = 22)

            v_head_send = dist.P2POp(dist.isend, cur_v_head, peer=send_to, group = cp_group, tag = 33)
            v_tail_send = dist.P2POp(dist.isend, cur_v_tail, peer=send_to, group = cp_group, tag = 44)

            k_head_recv = dist.P2POp(dist.irecv, recv_k_head, peer=recv_from, group = cp_group, tag = 11)
            k_tail_recv = dist.P2POp(dist.irecv, recv_k_tail, peer=recv_from, group = cp_group, tag = 22)

            v_head_recv = dist.P2POp(dist.irecv, recv_v_head, peer=recv_from, group = cp_group, tag = 33)
            v_tail_recv = dist.P2POp(dist.irecv, recv_v_tail, peer=recv_from, group = cp_group, tag = 44)

            reqs = dist.batch_isend_irecv([k_head_send, k_head_recv, v_head_send, v_head_recv, k_tail_send, k_tail_recv, v_tail_send, v_tail_recv])


        q_head_idx, q_tail_idx = _owned_chunk_indices(rank, cp_size)          # your queries
        source = _source_rank_at_step(rank, step, cp_size) 
        k_head_idx, k_tail_idx = _owned_chunk_indices(source, cp_size)   

        for k, v, k_idx in [(cur_k_head, cur_v_head, k_head_idx),(cur_k_tail, cur_v_tail, k_tail_idx)]:        
            status = _classify(q_head_idx, k_idx)

            if status == "S": continue
            o_head_blk, m_head_blk, l_head_blk = _block_attention(q_head, k, v, causal=(status=="T"))
            o_head_new, m_head_new, l_head_new = _merge(o_head, m_head, l_head, o_head_blk, m_head_blk, l_head_blk)
            o_head, m_head, l_head = o_head_new, m_head_new, l_head_new 

        for k, v, k_idx in [(cur_k_head, cur_v_head, k_head_idx),(cur_k_tail, cur_v_tail, k_tail_idx)]: 
            status = _classify(q_tail_idx, k_idx)

            if status == "S": continue            
            o_tail_blk, m_tail_blk, l_tail_blk = _block_attention(q_tail, k, v, causal=(status=="T"))
            o_tail_new, m_tail_new, l_tail_new = _merge(o_tail, m_tail, l_tail, o_tail_blk, m_tail_blk, l_tail_blk)
            o_tail, m_tail, l_tail = o_tail_new, m_tail_new, l_tail_new 

        if step < cp_size-1:   
            for req in reqs:
                req.wait()

            cur_k_head, recv_k_head = recv_k_head, cur_k_head
            cur_v_head, recv_v_head = recv_v_head, cur_v_head
            cur_k_tail, recv_k_tail = recv_k_tail, cur_k_tail
            cur_v_tail, recv_v_tail = recv_v_tail, cur_v_tail

    return o_head/l_head, m_head + torch.log(l_head), o_tail/l_tail, m_tail + torch.log(l_tail)


# ==========================================================================
# 1. The zigzag backward ring.
#
#    STATE (init before the step loop, persists across steps):
#      dq_head, dq_tail        local grad accumulators (zeros), NEVER rotate
#      cur_dk_head, cur_dv_head, cur_dk_tail, cur_dv_tail
#                              rotating partials (zeros), ride with their KV
#      cur_k_head, cur_v_head, cur_k_tail, cur_v_tail
#                              the rotating KV blocks (seed from own chunks)
#      recv_* buffers for every rotating tensor
#
#    Per step (mirror Stage 3's backward loop, with the Stage 4 dispatch inside):
#      - resolve the visiting KV's source rank and its two global chunk indices
#        (_source_rank_at_step -> _owned_chunk_indices), exactly as forward did.
#      - your two query chunks have fixed global indices (r, 2n-1-r).
#      - for each (query chunk, visiting KV chunk) pair  [4 pairs]:
#            case = _classify(q_idx, k_idx)
#            if case == "S":  contribute nothing, move on
#            else: dq_c, dk_c, dv_c = _block_backward(q_chunk, o_chunk, dO_chunk,
#                                       L_chunk, k_chunk, v_chunk, scale,
#                                       causal=(case=="T"))
#                  dq_<that query chunk> += dq_c
#                  add dk_c, dv_c into the partial for THAT VISITING KV CHUNK
#      - rotate: KV blocks AND their dK/dV partials move one hop (lockstep swap,
#        four-tensor family per chunk as in Stage 3). cp_size transfers incl. the
#        final delivery hop that carries each partial home with no compute after.
#
#    GUARD cp_size == 1: no rotation, no comm (the CP-off path). Same guard bug
#    that segfaulted you twice — put it in from the start this time.
#
#    RETURNS: dq_head, dq_tail, dk_head, dk_tail, dv_head, dv_tail
#      each [B, nh, chunk_len, hd], each the COMPLETE grad for that owned chunk.
#      (dk_head/dv_head are the partials that came home for YOUR head KV chunk;
#       dk_tail/dv_tail for your tail KV chunk.)
#
#    Two traps carried from earlier stages, stated so you predict not discover:
#      - _block_backward needs a causal flag now. If yours lacks one, the T case
#        recomputes P without the mask and the diagonal blocks' grads are wrong
#        (C blocks still pass -> partial-green failure pointing straight at T).
#      - the S skip is on the COMPUTE only. The partial for a KV chunk you didn't
#        use this step must still rotate onward unchanged, or a downstream rank's
#        contribution to that chunk never reaches the owner.
# ==========================================================================
def _zigzag_backward(q_head, q_tail, o_head, o_tail, dO_head, dO_tail,
                     L_head, L_tail, k_head, k_tail, v_head, v_tail,
                     cp_group, cp_size, send_to, recv_from, rank, scale):
    """
    RETURNS: dq_head, dq_tail, dk_head, dk_tail, dv_head, dv_tail
    """
    # if cp_size == 1:
    #     dq, dk, dv = _block_backward(q_local, o_local, dO_local, L_local, k_local, v_local, scale)
    #     return dq, dk, dv
        
    cur_k_head = k_head.clone()
    cur_v_head = v_head.clone()

    cur_k_tail = k_tail.clone()
    cur_v_tail = v_tail.clone()

    recv_k_head = torch.empty_like(k_head)
    recv_v_head = torch.empty_like(v_head)

    recv_k_tail = torch.empty_like(k_tail)
    recv_v_tail = torch.empty_like(v_tail)

    recv_dK_head = torch.empty_like(k_head)
    recv_dV_head = torch.empty_like(v_head)

    recv_dK_tail = torch.empty_like(k_tail)
    recv_dV_tail = torch.empty_like(v_tail)

    cur_dQ_head = torch.zeros_like(q_head)
    cur_dK_head = torch.zeros_like(k_head)
    cur_dV_head = torch.zeros_like(v_head)
    
    cur_dQ_tail = torch.zeros_like(q_tail)
    cur_dK_tail = torch.zeros_like(k_tail)
    cur_dV_tail = torch.zeros_like(v_tail)

    for step in range(cp_size):

        k_send_head = dist.P2POp(dist.isend, cur_k_head, peer=send_to, group=cp_group, tag = 11)
        v_send_head = dist.P2POp(dist.isend, cur_v_head, peer=send_to, group=cp_group, tag = 22)
        k_recv_head = dist.P2POp(dist.irecv, recv_k_head, peer=recv_from, group=cp_group, tag = 11)
        v_recv_head = dist.P2POp(dist.irecv, recv_v_head, peer=recv_from, group=cp_group, tag = 22)

        k_send_tail = dist.P2POp(dist.isend, cur_k_tail, peer=send_to, group=cp_group, tag = 31)
        v_send_tail = dist.P2POp(dist.isend, cur_v_tail, peer=send_to, group=cp_group, tag = 32)
        k_recv_tail = dist.P2POp(dist.irecv, recv_k_tail, peer=recv_from, group=cp_group, tag = 31)
        v_recv_tail = dist.P2POp(dist.irecv, recv_v_tail, peer=recv_from, group=cp_group, tag = 32)
        
        
        dK_recv_head = dist.P2POp(dist.irecv, recv_dK_head, peer=recv_from, group=cp_group, tag = 33)
        dV_recv_head = dist.P2POp(dist.irecv, recv_dV_head, peer=recv_from, group=cp_group, tag = 44)

        dK_recv_tail = dist.P2POp(dist.irecv, recv_dK_tail, peer=recv_from, group=cp_group, tag = 55)
        dV_recv_tail = dist.P2POp(dist.irecv, recv_dV_tail, peer=recv_from, group=cp_group, tag = 66)

        
        q_head_idx, q_tail_idx = _owned_chunk_indices(rank, cp_size)          # your queries
        source = _source_rank_at_step(rank, step, cp_size) 
        k_head_idx, k_tail_idx = _owned_chunk_indices(source, cp_size)   

        for k, v, k_idx, kv_type in [(cur_k_head, cur_v_head, k_head_idx, "head"),(cur_k_tail, cur_v_tail, k_tail_idx, "tail")]:        
            status = _classify(q_head_idx, k_idx)
            if status == "S": continue
            dQ_head, dK_head, dV_head = _block_backward(q_head, o_head, dO_head, L_head, k, v, scale, causal = (status == "T"))

            cur_dQ_head = cur_dQ_head + dQ_head.clone()
            
            if kv_type == "head":
                cur_dK_head = cur_dK_head + dK_head.clone() 
                cur_dV_head = cur_dV_head + dV_head.clone()
            else:
                cur_dK_tail = cur_dK_tail + dK_head.clone() 
                cur_dV_tail = cur_dV_tail + dV_head.clone()
        
        for k, v, k_idx, kv_type in [(cur_k_head, cur_v_head, k_head_idx, "head"),(cur_k_tail, cur_v_tail, k_tail_idx, "tail")]:        
            status = _classify(q_tail_idx, k_idx)
            if status == "S": continue        
            dQ_tail, dK_tail, dV_tail = _block_backward(q_tail, o_tail, dO_tail, L_tail, k, v, scale, causal = (status == "T"))

            cur_dQ_tail = cur_dQ_tail + dQ_tail.clone()
            
            if kv_type == "head":
                cur_dK_head = cur_dK_head + dK_tail.clone() 
                cur_dV_head = cur_dV_head + dV_tail.clone()
            else:
                cur_dK_tail = cur_dK_tail + dK_tail.clone() 
                cur_dV_tail = cur_dV_tail + dV_tail.clone()

        dK_send_head = dist.P2POp(dist.isend, cur_dK_head, peer=send_to, group=cp_group, tag = 33)
        dV_send_head = dist.P2POp(dist.isend, cur_dV_head, peer=send_to, group=cp_group, tag = 44)

        dK_send_tail = dist.P2POp(dist.isend, cur_dK_tail, peer=send_to, group=cp_group, tag = 55)
        dV_send_tail = dist.P2POp(dist.isend, cur_dV_tail, peer=send_to, group=cp_group, tag = 66)        

        reqs = dist.batch_isend_irecv([k_send_head, v_send_head, k_recv_head, v_recv_head, k_send_tail, v_send_tail, k_recv_tail, v_recv_tail,
                                       dK_recv_head, dV_recv_head, dK_recv_tail, dV_recv_tail, dK_send_head, dV_send_head,
                                       dK_send_tail, dV_send_tail])

        for req in reqs:
            req.wait()

        cur_k_head, recv_k_head = recv_k_head, cur_k_head
        cur_v_head, recv_v_head = recv_v_head, cur_v_head
        cur_k_tail, recv_k_tail = recv_k_tail, cur_k_tail
        cur_v_tail, recv_v_tail = recv_v_tail, cur_v_tail

        cur_dK_head, recv_dK_head = recv_dK_head, cur_dK_head
        cur_dV_head, recv_dV_head = recv_dV_head, cur_dV_head
        cur_dK_tail, recv_dK_tail = recv_dK_tail, cur_dK_tail
        cur_dV_tail, recv_dV_tail= recv_dV_tail, cur_dV_tail


    return  cur_dQ_head, cur_dQ_tail, cur_dK_head, cur_dK_tail, cur_dV_head, cur_dV_tail



# ==========================================================================
# 2. Autograd Function.
#    forward: run _zigzag_forward_with_lse; save for backward q/k/v (both
#      chunks), o_head/o_tail, L_head/L_tail; stash ring metadata + scale on ctx;
#      return (o_head, o_tail).  NOTE: forward now returns TWO tensors, so
#      backward receives TWO upstream grads (grad_o_head, grad_o_tail).
#    backward: unpack saved tensors, call _zigzag_backward, return one grad per
#      forward input positionally. Forward inputs are the six chunk tensors then
#      the non-tensors -> six real grads then None per non-tensor. Count them.
# ==========================================================================
class ZigzagAttention(torch.autograd.Function):

    @staticmethod
    def forward(ctx, q_head, q_tail, k_head, k_tail, v_head, v_tail,
                cp_group, cp_size, send_to, recv_from, rank):
        o_head, L_head, o_tail, L_tail =  _zigzag_forward_with_lse(q_head, q_tail, k_head, k_tail, v_head, v_tail,
                             cp_group, cp_size, send_to, recv_from, rank)
        ctx.save_for_backward(q_head, q_tail, k_head, k_tail, v_head, v_tail, o_head, L_head, o_tail, L_tail)
        ctx.cp_group = cp_group
        ctx.cp_size = cp_size
        ctx.send_to = send_to
        ctx.recv_from = recv_from
        ctx.rank = rank
        ctx.scale = 1.0 / (q_head.shape[-1] ** 0.5)
        return o_head, o_tail


    @staticmethod
    def backward(ctx, grad_o_head, grad_o_tail):
        
        q_head, q_tail, k_head, k_tail, v_head, v_tail, o_head, L_head, o_tail, L_tail = ctx.saved_tensors
        cur_dQ_head, cur_dQ_tail, cur_dK_head, cur_dK_tail, cur_dV_head, cur_dV_tail = _zigzag_backward(q_head, q_tail, o_head, o_tail, 
                                                                                                        grad_o_head, grad_o_tail, L_head, L_tail, 
                                                                                                        k_head, k_tail, v_head, v_tail,
                                                                                                        ctx.cp_group, ctx.cp_size, ctx.send_to, ctx.recv_from, 
                                                                                                        ctx.rank, ctx.scale)
        return cur_dQ_head, cur_dQ_tail, cur_dK_head, cur_dK_tail, cur_dV_head, cur_dV_tail, None, None, None, None, None, None


def zigzag_attention(q_head, q_tail, k_head, k_tail, v_head, v_tail,
                     cp_group, cp_size, send_to, recv_from, rank):
    return ZigzagAttention.apply(q_head, q_tail, k_head, k_tail, v_head, v_tail,
                                 cp_group, cp_size, send_to, recv_from, rank)


# ==========================================================================
# 3. Self-checking gloo test.
#    Oracle: single-process CAUSAL full attention; run its backward; keep
#    dq_full/dk_full/dv_full. Shard zigzag, run zigzag_attention fwd+bwd, and for
#    each owned chunk compare its grad against the oracle sliced at that chunk's
#    GLOBAL position. Six checks per rank (dq/dk/dv x head/tail).
#
#    Diagnostic reading, same discipline as before:
#      - all six red, uniform   -> classification / indexing (check _classify
#                                  and the source-rank mapping first)
#      - dq green, dk/dv red    -> partial routing (Stage-3-style delivery bug)
#      - only T-touching chunks red -> causal flag missing in _block_backward
#
#    Configs: cp_size in {1, 2, 4}. 1 is the no-comm guard path; 4 is honest.
#
#    run:  torchrun --nproc_per_node=4 cp_v2_zigzag_backward.py
# ==========================================================================
def _run_test():
    dist.init_process_group("gloo")
    rank, world = dist.get_rank(), dist.get_world_size()
    cp_group, cp_size = dist.group.WORLD, world
    send_to = (rank + 1) % cp_size
    recv_from = (rank - 1) % cp_size

    torch.manual_seed(0)
    B, nh, S_total, hd = 2, 4, 32, 16
    assert S_total % (2 * cp_size) == 0
    chunk_len = S_total // (2 * cp_size)
    scale = 1.0 / (hd ** 0.5)

    q_full = torch.randn(B, nh, S_total, hd, requires_grad=True)
    k_full = torch.randn(B, nh, S_total, hd, requires_grad=True)
    v_full = torch.randn(B, nh, S_total, hd, requires_grad=True)

    # ---- oracle: single-process CAUSAL attention ----
    ref = F.scaled_dot_product_attention(q_full, k_full, v_full, is_causal=True)
    ref.sum().backward()
    ref_dq, ref_dk, ref_dv = q_full.grad.clone(), k_full.grad.clone(), v_full.grad.clone()

    # ---- shard zigzag (detached leaves per chunk so grads land on them) ----
    def _shard_leaf(x_full):
        h, t = _shard_zigzag(x_full.detach(), rank, cp_size)
        return h.clone().requires_grad_(True), t.clone().requires_grad_(True)

    q_h, q_t = _shard_leaf(q_full)
    k_h, k_t = _shard_leaf(k_full)
    v_h, v_t = _shard_leaf(v_full)

    o_h, o_t = zigzag_attention(q_h, q_t, k_h, k_t, v_h, v_t,
                                cp_group, cp_size, send_to, recv_from, rank)
    (o_h.sum() + o_t.sum()).backward()

    head_idx, tail_idx = _owned_chunk_indices(rank, cp_size)
    def _sl(t, idx):
        return t[:, :, idx * chunk_len:(idx + 1) * chunk_len, :]

    checks = {
        "dq_h": torch.allclose(q_h.grad, _sl(ref_dq, head_idx), atol=1e-5),
        "dq_t": torch.allclose(q_t.grad, _sl(ref_dq, tail_idx), atol=1e-5),
        "dk_h": torch.allclose(k_h.grad, _sl(ref_dk, head_idx), atol=1e-5),
        "dk_t": torch.allclose(k_t.grad, _sl(ref_dk, tail_idx), atol=1e-5),
        "dv_h": torch.allclose(v_h.grad, _sl(ref_dv, head_idx), atol=1e-5),
        "dv_t": torch.allclose(v_t.grad, _sl(ref_dv, tail_idx), atol=1e-5),
    }
    summary = " ".join(f"{k}={'T' if ok else 'F'}" for k, ok in checks.items())
    print(f"[rank {rank}] {summary}  (chunks {head_idx},{tail_idx})")
    dist.destroy_process_group()


if __name__ == "__main__":
    _run_test()