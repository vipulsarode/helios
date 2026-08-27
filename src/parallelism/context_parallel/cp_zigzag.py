"""
cp_v2_zigzag.py  —  Context Parallelism, zigzag causal ring forward (Stage 4).

WHAT CARRIES OVER UNCHANGED FROM STAGE 2/3 (do not re-derive):
  - the ring rotation (2-buffer, post->compute->wait->swap, cp_size steps)
  - the online-softmax merge (O, m, l)
  - the block attention math
  - the autograd Function wrapper + ring backward (Stage 3), once forward is green

WHAT IS NEW IN STAGE 4 (this is the whole stage):
  1. SHARDING. Split the sequence into 2*cp_size chunks, not cp_size. Rank r
     owns chunk r AND chunk (2*cp_size - 1 - r): one head chunk, one tail chunk.
     This is what balances causal load — the lightest query chunk (earliest)
     is paired with the heaviest (latest), so every rank does equal work.
  2. TWO ACCUMULATORS PER RANK. Each of a rank's two query chunks produces its
     own attention output with its own (O, m, l). They are DIFFERENT rows of the
     output and never merge with each other.
  3. DATA-DEPENDENT MASKING. At each ring step, for each (local query chunk,
     visiting KV chunk) pair, classify by GLOBAL chunk index:
         k_idx <  q_idx  -> C : full attention, is_causal=False
         k_idx == q_idx  -> T : causal attention, is_causal=True  (diagonal only)
         k_idx >  q_idx  -> S : skip the COMPUTE (but NOT the comm — the block
                                must still rotate onward for downstream ranks)
  4. INDEX TRACKING. At step s on rank r, the visiting KV originated on source
     rank j = (r - s) mod cp_size, whose two global chunk indices are
     j and (2*cp_size - 1 - j). Your two local query chunk indices are
     r and (2*cp_size - 1 - r).

  run:  torchrun --nproc_per_node=4 cp_v2_zigzag.py
"""

import torch
import torch.distributed as dist
import torch.nn.functional as F
import math

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
# 0. Global-chunk-index helpers.  Pure integer arithmetic — the heart of zigzag.
# ==========================================================================
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

# ==========================================================================
# 1. SHARDING.  Turn a full [B, nh, S_total, hd] tensor into THIS rank's two
#    chunks, and provide the inverse (gather chunks back to full sequence order)
#    for the test's oracle comparison.
# ==========================================================================
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

# ==========================================================================
# 2. Block attention returning softmax stats, with a mask mode.
#    (Same as Stage 3's _block_attention but takes causal flag; for "C" pass
#     causal=False, for "T" pass causal=True. "S" never calls this.)
#    You may reuse your Stage 3 _block_attention verbatim if it already takes a
#    causal argument; otherwise add one.
# ==========================================================================
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
# 3. The zigzag forward.
#
#    Start from your Stage 2 loop. It already does the rotation you need:
#    post -> compute -> wait -> swap, cp_size steps. You are changing ONE thing
#    inside it — the single block-attention call becomes a decision about which
#    (query chunk, key chunk) pairings to compute this step.
#
#    Directions (not code) — work these out from what you already derived:
#
#    - Accumulators: how many (O,m,l) states does this rank carry now, and why
#      can't the two of them ever merge into each other? Init each the way
#      Stage 2 inits its one.
#
#    - Buffers: the rotating KV is now two chunks per rank. Decide fused-vs-
#      separate messages (you reasoned the tradeoff already) and be consistent.
#
#    - Per step: identify the source rank of the visiting KV (_source_rank_at_step),
#      expand it to its two global chunk indices (_owned_chunk_indices). Your two
#      local query chunks also have known global indices. That is the full set of
#      (query, key) pairings for the step — classify each with _classify and act:
#      S skips the COMPUTE, C/T call _block_attention with the right causal flag
#      and merge into that query chunk's accumulator.
#
#    - Invariant to preserve: the KV send/recv fires once per step regardless of
#      how many pairings were S. If comm ends up inside the pairing loop, that's
#      the bug.
#
#    - Return: the two normalized outputs (O/l), one per owned query chunk.
#
#    BLANK F: implement.
# ==========================================================================
def _zigzag_forward(q_head, q_tail, k_head, k_tail, v_head, v_tail,
                    cp_group, cp_size, send_to, recv_from, rank):
    """
    Each of the six inputs is [B, nh, chunk_len, hd] (this rank's two chunks).
    Returns (o_head, o_tail): attention outputs for this rank's two query chunks.
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

    return o_head/l_head , o_tail/l_tail




# ==========================================================================
# 4. Self-checking gloo test.
#    Oracle: single-process CAUSAL full attention on the same seeded inputs.
#    Then shard zigzag, run _zigzag_forward, scatter the two output chunks back
#    to their GLOBAL positions, and compare against the oracle sliced at those
#    positions. This scatter-back is why _owned_chunk_indices must be correct:
#    a wrong index passes the per-rank shapes but compares against the wrong
#    slice of truth.
#
#    run:  torchrun --nproc_per_node=4 cp_v2_zigzag.py
# ==========================================================================
def _run_test():
    dist.init_process_group("gloo")
    rank, world = dist.get_rank(), dist.get_world_size()
    cp_group, cp_size = dist.group.WORLD, world
    send_to = (rank + 1) % cp_size
    recv_from = (rank - 1) % cp_size
    if rank == 0:
        print("owned:", [_owned_chunk_indices(r, cp_size) for r in range(cp_size)])
        print("classify (0,0),(0,7),(7,0),(7,7):",
            _classify(0,0), _classify(0,7), _classify(7,0), _classify(7,7))

    torch.manual_seed(0)
    B, nh, S_total, hd = 2, 4, 32, 16
    assert S_total % (2 * cp_size) == 0
    chunk_len = S_total // (2 * cp_size)

    q_full = torch.randn(B, nh, S_total, hd)
    k_full = torch.randn(B, nh, S_total, hd)
    v_full = torch.randn(B, nh, S_total, hd)

    # ---- oracle: single-process CAUSAL attention ----
    with torch.no_grad():
        ref = F.scaled_dot_product_attention(q_full, k_full, v_full, is_causal=True)

    # ---- shard zigzag ----
    q_head, q_tail = _shard_zigzag(q_full, rank, cp_size)
    k_head, k_tail = _shard_zigzag(k_full, rank, cp_size)
    v_head, v_tail = _shard_zigzag(v_full, rank, cp_size)

    with torch.no_grad():
        o_head, o_tail = _zigzag_forward(q_head, q_tail, k_head, k_tail, v_head, v_tail,
                                         cp_group, cp_size, send_to, recv_from, rank)

    # ---- compare each output chunk against the oracle at its GLOBAL slice ----
    head_idx, tail_idx = _owned_chunk_indices(rank, cp_size)
    def _slice(idx):
        return ref[:, :, idx * chunk_len:(idx + 1) * chunk_len, :]
    head_ok = torch.allclose(o_head, _slice(head_idx), atol=1e-5)
    tail_ok = torch.allclose(o_tail, _slice(tail_idx), atol=1e-5)

    print(f"[rank {rank}] head_ok={head_ok} tail_ok={tail_ok}  "
          f"(chunks {head_idx},{tail_idx})")
    dist.destroy_process_group()


if __name__ == "__main__":
    _run_test()