"""
process_group_manager.py

Central registry for all parallelism axes in Helios: DP, PP, CP, TP.

Mental model (locked during design):
  - world_size GPUs form one 4D grid of shape (dp, pp, cp, tp).
  - A global rank is a flattened (mixed-radix) index into that grid.
  - Ordering, outermost -> innermost:  dp, pp, cp, tp
      tp innermost  -> group members are CONSECUTIVE global ranks (intra-node, NVLink)
      dp outermost  -> group members are FARTHEST apart (inter-node, overlappable)
  - Membership rule (the whole logical core):
      two ranks share axis-X's group  iff  they agree on every axis EXCEPT X.

Deadlock rule (do NOT violate):
  dist.new_group is COLLECTIVE. Every rank in the world must call it for EVERY
  group, in the SAME order, even groups it is not a member of. Keep only the
  handle for the group you belong to; discard the rest.
"""

import torch
import torch.distributed as dist


class ProcessGroupManager:

    def __init__(self, dp: int, pp: int, cp: int, tp: int):
        # --- 0. globals -----------------------------------------------------
        # TODO: read world_size and global_rank from the (already-initialized)
        #       dist backend. Assert dp * pp * cp * tp == world_size, else the
        #       grid doesn't tile the world -> fail loudly with a clear message.
        self.dp_size = dp
        self.pp_size = pp
        self.cp_size = cp
        self.tp_size = tp
        self.world_size = dist.get_world_size()
        self.global_rank = dist.get_rank()

        assert self.world_size == dp * pp * cp * tp, f"World size ({self.world_size}) != TP ({tp}) * CP ({cp}) * PP ({pp}) * DP ({dp})"

        # --- 1. build the grid ---------------------------------------------
        # TODO: build the 4D mesh tensor.
        #   Q: what exactly is reshape(dp, pp, cp, tp) doing to arange(world)?
        #   Q: which axis must be LAST for `reshape(-1, k)` to slice it for free?
        # self.mesh = ...

        # --- 2. build every axis's groups ----------------------------------
        # For each axis, this must populate:
        #     self.{axis}_group        (the ProcessGroup handle for MY group, or None if size==1)
        #     self.{axis}_local_rank   (my index within that group; 0 if size==1)
        #     self.{axis}_ranks        (ORDERED global-rank list of my group; needed for neighbors)
        # Use the same helper for all four so the membership rule lives in ONE place.
        #
        # TODO: call the helper once per axis. Mind the ORDER — it must be the
        #       same order on every rank (see deadlock rule).
        # self._build_axis_groups(axis_name="tp", axis_index=?, axis_size=tp)
        # self._build_axis_groups(axis_name="cp", axis_index=?, axis_size=cp)
        # self._build_axis_groups(axis_name="pp", axis_index=?, axis_size=pp)
        # self._build_axis_groups(axis_name="dp", axis_index=?, axis_size=dp)

        # --- 3. precompute neighbors ---------------------------------------
        # TODO: derive CP ring neighbors and PP line neighbors (see stubs below).
        # self._compute_cp_neighbors()
        # self._compute_pp_neighbors()

    # ----------------------------------------------------------------------
    # Group construction
    # ----------------------------------------------------------------------
    def _build_axis_groups(self, axis_name: str, axis_index: int, axis_size: int):
        """
        Create ALL groups along one axis, keep only the handle for MY group.

        Must satisfy:
          - enumerate every group along `axis_index` from self.mesh
              Q: for the LAST axis you can reshape(-1, k). For a NON-last axis,
                 what do you have to do to the mesh first before reshape(-1, k)?
          - EVERY rank calls dist.new_group for EVERY group, identical order
              (deadlock rule) — even the groups it isn't in.
          - keep group handle + my local_rank + ordered rank-list only for the
            group that contains self.global_rank.
          - EDGE CASE axis_size == 1: no real comm partner exists. Skip real
            group creation; set group=None, local_rank=0, ranks=[global_rank].
            Downstream code guards collectives with `if {axis}_size > 1`.

        TODO: implement. Set attributes via setattr(self, f"{axis_name}_group", ...) etc.
        """
        raise NotImplementedError

    # ----------------------------------------------------------------------
    # Neighbor computation
    # ----------------------------------------------------------------------
    def _compute_cp_neighbors(self):
        """
        CP is a RING (the sequence is a loop for KV passing).

        From self.cp_ranks (ordered) and self.cp_local_rank, compute:
            self.cp_send_rank   (who I send my KV block to)
            self.cp_recv_rank   (who I receive the next KV block from)

        Q: does the ring WRAP? (last local rank -> first local rank?)
        Q: which modular-arithmetic expression gives send vs recv?
        Q: cp_size == 1 -> what are send/recv? (there's no one to talk to)

        TODO: implement.
        """
        raise NotImplementedError

    def _compute_pp_neighbors(self):
        """
        PP is a LINE (a pipeline has a first and a last; it does NOT wrap).

        From self.pp_ranks (ordered) and self.pp_local_rank, compute:
            self.pp_next_rank   (send activations forward; None if last stage)
            self.pp_prev_rank   (recv activations from; None if first stage)
            self.is_pp_first     (owns embeddings / loads raw tokens)
            self.is_pp_last      (owns LM head / computes loss)

        Q: how does this differ from the CP ring? (what must NOT wrap here?)

        TODO: implement.
        """
        raise NotImplementedError

    # ----------------------------------------------------------------------
    # Debug
    # ----------------------------------------------------------------------
    def __repr__(self):
        # TODO: return a one-line summary of THIS rank:
        #   global_rank, its (dp,pp,cp,tp) coord, each axis's ordered ranks,
        #   cp send/recv, pp next/prev. This is your smoke-test output.
        raise NotImplementedError


# --------------------------------------------------------------------------
# gloo smoke test
#   run:  torchrun --nproc_per_node=8 process_group_manager.py
#   config: tp=2 cp=2 pp=2 dp=1  ->  world_size = 8
# --------------------------------------------------------------------------
if __name__ == "__main__":
    # TODO:
    #   1. init the process group on the "gloo" backend (CPU validation first).
    #      Q: where do rank / world_size come from under torchrun?
    #   2. construct ProcessGroupManager(dp=1, pp=2, cp=2, tp=2).
    #   3. print repr(pgm) from every rank.
    #
    # Eyeball the printout:
    #   - every TP group == 2 ranks agreeing on (dp,pp,cp), consecutive globals
    #   - every CP group == 2 ranks at stride 2
    #   - every PP group == 2 ranks at stride 4
    #   - CP send/recv WRAPS; PP next/prev does NOT
    #   - dp=1 -> dp_group is None, dp_local_rank 0
    #
    #   4. tear down: dist.destroy_process_group()
    pass