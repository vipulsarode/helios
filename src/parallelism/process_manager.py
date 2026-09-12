import os
import torch
import torch.distributed as dist


class ProcessGroupManager:
    """4D device mesh (DP, PP, CP, TP) for Helios.

    Grid axis order, outermost -> innermost: dp, pp, cp, tp.
      - TP innermost  : contiguous ranks, so TP all-reduce (twice/layer,
                        blocking) stays on the fastest links.
      - CP just outside TP : ring KV exchange, p2p, overlappable.
      - PP next       : thin boundary-activation send/recv, but critical-path.
      - DP outermost  : reduce-scatter is bulk yet overlappable with backward,
                        so it can tolerate the widest / slowest interconnect cut.

    The whole object is one job: turn the flat `global_rank` the launcher
    gives each process into (a) its per-axis coordinate and (b) the dist
    process groups it participates in on each axis.
    """

    def __init__(self, dp_size, pp_size, cp_size, tp_size):
        self.global_rank = dist.get_rank()
        self.world_size = dist.get_world_size()
        self.local_rank = int(os.environ.get("LOCAL_RANK", self.global_rank % self.world_size))

        assert self.world_size == dp_size * pp_size * cp_size * tp_size, (
            f"world_size={self.world_size} != "
            f"dp={dp_size} * pp={pp_size} * cp={cp_size} * tp={tp_size}"
        )
        self.dp_size, self.pp_size = dp_size, pp_size
        self.cp_size, self.tp_size = cp_size, tp_size

        # coordinate -> rank. Row-major (C-order) so tp varies fastest.
        self.grid = torch.arange(self.world_size).view(dp_size, pp_size, cp_size, tp_size)

        # rank -> coordinate: the inverse decode. Equivalent to the mixed-radix
        # peel (t = r % tp; c = r//tp % cp; ...); nonzero just searches for it.
        self.dp_rank, self.pp_rank, self.cp_rank, self.tp_rank = (
            (self.grid == self.global_rank).nonzero().flatten().tolist()
        )

        # ------------------------------------------------------------------
        # Axis process groups.
        # A group on an axis = "hold every OTHER axis fixed, sweep this one."
        # Enumerating all such slices partitions the world into the groups for
        # that axis; new_subgroups_by_enumeration builds every group and hands
        # each rank back the one it belongs to (element [0]).
        # ------------------------------------------------------------------
        self.tp_group = dist.new_subgroups_by_enumeration(
            [self.grid[d, p, c, :].tolist()
             for d in range(dp_size) for p in range(pp_size) for c in range(cp_size)]
        )[0]
        self.cp_group = dist.new_subgroups_by_enumeration(
            [self.grid[d, p, :, t].tolist()
             for d in range(dp_size) for p in range(pp_size) for t in range(tp_size)]
        )[0]
        self.pp_group = dist.new_subgroups_by_enumeration(
            [self.grid[d, :, c, t].tolist()
             for d in range(dp_size) for c in range(cp_size) for t in range(tp_size)]
        )[0]
        self.dp_group = dist.new_subgroups_by_enumeration(
            [self.grid[:, p, c, t].tolist()
             for p in range(pp_size) for c in range(cp_size) for t in range(tp_size)]
        )[0]

        # Composite groups: sweep two axes at once.
        # cp_dp: everyone sharing (pp, tp). CP shards the sequence, DP shards
        # the batch; reducing across BOTH (with pp/tp fixed) is what you need
        # for a correct global loss / grad-norm over the full replica.
        self.cp_dp_group = dist.new_subgroups_by_enumeration(
            [self.grid[:, p, :, t].flatten().tolist()
             for p in range(pp_size) for t in range(tp_size)]
        )[0]
        self.pp_dp_group = dist.new_subgroups_by_enumeration(
            [self.grid[:, :, c, t].flatten().tolist()
             for c in range(cp_size) for t in range(tp_size)]
        )[0]

        self.world_group = dist.group.WORLD

        # The ordered rank list I belong to on each axis (for neighbour math
        # and for reading membership off at a glance while debugging).
        self.tp_group_ids = self.grid[self.dp_rank, self.pp_rank, self.cp_rank, :].tolist()
        self.cp_group_ids = self.grid[self.dp_rank, self.pp_rank, :, self.tp_rank].tolist()
        self.pp_group_ids = self.grid[self.dp_rank, :, self.cp_rank, self.tp_rank].tolist()
        self.dp_group_ids = self.grid[:, self.pp_rank, self.cp_rank, self.tp_rank].tolist()

        # Per-axis sizes.
        self.tp_world_size = dist.get_world_size(group=self.tp_group)
        self.cp_world_size = dist.get_world_size(group=self.cp_group)
        self.pp_world_size = dist.get_world_size(group=self.pp_group)
        self.dp_world_size = dist.get_world_size(group=self.dp_group)

        self.tp_first_rank, self.tp_last_rank = self.tp_group_ids[0], self.tp_group_ids[-1]
        self.dp_first_rank, self.dp_last_rank = self.dp_group_ids[0], self.dp_group_ids[-1]

        # CP: ring neighbours WRAP (ring attention passes KV around a circle).
        self.cp_first_rank, self.cp_last_rank = self.cp_group_ids[0], self.cp_group_ids[-1]
        self.cp_send_rank = self.cp_group_ids[(self.cp_rank + 1) % self.cp_world_size]
        self.cp_recv_rank = self.cp_group_ids[(self.cp_rank - 1) % self.cp_world_size]

        # PP: neighbours are LINEAR, no wrap; the ends are None (nothing to
        # send past stage 0 / the last stage).
        self.pp_first_rank, self.pp_last_rank = self.pp_group_ids[0], self.pp_group_ids[-1]
        self.pp_is_first_stage = self.pp_rank == 0
        self.pp_is_last_stage = self.pp_rank == self.pp_world_size - 1
        self.pp_next_rank = None if self.pp_is_last_stage else self.pp_group_ids[self.pp_rank + 1]
        self.pp_prev_rank = None if self.pp_is_first_stage else self.pp_group_ids[self.pp_rank - 1]

    def __str__(self):
        return (f"DP({self.dp_world_size})-PP({self.pp_world_size})-"
                f"CP({self.cp_world_size})-TP({self.tp_world_size})-"
                f"Rank({self.global_rank})")


process_group_manager = None


def setup_process_group_manager(dp_size, pp_size, cp_size, tp_size):
    """Build the singleton the rest of Helios reads (Picotron-style global)."""
    global process_group_manager
    process_group_manager = ProcessGroupManager(dp_size, pp_size, cp_size, tp_size)
    return process_group_manager