import torch
import torch.distributed as dist
import os

from src.collectives import AllGather, AllReduce, AlltoAll, ReduceScatter, RingAllReduce, broadcast, reduce


def init():
    dist.init_process_group(backend="gloo")



# ── helpers ───────────────────────────────────────────────────────────────────

def log(rank, test_name, passed):
    status = "PASSED" if passed else "FAILED"
    print(f"[Rank {rank}] {test_name}: {status}")


# ── tests ─────────────────────────────────────────────────────────────────────

def test_broadcast():
    """
    Every rank starts with a different value.
    After broadcast from rank 0, all ranks must hold rank 0's original tensor.
    """
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    # each rank fills its tensor with its own rank value
    tensor = torch.full((4,), float(rank))

    # ground truth: what rank 0 started with
    ref = torch.full((4,), 0.0)

    result = broadcast(tensor, src=0)
    passed = torch.allclose(result, ref)
    log(rank, "broadcast", passed)

    # also test with a non-zero src
    tensor2 = torch.full((4,), float(rank))
    ref2 = torch.full((4,), float(world_size - 1))
    result2 = broadcast(tensor2, src=world_size - 1)
    passed2 = torch.allclose(result2, ref2)
    log(rank, "broadcast (non-zero src)", passed2)


def test_reduce():
    """
    Each rank contributes its rank value.
    After reduce to rank 0, rank 0 must hold the sum 0+1+...+(world_size-1).
    All other ranks' return values are ignored (they sent their data away).
    """
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    tensor = torch.full((4,), float(rank))
    expected_sum = float(sum(range(world_size)))
    ref = torch.full((4,), expected_sum)

    result = reduce(tensor, dst=0)

    if rank == 0:
        passed = torch.allclose(result, ref)
        log(rank, "reduce", passed)
    else:
        log(rank, "reduce", True)   # non-dst ranks just sent, nothing to check


def test_allreduce():
    """
    Same setup as reduce, but every rank must hold the sum at the end.
    """
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    tensor = torch.full((4,), float(rank))
    expected_sum = float(sum(range(world_size)))
    ref = torch.full((4,), expected_sum)

    result = AllReduce(tensor.clone(), dst=0)
    passed = torch.allclose(result, ref)
    log(rank, "allreduce (naive)", passed)


def test_reduce_scatter():
    """
    Verify against dist.reduce_scatter_tensor.
    Each rank contributes rank-valued chunks; after reduce-scatter,
    rank r holds the fully reduced version of chunk r.
    """
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    tensor = torch.full((world_size * 4,), float(rank))

    # ground truth
    ref = torch.zeros(4)
    dist.reduce_scatter_tensor(ref, tensor.clone())

    result = ReduceScatter(tensor.clone())
    passed = torch.allclose(result, ref)
    log(rank, "reduce_scatter", passed)


def test_allgather():
    """
    Each rank starts with a chunk filled with its own rank value.
    After allgather, every rank must hold [0,0,...,1,1,...,2,2,...].
    """
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    chunk = torch.full((4,), float(rank))

    # ground truth
    ref_chunks = [torch.zeros(4) for _ in range(world_size)]
    dist.all_gather(ref_chunks, chunk.clone())
    ref = torch.cat(ref_chunks)

    result_chunks = AllGather(chunk.clone())
    result = torch.cat(result_chunks)
    passed = torch.allclose(result, ref)
    log(rank, "allgather", passed)


def test_ring_allreduce():
    """
    RingAllReduce = ReduceScatter + AllGather.
    Must match dist.all_reduce exactly.
    """
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    tensor = torch.full((world_size * 4,), float(rank))

    # ground truth
    ref = tensor.clone()
    dist.all_reduce(ref)

    result = RingAllReduce(tensor.clone())
    passed = torch.allclose(result, ref)
    log(rank, "ring_allreduce", passed)


def test_alltoall():
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    tensor = torch.cat([
        torch.full((4,), float(rank * 10 + i))
        for i in range(world_size)
    ])

    # ground truth built manually — no dist.all_to_all needed
    # after alltoall, chunk i on rank r should contain what rank i prepared for rank r
    # rank i prepares chunk r as: i*10 + r
    ref = torch.cat([
        torch.full((4,), float(i * 10 + rank))
        for i in range(world_size)
    ])

    result_chunks = AlltoAll(tensor.clone())
    result = torch.cat(result_chunks)
    passed = torch.allclose(result, ref)
    log(rank, "alltoall", passed)


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    init()
    rank = dist.get_rank()

    if rank == 0:
        print(f"\n{'='*50}")
        print(f"  Running collective tests — world_size={dist.get_world_size()}")
        print(f"{'='*50}\n")

    dist.barrier()

    test_broadcast()
    dist.barrier()

    test_reduce()
    dist.barrier()

    test_allreduce()
    dist.barrier()

    # test_reduce_scatter()
    # dist.barrier()

    # test_allgather()
    # dist.barrier()

    # test_ring_allreduce()
    # dist.barrier()

    test_alltoall()
    dist.barrier()

    if rank == 0:
        print(f"\n{'='*50}")
        print("  All tests complete.")
        print(f"{'='*50}\n")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()