import torch
import torch.distributed as dist


def broadcast(tensor, src = 0):

    rank = dist.get_rank()
    world_size = dist.get_world_size()

    if rank == src:
        for dst in range(world_size):
            if dst != src:
                dist.send(tensor=tensor, dst = dst)
    else:
        dist.recv(tensor=tensor, src = src)
    return tensor


def reduce(tensor, dst = 0):

    rank = dist.get_rank()
    world_size = dist.get_world_size()

    if rank == dst:
        buffer = torch.zeros_like(tensor)
        for src in range(world_size):
            if dst!= src:
                dist.recv(tensor=buffer, src=src)
                tensor += buffer
    else:
        dist.send(tensor = tensor, dst = dst)

    return tensor


def AllReduce(tensor, dst = 0):

    tensor = reduce(tensor=tensor, dst = dst)
    tensor = broadcast(tensor=tensor, src = dst)

    return tensor


def ReduceScatter(tensor):

    rank = dist.get_rank()
    world_size = dist.get_world_size()

    tensor_chunks = list(torch.chunk(tensor, world_size))

    buffer = torch.zeros_like(tensor_chunks[0])
    
    left_rank = (rank - 1) % world_size
    right_rank = (rank + 1) % world_size    
    
    for i in range(world_size):
            dist.send(tensor_chunks[(rank-i)%world_size], dst = right_rank)
            dist.recv(buffer, src = left_rank)
            tensor_chunks[(left_rank - i) % world_size] += buffer
            
    return tensor_chunks[rank]


def AllGather(reduced_chunk):
    
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    left_rank = (rank - 1) % world_size
    right_rank = (rank + 1) % world_size

    buffer = torch.empty_like(reduced_chunk)

    full_tensor = [None] * world_size
    full_tensor[rank] = reduced_chunk
    
    for i in range(world_size-1):
        dist.send(full_tensor[(rank-i)%world_size], dst = right_rank)
        dist.recv(buffer, src = left_rank)
        full_tensor[(left_rank - i)%world_size] = buffer.clone()
    
    return full_tensor


def RingAllReduce(tensor):
    reduced_chunk = ReduceScatter(tensor)
    full_tensor = AllGather(reduced_chunk)
    return torch.cat(full_tensor)

def AlltoAll(tensor):

    rank = dist.get_rank()
    world_size = dist.get_world_size()

    tensor_chunks = list(torch.chunk(tensor, world_size))

    full_tensor = [None]*world_size
    full_tensor[rank] = tensor_chunks[rank]

    send_ops = []
    recv_ops = []

    
    for i in range(world_size):
        if i != rank:
            send_op = dist.P2POp(dist.isend, tensor_chunks[i], dst = i)
            send_ops.append(send_op)
        else:
            continue

    for i in range(world_size):
        buffer = torch.empty_like(tensor_chunks[0])
        if i != rank:
            recv_op = dist.P2POp(dist.irecv, buffer, src = i)
            recv_ops.append(recv_op)
            full_tensor[i] = buffer
        else:
            continue

    reqs = dist.batch_isend_irecv(send_ops + recv_ops)
    for req in reqs:
        req.wait()


    return full_tensor