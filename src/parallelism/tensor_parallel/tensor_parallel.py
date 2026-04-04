"""
Tensor Parallelism — From Scratch                                                
===================================================
"""

import torch
import torch.nn as nn
import torch.distributed as dist
from torch import Tensor
from typing import Optional


# =============================================================================
# Communication Primitives (Autograd-aware)
# =============================================================================

class CopyToParallelRegion(torch.autograd.Function):
    """Forward: identity (copy input to all ranks).
       Backward: all-reduce gradients."""

    @staticmethod
    def forward(ctx, input: Tensor, group: dist.ProcessGroup) -> Tensor:
        ctx.group = group
        return input

    @staticmethod
    def backward(ctx, grad_output: Tensor):
        dist.all_reduce(grad_output, op=dist.ReduceOp.SUM, group = ctx.group)
        return grad_output, None


class ReduceFromParallelRegion(torch.autograd.Function):
    """Forward: all-reduce across ranks.
       Backward: identity (pass gradients through)."""

    @staticmethod
    def forward(ctx, input: Tensor, group: dist.ProcessGroup) -> Tensor:
        dist.all_reduce(input, op=dist.ReduceOp.SUM, group=group)
        return input

    @staticmethod
    def backward(ctx, grad_output: Tensor):
        return grad_output, None


class ScatterToParallelRegion(torch.autograd.Function):
    """Forward: split input along a dimension and scatter to ranks.
       Backward: all-gather gradients."""

    @staticmethod
    def forward(ctx, input: Tensor, group: dist.ProcessGroup) -> Tensor:
        rank=dist.get_rank(group)
        ctx.group=group
        ctx.world_size = dist.get_world_size(group)

        chunks = torch.chunk(input, chunks = ctx.world_size, dim=0)

        return chunks[rank].contiguous()           
    
    @staticmethod
    def backward(ctx, grad_output: Tensor):
        tensor_list = [torch.zeros_like(grad_output) for _ in range(ctx.world_size)]
        dist.all_gather(tensor_list=tensor_list, tensor=grad_output, group=ctx.group)
        return torch.cat(tensor_list, dim=0), None


class GatherFromParallelRegion(torch.autograd.Function):
    """Forward: all-gather shards from all ranks.
       Backward: scatter (split) gradients back."""

    @staticmethod
    def forward(ctx, input: Tensor, group: dist.ProcessGroup) -> Tensor:
        ctx.group=group
        ctx.world_size = dist.get_world_size(group)
        ctx.rank = dist.get_rank(group)

        tensor_list = [torch.zeros_like(input) for _ in range(ctx.world_size)]
        dist.all_gather(tensor_list=tensor_list, tensor=input, group=ctx.group)
        return torch.cat(tensor_list, dim=0)

    @staticmethod
    def backward(ctx, grad_output: Tensor):
        
        chunks = torch.chunk(grad_output, chunks = ctx.world_size, dim=0)
        return chunks[ctx.rank].contiguous(), None     

