"""
Tensor Parallelism — From Scratch                                                
===================================================
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
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
        grad_output = grad_output.clone()
        dist.all_reduce(grad_output, op=dist.ReduceOp.SUM, group = ctx.group)
        return grad_output, None


class ReduceFromParallelRegion(torch.autograd.Function):
    """Forward: all-reduce across ranks.
       Backward: identity (pass gradients through)."""

    @staticmethod
    def forward(ctx, input: Tensor, group: dist.ProcessGroup) -> Tensor:
        input = input.clone()
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

        chunks = torch.chunk(input, chunks = ctx.world_size, dim=-1)

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
        
        chunks = torch.chunk(grad_output, chunks = ctx.world_size, dim=-1)
        return chunks[ctx.rank].contiguous(), None     
    



# =============================================================================
# Functional wrappers
# =============================================================================

def copy_to_parallel_region(input: Tensor, group: dist.ProcessGroup) -> Tensor:
    """Applies CopyToParallelRegion."""
    return CopyToParallelRegion.apply(input, group)

def reduce_from_parallel_region(input: Tensor, group: dist.ProcessGroup) -> Tensor:
    """Applies ReduceFromParallelRegion."""
    return ReduceFromParallelRegion.apply(input, group)


def scatter_to_parallel_region(input: Tensor, group: dist.ProcessGroup) -> Tensor:
    """Applies ScatterToParallelRegion."""
    return ScatterToParallelRegion.apply(input, group)


def gather_from_parallel_region(input: Tensor, group: dist.ProcessGroup) -> Tensor:
    """Applies GatherFromParallelRegion."""
    return GatherFromParallelRegion.apply(input, group)


# =============================================================================
# Parallel Linear Layers
# =============================================================================

class ColumnParallelLinear(nn.Module):
    """Linear layer with weight sharded along the output dimension (columns).
    
    - Weight shape per rank: [output_features // tp_world_size, input_features]
    - Handles input communication and output distribution.
    - gather_output: if True, all-gather output across ranks.
    """

    def __init__(
        self,
        input_features: int,
        output_features: int,
        group: dist.ProcessGroup,
        bias: bool = True,
        gather_output: bool = True,
    ):
        super().__init__()

        self.input_features = input_features
        self.output_features = output_features
        self.group = group
        self.gather_output = gather_output

        self.rank = dist.get_rank()
        self.world_size = dist.get_world_size()

        self.output_partition_size = output_features//self.world_size
        self.weight = nn.Parameter(torch.Tensor(self.output_partition_size, input_features))

        if bias:
            self.bias = nn.Parameter(torch.Tensor(self.output_partition_size))
            with torch.no_grad():
                self.bias.zero_()
        else:
            self.register_parameter("bias", None)


    def forward(self, input: Tensor) -> Tensor:
    
        input = copy_to_parallel_region(x, group=self.group)

        x = F.Linear(input, self.weight, self.bias)

        if self.gather_output:
            x = gather_from_parallel_region(x, group = self.group)

        return x


class RowParallelLinear(nn.Module):
    """Linear layer with weight sharded along the input dimension (rows).
    
    - Weight shape per rank: [output_features, input_features // tp_world_size]
    - Handles partial matmul and all-reduce of results.
    - input_is_parallel: if True, input is already scattered.
    """

    def __init__(
        self,
        input_features: int,
        output_features: int,
        group: dist.ProcessGroup,
        bias: bool = True,
        input_is_parallel: bool = False,
    ):
        super().__init__()

        self.input_features = input_features
        self.output_features = output_features
        self.group = group
        self.input_is_parallel = input_is_parallel

        self.rank = dist.get_rank()
        self.world_size = dist.get_world_size()

        self.input_partition_size = input_features//self.world_size
        self.weight = nn.Parameter(torch.Tensor(output_features, self.input_partition_size))

        if bias:
            self.bias = nn.Parameter(torch.Tensor(self.input_partition_size))
            with torch.no_grad():
                self.bias.zero_()
        else:
            self.register_parameter("bias", None)

    def forward(self, input: Tensor) -> Tensor:
        # TODO: optional scatter + F.linear + reduce
        if not self.input_is_parallel:
            input = scatter_to_parallel_region(input, group=self.group)

        x = F.Linear(input, self.weight)

        x = reduce_from_parallel_region(x, group=self.group)

        return x if self.bias is None else x + self.bias


# =============================================================================
# Parallel Embedding
# =============================================================================

class ParallelEmbedding(nn.Module):
    """Embedding table sharded across the vocabulary dimension.
    
    Each rank owns vocab_size // tp_world_size rows of the embedding table.
    Out-of-range indices on a given rank get zeroed; results are all-reduced.
    """

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        group: dist.ProcessGroup,
    ):
        super().__init__()
        # TODO: compute local vocab range, create local embedding
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.group = group

        self.rank = dist.get_rank()
        self.world_size = dist.get_world_size()
        
        self.local_vocab_size = num_embeddings//self.world_size  
        self.local_vocab_range_start = self.rank*self.local_vocab_size
        self.local_vocab_range_end = self.local_vocab_range_start + self.local_vocab_size

        self.local_embedding = nn.Embedding(self.local_vocab_size, embedding_dim)

    
    def forward(self, input: Tensor) -> Tensor:
        # TODO: mask, embed, all-reduce

        mask = ~((input >= self.local_vocab_range_start) & (input < self.local_vocab_range_end))
        input = input - self.local_vocab_range_start
        input = torch.where(mask, 0, input)

        embedding = self.local_embedding(input)
        embedding_mask = mask.unsqueeze(-1).expand_as(embedding)
        embedding = torch.where(embedding_mask, 0, embedding)
        
        embedding = reduce_from_parallel_region(embedding, group=self.group)

        return embedding
        


# =============================================================================
# SECTION 5: Transformer Block with TP (MLP only for now)
# =============================================================================

class TensorParallelMLP(nn.Module):
    """Standard 2-layer MLP sharded via TP.
    
    Pattern: ColumnParallel -> activation -> RowParallel
    No redundant communication between the two linears.
    """

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        group: dist.ProcessGroup,
    ):
        super().__init__()
        # TODO: wire up ColumnParallelLinear and RowParallelLinear
        self.group = group
        self.model = nn.Sequential(
            ColumnParallelLinear(hidden_size, intermediate_size, group=self.group, bias=True, gather_output=False),
            nn.GELU(),
            RowParallelLinear(intermediate_size, hidden_size, group=self.group, bias=True, input_is_parallel=True)
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.model(x)


class TensorParallelAttention(nn.Module):
    """Multi-head attention with Q, K, V as ColumnParallel
    and output projection as RowParallel.
    
    Each rank handles num_heads // tp_world_size heads.
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        group: dist.ProcessGroup,
    ):
        super().__init__()
        # TODO: Q/K/V as ColumnParallel (gather_output=False), O as RowParallel
        pass

    def forward(self, x: Tensor) -> Tensor:
        # TODO: project, reshape to local heads, attention, project back
        pass


# =============================================================================
# SECTION 6: Weight Loading / Sharding Utilities
# =============================================================================

def shard_model_for_tp(
    model: nn.Module,
    group: dist.ProcessGroup,
) -> nn.Module:
    """Takes a full (unsharded) model and replaces layers with TP equivalents.
    
    - Identifies nn.Linear layers that should become Column/Row parallel.
    - Slices pretrained weights correctly per rank.
    - Returns the modified model.
    """
    pass


def load_sharded_weights(
    model: nn.Module,
    state_dict: dict,
    group: dist.ProcessGroup,
) -> None:
    """Loads a full state_dict into a TP-sharded model.
    
    Each rank slices the weights it owns from the full state_dict.
    """
    pass


# =============================================================================
# SECTION 7: Main / Test Harness
# =============================================================================

def main():
    """
    - Initialize process group
    - Create a small transformer / MLP
    - Shard it for TP
    - Run a forward + backward pass
    - Verify gradients are correct across ranks
    """
    pass


if __name__ == "__main__":
    main()


