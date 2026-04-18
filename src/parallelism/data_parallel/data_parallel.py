"""
Data Parallelism 
=================================================
Implement everything from scratch. 
"""

import torch
import torch.distributed as dist
import torch.nn as nn
from torch import Tensor
from typing import List, Dict, Optional


# =============================================================================
# SECTION 1: Gradient Bucketing
# =============================================================================

class DataParallelBucket:
    """
    A single bucket that accumulates gradients for a subset of parameters
    and fires an all-reduce when full.
    
    __init__(self, params, dp_group):
        - Store the parameters assigned to this bucket
        - Allocate a flat buffer to hold all their gradients contiguously
        - Track how many gradients have been accumulated vs total expected
        - Store the process group for communication
    
    reset(self):
        - Reset accumulation state for next backward pass
        - Zero out the flat buffer
    
    all_reduce(self):
        - Launch async all-reduce on the flat buffer (average across DP ranks)
        - Return the async handle
    
    copy_grads_to_buffer(self):
        - Copy each parameter's .grad (or .main_grad) data into
          the correct offset of the flat buffer
    
    copy_buffer_to_grads(self):
        - Copy averaged gradients from the flat buffer back into
          each parameter's .grad (or .main_grad)
    """
    
    def __init__(self, params: List[nn.Parameter], dp_group: dist.ProcessGroup):
        self.params = params
        self.dp_group = dp_group
        self.flat_buffer = torch.zeros(sum(p.numel() for p in self.params), device=params[0].device).contiguous()
        
        self.async_ar_handle = None


    def reset(self):
        self.async_ar_handle = None
        self.flat_buffer.zero_()

    def all_reduce(self):
        self.async_ar_handle = dist.all_reduce(self.flat_buffer, op=dist.ReduceOp.AVG, group=self.dp_group, async_op=True)
        return self.async_ar_handle

    def copy_grads_to_buffer(self):
        
        offset = 0
        for p in self.params:
            self.flat_buffer[offset:offset + p.numel()].view(p.shape).copy_(p.grad)
            offset += p.numel()

    def copy_buffer_to_grads(self):
        offset = 0
        for p in self.params:
            p.grad.copy_(self.flat_buffer[offset:offset + p.numel()].view(p.shape))
            offset += p.numel()


# =============================================================================
# SECTION 2: Bucket Manager
# =============================================================================

class BucketManager:
    """
    Manages multiple buckets and orchestrates the backward-pass overlap
    of gradient computation with all-reduce communication.
    
    __init__(self, model, dp_group, bucket_size_mb):
        - Iterate over model parameters in REVERSE order
        - Assign parameters to buckets based on size threshold
        - Register AccumulateGrad hooks on each parameter
        - Build mapping from parameter -> bucket index
    
    _build_buckets(self, params, bucket_size_mb):
        - Partition parameters into buckets where each bucket's
          total gradient size doesn't exceed bucket_size_mb
        - Return list of DataParallelBucket objects
    
    _register_hooks(self):
        - For each parameter, register a hook on its AccumulateGrad node
        - The hook should: mark the param as "grad ready" in its bucket,
          and when a bucket is full, fire its all-reduce
    
    _hook_fn(self, param, bucket_idx):
        - Called when param's gradient is computed
        - Increment the bucket's ready count
        - If bucket is full: copy grads to buffer, launch all-reduce
    
    wait_all_reduces(self):
        - Wait on all pending async all-reduce handles
        - Copy averaged gradients back from buffers to param grads
    
    reset(self):
        - Reset all buckets for the next iteration
    """

    def __init__(self, model: nn.Module, dp_group: dist.ProcessGroup, bucket_size_mb: float = 25.0):
        
        self.model = model
        self.dp_group = dp_group
        self.bucket_size_mb = bucket_size_mb
        self.param_to_bucket = {}

        # Assuming torch.float32
        self.grad_size = 4
        self.total_grad_in_bucket = bucket_size_mb*1024*1024//self.grad_size

        reversed_params = reversed([param for param in model.parameters()])
        
        self.buckets = self._build_buckets(reversed_params)
        self.num_params_ready_per_bucket = [0 for _ in range(len(self.buckets))]
        self.async_reqs = []
        

        self._register_hooks()

    def _build_buckets(self, params: List[nn.Parameter]) -> List[DataParallelBucket]:
        
        curr_bucket_size = 0
        curr_bucket_params = []
        buckets = []

        for p in params:
            if p.requires_grad:
                curr_bucket_params.append(p)
                curr_bucket_size += p.numel()

                if curr_bucket_size >= self.total_grad_in_bucket:
                    bucket = DataParallelBucket(curr_bucket_params, dp_group=self.dp_group)
                    buckets.append(bucket)
                    for curr_p in curr_bucket_params:
                        self.param_to_bucket[curr_p] = bucket
                    curr_bucket_size = 0
                    curr_bucket_params = [] 

        if curr_bucket_params:
            bucket = DataParallelBucket(curr_bucket_params, dp_group=self.dp_group)
            buckets.append(bucket)            
            for p in curr_bucket_params:
                self.param_to_bucket[p] = bucket
        
        return buckets           

    def _register_hooks(self):
        
        for p in self.model.parameters():
            if p.requires_grad:
                p.register_post_accumulate_grad_hook(self._hook_fn(p, self.buckets.index(self.param_to_bucket[p]))) 

    def _hook_fn(self, param: nn.Parameter, bucket_idx: int):

        def hook(*unused):
            bucket_params = len(self.buckets[bucket_idx].params)
            self.num_params_ready_per_bucket[bucket_idx] += 1

            if self.num_params_ready_per_bucket[bucket_idx]==bucket_params: 
                self.buckets[bucket_idx].copy_grads_to_buffer()
                self.async_reqs.append(self.buckets[bucket_idx].all_reduce())
        return hook

    def wait_all_reduces(self):
        for req in self.async_reqs:
            req.wait()
        for bucket in self.buckets:
            bucket.copy_buffer_to_grads()

    def reset(self):
        for bucket in self.buckets:
            bucket.flat_buffer.zero_()
        self.async_reqs = []
        self.num_params_ready_per_bucket = [0 for _ in range(len(self.buckets))]



# =============================================================================
# SECTION 3: Main Grad Buffer (Optional but Picotron does this)
# =============================================================================

def set_main_grad_buffers(model: nn.Module):
    """
    For each parameter in the model:
    - Allocate a persistent .main_grad tensor (same shape/dtype/device)
    - Register an AccumulateGrad hook that writes the gradient into
      .main_grad instead of the default .grad
    - This decouples gradient storage from autograd's internal buffers
    """
    pass


# =============================================================================
# SECTION 4: Data Parallel Wrapper
# =============================================================================

class DataParallel:
    """
    Top-level orchestrator that wraps a model for data-parallel training.
    
    __init__(self, model, dp_group, bucket_size_mb):
        - Store model and process group
        - Optionally set up main_grad buffers
        - Create BucketManager
    
    pre_backward(self):
        - Reset bucket manager state before backward pass
    
    post_backward(self):
        - Wait for all all-reduces to complete
        - Ensure gradients are synchronized and written back
    
    broadcast_params(self):
        - Broadcast all model parameters from rank 0 to all DP ranks
        - Called once at initialization to ensure identical starting weights
    """

    def __init__(self, model: nn.Module, dp_group: dist.ProcessGroup, bucket_size_mb: float = 25.0):
        pass

    def pre_backward(self):
        pass

    def post_backward(self):
        pass

    def broadcast_params(self):
        pass


# =============================================================================
# SECTION 5: Training Loop Integration
# =============================================================================

def train_step(model, dp_wrapper, optimizer, data, labels, loss_fn):
    """
    Single training step with data parallelism.
    
    1. Forward pass
    2. Compute loss
    3. dp_wrapper.pre_backward()
    4. loss.backward()
    5. dp_wrapper.post_backward()
    6. optimizer.step()
    7. optimizer.zero_grad() (or zero main_grads)
    """
    pass


# =============================================================================
# SECTION 6: Process Group Setup (if you want end-to-end)
# =============================================================================

def setup_dp_group(world_size: int, rank: int):
    """
    Initialize distributed backend and create the DP process group.
    """
    pass


if __name__ == "__main__":
    # Your end-to-end test goes here.
    pass