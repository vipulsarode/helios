"""
pipeline_parallel.py

From-scratch implementation of Pipeline Parallelism, Picotron inspired.
Supports:
  - AFAB (All-Forward-All-Backward)
  - 1F1B (One-Forward-One-Backward)
"""

import torch
import torch.nn as nn
import torch.distributed as dist
from typing import Optional, List, Tuple, Any


# =============================================================================
# PROCESS GROUP / CONTEXT
# =============================================================================

class PipelineParallelContext:
    """
    Holds PP-specific distributed state.

    Responsibilities:
      - Track pp_rank, pp_world_size
      - Identify is_first_stage / is_last_stage
      - Know prev_rank and next_rank for send/recv
      - Hold the PP process group
    """
    def __init__(self, pp_group: dist.ProcessGroup):
        self.rank = dist.get_rank()
        self.world_size = dist.get_world_size()

        if self.rank == 0:
            self.is_first_stage = True
        if self.rank == self.world_size - 1:
            self.is_last_stage = True

        if self.is_first_stage:
            self.prev_rank = None
            self.next_rank = self.rank + 1
        elif self.is_last_stage:
            self.prev_rank = self.rank - 1
            self.next_rank = None
        else:
            self.prev_rank = self.rank - 1
            self.next_rank = self.rank + 1


# =============================================================================
# MODEL PARTITIONING
# =============================================================================

class PipelineStage(nn.Module):
    """
    Wraps the subset of layers owned by this PP rank.

    Responsibilities:
      - Given a full model (or model config) and pp_rank/pp_world_size,
        slice out the layers this rank is responsible for.
      - Handle embedding placement (first stage) and lm_head placement
        (last stage).
      - Expose a clean .forward() that takes an activation tensor and
        returns an activation tensor (or loss, if last stage).
    """
    def __init__(self, full_model_builder, pp_context: PipelineParallelContext):
        super().__init__()
        self.full_model_builder = full_model_builder
        self.pp_context = pp_context

        self.num_layers = len(self.full_model_builder.named_modules()) // self.pp_context.world_size
        self.layers = self.full_model_builder[self.pp_context.rank*self.num_layers : (self.pp_context.rank+1)*self.num_layers]
        self.total_loss = []

    def forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        
        if self.pp_context.is_last_stage:
            for layer in self.layers:
                loss = layer(x, *args, *kwargs) # input labels here
                self.total_loss.append(loss)
            return self.total_loss.sum()
        
        for layer in self.layers:
            activations = layer(x)

        return activations


# =============================================================================
# POINT-TO-POINT COMMUNICATION
# =============================================================================

class PipelineComms:
    """
    All send/recv primitives used by the schedule.

    Responsibilities:
      - Shape/dtype negotiation (so recv knows what buffer to allocate).
      - Blocking send/recv for AFAB.
      - Non-blocking (isend/irecv) send/recv for 1F1B overlap.
      - Batched send+recv for 1F1B steady state (e.g. batch_isend_irecv).
      - Handle the edge cases: first stage has no prev, last stage has no next.
    """

    def __init__(self, pp_context: PipelineParallelContext):
        self.pp_context = pp_context

    def recv_forward(self, shape, dtype) -> Optional[torch.Tensor]:
        """Receive activation from prev rank during forward pass."""
        
        buffer = torch.empty(shape, dtype=dtype, requires_grad=True)
        dist.recv(buffer, src=self.pp_context.prev_rank)


    def send_forward(self, activation: torch.Tensor) -> None:
        """Send activation to next rank during forward pass."""
        
        dist.send(activation, dst=self.pp_context.next_rank)


    def recv_backward(self, shape, dtype) -> Optional[torch.Tensor]:
        """Receive grad from next rank during backward pass."""
        
        buffer = torch.empty(shape, dtype=dtype, requires_grad=True)
        dist.recv(buffer, src=self.pp_context.next_rank)
        

    def send_backward(self, grad: torch.Tensor) -> None:
        """Send grad to prev rank during backward pass."""
        
        dist.send(grad, dst = self.pp_context.prev_rank)
        

    def send_forward_recv_backward(self, activation: torch.Tensor, shape, dtype):
        """Fused op for 1F1B steady state."""
        
        recv_buffer = torch.empty(shape, dtype=dtype, requires_grad=True)
        send_op = dist.P2POp(dist.isend, activation, peer=self.pp_context.next_rank)
        recv_op = dist.P2POp(dist.irecv, recv_buffer, peer=self.pp_context.next_rank)
        reqs = dist.batch_isend_irecv([send_op, recv_op])
        [req.wait() for req in reqs]
        
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        if torch.cpu.is_available():
            torch.cpu.synchronize()

        return recv_buffer

    def send_backward_recv_forward(self, grad: torch.Tensor, shape, dtype):
        """Fused op for 1F1B steady state."""

        recv_buffer = torch.empty(shape, dtype=dtype, requires_grad=True)
        send_op = dist.P2POp(dist.isend, grad, peer=self.pp_context.prev_rank)
        recv_op = dist.P2POp(dist.irecv, recv_buffer, peer=self.pp_context.prev_rank)
        reqs = dist.batch_isend_irecv([send_op, recv_op])
        [req.wait() for req in reqs]
        
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        if torch.cpu.is_available():
            torch.cpu.synchronize()

        return recv_buffer


# =============================================================================
# MICROBATCH UTILITIES
# =============================================================================

def split_batch_into_microbatches(
    batch: torch.Tensor,
    num_microbatches: int,
) -> List[torch.Tensor]:
    """
    Split a global batch into `num_microbatches` equal chunks along batch dim.
    """
    mbs = torch.chunk(batch, num_microbatches, dim=0)
    return mbs


# =============================================================================
# SCHEDULES
# =============================================================================

def pipeline_step_afab(
    stage: PipelineStage,
    comms: PipelineComms,
    pp_context: PipelineParallelContext,
    batch: torch.Tensor,
    targets: Optional[torch.Tensor],
    loss_fn,
    num_microbatches: int,
) -> torch.Tensor:
    """
    All-Forward-All-Backward schedule.

    Responsibilities:
      - Phase 1: For each microbatch, recv_forward -> stage.forward ->
                 send_forward. Cache activations (and inputs) for backward.
      - Phase 2: For each microbatch in reverse, recv_backward ->
                 backward through cached activations -> send_backward.
      - Accumulate loss on last stage.
      - Return total loss (only meaningful on last stage).

    Key design questions you need to answer:
      - Where do you store activations between phases?
      - How do you run backward for a non-last stage (what is the loss proxy)?
      - How do you retain_grad / require_grad on received activations so
        grads can flow back?
    """
    
    mbs = split_batch_into_microbatches(batch=batch, num_microbatches=num_microbatches)
    if pp_context.is_last_stage:
        target_mbs = split_batch_into_microbatches(batch=targets, num_microbatches=num_microbatches)

    inputs = []
    activations = []
    losses = []


    shape = mbs[0].shape
    dtype = mbs[0].dtype
    
    # forward loop
    for i in range(len(mbs)):
        
        if pp_context.is_first_stage:
            input = mbs[i]
            activation = stage.forward(input)
            comms.send_forward(activation.detach())
        elif pp_context.is_last_stage:
            input = comms.recv_forward(shape, dtype)
            output = stage.forward(input)
            loss = loss_fn(output, target_mbs[i])
        else:
            input = comms.recv_forward(shape, dtype)
            activation = stage.forward(input)
            comms.send_forward(activation.detach())

        inputs.append(input)
        activations.append(activation)
        losses.append(loss)

    total_loss = 0

    # backward loop
    for i in range(len(mbs)):

        if pp_context.is_first_stage:
            grad = comms.recv_backward(shape, dtype)
            activations.pop(-1).backward(grad)   
        
        elif pp_context.is_last_stage:
            loss = losses.pop(-1)/num_microbatches
            loss.backward()
            total_loss += loss
            comms.send_backward(inputs.pop(-1).grad)
        
        else:
            grad = comms.recv_backward(shape, dtype)
            activations.pop(-1).backward(grad)
            comms.send_backward(inputs.pop(-1).grad)


    return total_loss


def pipeline_step_1f1b(
    stage: PipelineStage,
    comms: PipelineComms,
    pp_context: PipelineParallelContext,
    batch: torch.Tensor,
    targets: Optional[torch.Tensor],
    loss_fn,
    num_microbatches: int,
) -> torch.Tensor:
    """
    1F1B schedule.

    Responsibilities:
      - Warmup: each rank does (pp_world_size - pp_rank - 1) forwards
        before starting backwards.
      - Steady state: alternate 1 forward and 1 backward per step, using
        fused send/recv ops to overlap comm with compute.
      - Cooldown: drain remaining backwards.
      - Accumulate loss on last stage.

    Key design questions:
      - How do you queue pending activations in FIFO order so the right
        one is consumed during the matching backward?
      - How do you handle the first and last ranks' asymmetric schedules?
      - When exactly do you call fused vs non-fused comm ops?
    """
    pass


# =============================================================================
# BACKWARD HELPER
# =============================================================================

def backward_step(
    input_activation: Optional[torch.Tensor],
    output_activation: torch.Tensor,
    output_grad: Optional[torch.Tensor],
    is_last_stage: bool,
    loss: Optional[torch.Tensor] = None,
) -> Optional[torch.Tensor]:
    """
    Run backward for a single microbatch on this stage.

    Responsibilities:
      - If last stage: backward from the loss scalar.
      - Else: backward from output_activation using output_grad as the
        incoming gradient (torch.autograd.backward with grad_tensors).
      - Return input_activation.grad so it can be sent to prev rank.
      - Handle the first stage case (no grad to send).
    """
    pass


# =============================================================================
# ENTRY POINT
# =============================================================================

def train_step(
    stage: PipelineStage,
    comms: PipelineComms,
    pp_context: PipelineParallelContext,
    optimizer: torch.optim.Optimizer,
    batch: torch.Tensor,
    targets: torch.Tensor,
    loss_fn,
    num_microbatches: int,
    schedule: str = "1f1b",
) -> torch.Tensor:
    """
    Full training step:
      - zero_grad
      - dispatch to AFAB or 1F1B
      - optimizer.step
      - return loss
    """
    pass


# =============================================================================
# MAIN / TEST HARNESS
# =============================================================================

def main():
    """
    - init process group
    - build PipelineParallelContext
    - build stage from a toy transformer
    - run a few train_steps
    - validate loss decreases
    - compare against single-GPU reference run for correctness
    """
    pass


if __name__ == "__main__":
    main()