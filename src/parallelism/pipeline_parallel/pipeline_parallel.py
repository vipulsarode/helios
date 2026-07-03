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
import torch.nn.functional as F
from typing import Optional, List, Tuple, Any

from config import cfg


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

        self.is_first_stage = False
        self.is_last_stage = False
        
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
        # self.full_model_builder = full_model_builder
        print([type(c).__name__ for c in full_model_builder.children()])
        self.pp_context = pp_context

        children = list(full_model_builder.children())

        embed_head = children[0]
        lm_head = children[2]


        self.model_layers = list(children[1])
        total_layers = len(self.model_layers)
        self.num_layers = total_layers // self.pp_context.world_size

        self.stage_layers = self.model_layers[self.pp_context.rank*self.num_layers : (self.pp_context.rank+1)*self.num_layers]
        
        if pp_context.is_first_stage:
            self.stage_layers = [embed_head] + self.stage_layers
        if pp_context.is_last_stage:
            self.stage_layers = self.stage_layers + [lm_head]


        self.stage_model = nn.Sequential(*self.stage_layers)


    def forward(self, x: torch.Tensor, labels=None) -> torch.Tensor:
        
        if self.pp_context.is_last_stage:
            logits = self.stage_model(x) 
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1))
            print(f"Logits Shape: {logits.shape}, Loss Shape: {loss.shape}")
            return loss
        
        activations = self.stage_model(x)

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
        
        buffer = torch.empty(shape, dtype=dtype)
        dist.recv(buffer, src=self.pp_context.prev_rank)
        buffer.requires_grad=True
        return buffer


    def send_forward(self, activation: torch.Tensor) -> None:
        """Send activation to next rank during forward pass."""
        dist.send(activation, dst=self.pp_context.next_rank)


    def recv_backward(self, shape, dtype) -> Optional[torch.Tensor]:
        """Receive grad from next rank during backward pass."""
        
        buffer = torch.empty(shape, dtype=dtype)
        dist.recv(buffer, src=self.pp_context.next_rank)
        # buffer.requires_grad=True
        return buffer

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
    outputs = []
    losses = []

 
    shape = (cfg.pp.batch_size, cfg.pp.seq_len, cfg.pp.d_model)
    dtype = cfg.pp.dtype
    
    # forward loop
    for i in range(len(mbs)):
        
        if pp_context.is_first_stage:
            input = mbs[i]
            output = stage.forward(input)
            output.requires_grad_(True)
            comms.send_forward(output.detach())
            outputs.append(output)
        elif pp_context.is_last_stage:
            input = comms.recv_forward(shape, dtype)
            print(f"Last stage input: {input.shape}, Last stage targets: {target_mbs[i].shape}")
            loss = stage.forward(input, labels=target_mbs[i])
            losses.append(loss)
        else:
            input = comms.recv_forward(shape, dtype)
            output = stage.forward(input)
            comms.send_forward(output.detach())
            outputs.append(output)

        inputs.append(input)
        

    total_loss = 0

    # backward loop
    for i in range(len(mbs)):

        if pp_context.is_first_stage:
            grad = comms.recv_backward(shape, dtype)
            outputs.pop(-1).backward(grad)  
        elif pp_context.is_last_stage:
            loss = losses.pop(-1)/num_microbatches
            print("LOSS :", loss)
            loss.backward()
            total_loss += loss.item()
            comms.send_backward(inputs.pop(-1).grad)
        else:
            grad = comms.recv_backward(shape, dtype)
            outputs.pop(-1).backward(grad)
            comms.send_backward(inputs.pop(-1).grad)

    return total_loss


def pipeline_step_1f1b(
    stage: PipelineStage,
    comms: PipelineComms,
    pp_context: PipelineParallelContext,
    batch: torch.Tensor,
    targets: Optional[torch.Tensor],
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
    
    world_size = dist.get_world_size()
    rank = dist.get_rank()

    mbs = split_batch_into_microbatches(batch=batch, num_microbatches=num_microbatches)
    if pp_context.is_last_stage:
        target_mbs = split_batch_into_microbatches(batch=targets, num_microbatches=num_microbatches)

    shape = (cfg.pp.batch_size, cfg.pp.seq_len, cfg.pp.d_model)
    dtype = cfg.pp.dtype

    warmup_steps = min(world_size - rank - 1, num_microbatches - 1)
    steady_steps = len(mbs) - warmup_steps
    drain_steps = warmup_steps

    inputs, outputs = [], []

    total_loss = 0

    # warmup
    for i in range(min(warmup_steps, num_microbatches)):
            
        if pp_context.is_first_stage:
            output_activation = stage.forward(mbs[i])
            comms.send_forward(output_activation)
            outputs.append(output_activation)
        elif pp_context.is_last_stage:
            pass
        else:    
            input_activation = comms.recv_forward(shape, dtype)
            output_activation = stage.forward(input_activation)
            comms.send_forward(output_activation)
            inputs.append(input_activation)
            outputs.append(output_activation)

    input_activation = None  
    # steady
    for i in range(steady_steps):
        
        if pp_context.is_first_stage:
            output_activation = stage.forward(mbs[warmup_steps + i]) 
            outputs.append(output_activation)
            output_grad = comms.send_forward_recv_backward(output_activation, shape, dtype)
            backward_step(pp_context=pp_context, input_activation=None, output_activation=outputs.pop(0), output_grad=output_grad)
        
        elif pp_context.is_last_stage:
            
            if input_activation is None:
                input_activation = comms.recv_forward(shape, dtype)

            loss = stage.forward(input_activation, target_mbs[i])
            loss = loss/num_microbatches
            total_loss += loss
            grad = backward_step(pp_context=pp_context, input_activation=input_activation, output_activation=None, output_grad = None,loss=loss) 

            if i == steady_steps - 1:
                comms.send_backward(grad)
            else:    
                input_activation = comms.send_backward_recv_forward(grad, shape, dtype)

        else:
            
            if input_activation is None:
                input_activation = comms.recv_forward(shape, dtype)

            output_activation = stage.forward(input_activation)
            outputs.append(output_activation)
            inputs.append(input_activation)
            output_grad = comms.send_forward_recv_backward(output_activation, shape, dtype)
            grad = backward_step(pp_context=pp_context, input_activation=inputs.pop(0), output_activation=outputs.pop(0), output_grad=output_grad)
            
            if i == steady_steps - 1:
                comms.send_backward(grad)
            else:    
                input_activation = comms.send_backward_recv_forward(grad, shape, dtype)
            

    # drain
    for i in range(drain_steps):
        
        if pp_context.is_last_stage:
            pass

        elif pp_context.is_first_stage:
            output_grad = comms.recv_backward(shape, dtype)
            backward_step(pp_context=pp_context, input_activation=None, output_activation=outputs.pop(0), output_grad=output_grad, loss = None)

        else:
            output_grad = comms.recv_backward(shape, dtype)
            grad = backward_step(pp_context=pp_context, input_activation=inputs.pop(0), output_activation=outputs.pop(0), output_grad=output_grad, loss = None)
            comms.send_backward(grad)
    
    
    assert (len(inputs) == 0 and len(outputs) == 0 ), f"rank {rank}: inputs={len(inputs)} outputs={len(outputs)}"
    print(f"Microbatches on the rank {rank} are drained successfully!")

    return total_loss
# =============================================================================
# BACKWARD HELPER
# =============================================================================

def backward_step(
    pp_context : PipelineParallelContext,
    input_activation: Optional[torch.Tensor],
    output_activation: Optional[torch.Tensor],
    output_grad: Optional[torch.Tensor],
    loss: Optional[torch.Tensor] = None
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

    if pp_context.is_last_stage:
        loss.backward()
        return input_activation.grad
    elif pp_context.is_first_stage:
        output_activation.backward(output_grad)
    else:
        output_activation.backward(output_grad)
        return input_activation.grad



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

    rank = dist.get_rank()

    optimizer.zero_grad()

    if schedule == "afab":
        loss = pipeline_step_afab(stage, comms, pp_context, batch, targets, num_microbatches)
    if schedule == "1f1b":
        loss = pipeline_step_1f1b(stage, comms, pp_context, batch, targets, num_microbatches)

    p = next(stage.parameters())
    before = p.detach().clone()
    optimizer.step()
    print(f"rank {rank} param delta: {(p - before).norm().item():.6e}")



    return loss


# =============================================================================
# MAIN / TEST HARNESS
# =============================================================================


class TransformerBlock(nn.Module):
    def __init__(self, d_model=64, nhead=4, ff_dim=128, dropout=0.0):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, nhead, batch_first=True, dropout=dropout)
        self.ff = nn.Sequential(
            nn.Linear(d_model, ff_dim),
            nn.GELU(),
            nn.Linear(ff_dim, d_model),
        )
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)

    def forward(self, x):
        x = x + self.attn(x, x, x, need_weights=False)[0]
        x = self.ln1(x)
        x = x + self.ff(x)
        x = self.ln2(x)
        return x

class TinyTransformer(nn.Module):
    def __init__(self, vocab_size=256, d_model=64, nhead=4, ff_dim=128, num_layers=8):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        self.vocab_size = vocab_size
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, nhead, ff_dim) for _ in range(num_layers)
        ])
        self.head = nn.Linear(d_model, vocab_size)
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, input_ids):
        x = self.embed(input_ids)
        for block in self.blocks:
            x = block(x)
        logits = self.head(x)

        # if self.is_last_stage and labels is not None:
        #     return self.loss_fn(logits.reshape(-1, self.vocab_size), labels.reshape(-1))
        
        return logits 

def main():
    """
    - init process group
    - build PipelineParallelContext
    - build stage from a toy transformer
    - run a few train_steps
    - validate loss decreases
    - compare against single-GPU reference run for correctness
    """
    
    dist.init_process_group(backend="gloo")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    process_group = dist.GroupMember.WORLD

    


    
    num_microbatches = 4

    
    # batches = torch.randint(0, 256, (8, 32))
    # target_batches = torch.randint(0, 256, (8, 32))

    vocab_size = 256
    B, S = 8, 32

    # A small FIXED dataset so there's real structure to learn (not fresh noise each step)
    torch.manual_seed(0)
    pp_context = PipelineParallelContext(pp_group=process_group)
    transformer_model = TinyTransformer()
    comms = PipelineComms(pp_context)
    stage = PipelineStage(transformer_model, pp_context=pp_context)
    optimizer = torch.optim.SGD(stage.parameters(), lr=0.01)


    batches = torch.randint(0, vocab_size, (B, S))
    target_batches = batches.clone()  

    torch.manual_seed(0)
    ref_model = TinyTransformer()             # full model, one process
    ref_opt = torch.optim.SGD(ref_model.parameters(), lr=0.01)
    # same fixed copy-task batch
    
    print("weights match:", torch.equal(
    transformer_model.head.weight, ref_model.head.weight))
    
    for i in range(15):
        ref_opt.zero_grad()
        logits = ref_model(batches)   # full forward, scalar loss
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), target_batches.view(-1))
        loss.backward()
        ref_opt.step()
        print(f"ref step {i}: {loss.item():.6f}")


    batch_loss = 0
    for i in range(15):
        # for batch, target_batch in zip(batches, target_batches):
        loss = train_step(stage, comms, pp_context, optimizer, batches, target_batches, num_microbatches, schedule="1f1b")
        batch_loss += loss
        
        print(f"Epoch {i+1}: {loss} ")
        batch_loss = 0

if __name__ == "__main__":
    main()