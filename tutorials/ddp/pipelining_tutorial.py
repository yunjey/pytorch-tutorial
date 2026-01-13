import os
import torch.distributed as dist
import torch
from torch.distributed.pipelining import pipeline, SplitPoint, PipelineStage, ScheduleGPipe
import torch
import torch.nn as nn
from dataclasses import dataclass

global rank, device, pp_group, stage_index, num_stages


@dataclass
class ModelArgs:
    dim: int = 512
    n_layers: int = 8
    n_heads: int = 8
    vocab_size: int = 10000


class Transformer(nn.Module):
    def __init__(self, model_args: ModelArgs):
        super().__init__()

        self.tok_embeddings = nn.Embedding(model_args.vocab_size, model_args.dim)

        # Using a ModuleDict lets us delete layers witout affecting names,
        # ensuring checkpoints will correctly save and load.
        self.layers = torch.nn.ModuleDict()
        for layer_id in range(model_args.n_layers):
            self.layers[str(layer_id)] = nn.TransformerDecoderLayer(model_args.dim, model_args.n_heads)

        self.norm = nn.LayerNorm(model_args.dim)
        self.output = nn.Linear(model_args.dim, model_args.vocab_size)

    def forward(self, tokens: torch.Tensor):
        # Handling layers being 'None' at runtime enables easy pipeline splitting
        h = self.tok_embeddings(tokens) if self.tok_embeddings else tokens

        for layer in self.layers.values():
            h = layer(h, h)

        h = self.norm(h) if self.norm else h
        output = self.output(h).clone() if self.output else h
        return output


def init_distributed():
    global rank, device, pp_group, stage_index, num_stages
    rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    device = torch.device(f"cuda:{rank}") if torch.cuda.is_available() else torch.device("cpu")
    dist.init_process_group()

    # This group can be a sub-group in the N-D parallel case
    pp_group = dist.new_group()
    stage_index = rank
    num_stages = world_size


def manual_model_split(model) -> PipelineStage:
    if stage_index == 0:
        # prepare the first stage model
        for i in range(4, 8):
            del model.layers[str(i)]
        model.norm = None
        model.output = None

    elif stage_index == 1:
        # prepare the second stage model
        for i in range(4):
            del model.layers[str(i)]
        model.tok_embeddings = None

    stage = PipelineStage(
        model,
        stage_index,
        num_stages,
        device,
    )
    return stage


if __name__ == "__main__":
    init_distributed()
    num_microbatches = 4
    model_args = ModelArgs()
    model = Transformer(model_args)

    # Dummy data
    x = torch.ones(32, 500, dtype=torch.long)
    y = torch.randint(0, model_args.vocab_size, (32, 500), dtype=torch.long)
    example_input_microbatch = x.chunk(num_microbatches)[0]

    # Option 1: Manual model splitting
    stage = manual_model_split(model)

    # Option 2: Tracer model splitting
    # stage = tracer_model_split(model, example_input_microbatch)

    model.to(device)
    x = x.to(device)
    y = y.to(device)


    def tokenwise_loss_fn(outputs, targets):
        loss_fn = nn.CrossEntropyLoss()
        outputs = outputs.reshape(-1, model_args.vocab_size)
        targets = targets.reshape(-1)
        return loss_fn(outputs, targets)


    schedule = ScheduleGPipe(stage, n_microbatches=num_microbatches, loss_fn=tokenwise_loss_fn)

    if rank == 0:
        schedule.step(x)
    elif rank == 1:
        losses = []
        output = schedule.step(target=y, losses=losses)
        print(f"losses: {losses}")
    dist.destroy_process_group()
