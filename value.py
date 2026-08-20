import torch
from torch import nn, Tensor
from torch.func import vmap, grad
import torch.nn.functional as F
from parameters import action_count, state_size

class ValueModel(nn.Module):
    def __init__(self):
        super().__init__()
        width = 100
        layer_count = 4
        self.projection = nn.Linear(state_size, width)
        self.layer_norms = nn.ModuleList([nn.LayerNorm(width) for _ in range(layer_count)])
        self.hidden_layers = nn.ModuleList([nn.Linear(width, width) for _ in range(layer_count)])
        self.final_norm = nn.LayerNorm(width)
        self.output_layer = nn.Linear(width, 1)
    def forward(self, x: Tensor) -> Tensor:
        x = self.projection(x)
        for norm, layer in zip(self.layer_norms, self.hidden_layers):
            x = x + layer(F.celu(norm(x)))
        return self.output_layer(self.final_norm(x))
    def __call__(self, *args, **kwds)->Tensor:
        return super().__call__(*args, **kwds)