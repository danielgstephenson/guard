import math
import torch
from torch import nn, Tensor
import torch.nn.functional as F
from parameters import arena_radius, hit0_value, hit1_value
from physics import check_hits

input_size = 28

def augment(state: Tensor)->Tensor:
    n = state.shape[0]
    vectors = state.reshape(n,8,2)
    a0p, a1p = vectors[:,1,:], vectors[:,3,:]
    b0p, b1p = vectors[:,5,:], vectors[:,7,:]
    rel = torch.stack((a0p-b0p,a0p-b1p,a1p-b1p,a1p-b0p),dim=1)
    aug_vectors = torch.cat((vectors,rel),dim=1)
    magnitudes = torch.sqrt(torch.sum(aug_vectors**2,dim=2))
    return torch.cat((state,magnitudes),dim=1)

def rotate(state: Tensor, turns45: int)->Tensor:
    n = state.shape[0]
    vectors = state.reshape(-1,2)
    angle = turns45 * torch.pi * 0.25
    rot_mat = torch.tensor([
        [math.cos(angle), -math.sin(angle)],
        [math.sin(angle),  math.cos(angle)]
    ])
    rot_vectors = vectors @ rot_mat.t()
    return rot_vectors.reshape(n, -1)

def reflect(state: Tensor)->Tensor:
    flip = torch.tensor([1.0,-1.0])
    n = state.shape[0]
    return (state.reshape(-1,2)*flip).reshape(n,-1)


class ValueModel(nn.Module):
    def __init__(self):
        super().__init__()
        width = 100
        layer_count = 4
        self.projection = nn.Linear(input_size, width)
        self.layer_norms = nn.ModuleList([nn.LayerNorm(width) for _ in range(layer_count)])
        self.hidden_layers = nn.ModuleList([nn.Linear(width, width) for _ in range(layer_count)])
        self.final_norm = nn.LayerNorm(width)
        self.output_layer = nn.Linear(width, 1)
    def forward(self, state: Tensor) -> Tensor:
        n = state.shape[0]
        copies = torch.cat([rotate(s,i) for i in range(8) for s in [state,reflect(state)]])
        values = self.base(copies).reshape(16,n,1)
        return torch.mean(values,dim=0)
    def base(self, state: Tensor) -> Tensor:
        hit0, hit1 = check_hits(state)
        augmented = augment(state)
        return torch.where(hit1,hit1_value,torch.where(hit0,hit0_value,self.net(augmented)))
    def net(self, x: Tensor) -> Tensor:
        x = self.projection(x / arena_radius)
        for norm, layer in zip(self.layer_norms, self.hidden_layers):
            x = x + layer(F.celu(norm(x)))
        return self.output_layer(self.final_norm(x))
    def __call__(self, *args, **kwds)->Tensor:
        return super().__call__(*args, **kwds)
