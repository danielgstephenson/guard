import torch
from torch import Tensor
import torch.nn.functional as F
from parameters import \
    arena_radius, agent_radius, blade_radius

def get_random_directions(n: int)->Tensor:
    normals = torch.randn(n, 2)
    unit = F.normalize(normals,p=2,dim=1)
    return unit

def get_random_vectors(n: int, max_scale=1.0) ->Tensor:
    directions = get_random_directions(n)
    scales = max_scale*torch.rand(n).unsqueeze(1)
    return scales*directions

def clamp_vectors(vectors: Tensor, cap: float)->Tensor:
    magnitudes = torch.sqrt(torch.sum(vectors**2,dim=1,keepdim=True))
    return torch.where(magnitudes<cap,vectors,cap*vectors/magnitudes)

max_agent_speed = 125
max_blade_speed = 550

def get_random_states(n: int)->Tensor:
    a0v = get_random_vectors(n,max_agent_speed)*torch.rand(n,1)
    a0p = get_random_vectors(n,arena_radius-agent_radius)
    a1v = get_random_vectors(n,max_agent_speed)*torch.rand(n,1)
    a1p = get_random_vectors(n,arena_radius-agent_radius)
    b0v = get_random_vectors(n,max_blade_speed)*torch.rand(n,1)
    offset0 = get_random_vectors(n,arena_radius-blade_radius)*torch.rand(n,1)
    b0p = clamp_vectors(a0p+offset0,arena_radius-blade_radius)
    b1v = get_random_vectors(n,max_blade_speed)*torch.rand(n,1)
    offset1 = get_random_vectors(n,arena_radius-blade_radius)*torch.rand(n,1)
    b1p = clamp_vectors(a1p+offset1,arena_radius-blade_radius)
    return torch.cat((a0v,a0p,a1v,a1p,b0v,b0p,b1v,b1p),dim=1)


