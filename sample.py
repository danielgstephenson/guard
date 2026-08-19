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

def get_random_scales(n: int)->Tensor:
    return 1 - torch.sqrt(torch.rand(n))

max_agent_speed = 125
max_blade_speed = 550

def get_random_states(n: int)->Tensor:
    a0v = get_random_vectors(n,max_agent_speed)
    a0p = get_random_vectors(n,arena_radius-agent_radius)
    a1v = get_random_vectors(n,max_agent_speed)
    a1p = get_random_vectors(n,arena_radius-agent_radius)
    b0v = get_random_vectors(n,max_blade_speed)
    b0p = get_random_vectors(n,arena_radius-blade_radius)
    b1v = get_random_vectors(n,max_blade_speed)
    b1p = get_random_vectors(n,arena_radius-blade_radius)
    return torch.cat((a0v,a0p,a1v,a1p,b0v,b0p,b1v,b1p),dim=1)


