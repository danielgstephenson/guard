import torch
from torch import Tensor
from physics import check_hit_pair
from parameters import arena_radius, hit0_value, hit1_value

def get_reward(state: Tensor)->Tensor:
    agent0 = state[:,0:4]
    agent1 = state[:,4:8]
    blade0 = state[:,8:12]
    blade1 = state[:,12:16]
    a1pos = agent1[:,2:4]
    a1dist = torch.sqrt(torch.sum(a1pos**2,dim=1,keepdim=True))
    hit0 = check_hit_pair(agent0,blade1)
    hit1 = check_hit_pair(agent1,blade0)
    return torch.where(hit1, hit1_value, torch.where(hit0, hit0_value, a1dist / arena_radius))