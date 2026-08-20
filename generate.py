import torch
from torch import Tensor
from physics import step
from value import ValueModel
from parameters import actions, action_count, state_size


def get_choice0(state: Tensor, model: ValueModel, precision: float)->Tensor:
    n = state.shape[0]
    actions0 = actions.repeat(n)
    actions1 = 0*actions0
    outcomes = torch.repeat_interleave(state,repeats=action_count,dim=0)
    outcomes = step(outcomes,actions0,actions1)
    values = model(outcomes)
    action_values = torch.reshape(values,shape=(n,action_count))
    probs = torch.softmax(precision*action_values,dim=1)
    choice0 = torch.multinomial(probs,num_samples=1).reshape(n)
    return choice0

def get_choice1(state: Tensor, model: ValueModel, precision: float)->Tensor:
    n = state.shape[0]
    actions1 = actions.repeat(n)
    actions0 = 0*actions1
    outcomes = torch.repeat_interleave(state,repeats=action_count,dim=0)
    outcomes = step(outcomes,actions0,actions1)
    values = model(outcomes)
    action_values = torch.reshape(values,shape=(n,action_count))
    probs = torch.softmax(-precision*action_values,dim=1)
    choice0 = torch.multinomial(probs,num_samples=1).reshape(n)
    return choice0

def get_reward(state: Tensor)->Tensor:
    a1pos = state[:,6:8]
    return torch.sqrt(torch.sum(a1pos**2,dim=1))

model = ValueModel()
state = torch.rand(100,state_size)
choice0 = get_choice0(state,model,2)
reward = get_reward(state)
print(reward.shape)
print(reward)
                                       
