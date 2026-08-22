import torch
from torch import Tensor
from physics import check_hit_pair, get_next
from sample import get_random_states
from value import ValueModel
from parameters import actions, action_count, state_size, time_step, arena_radius


def get_choice0(state: Tensor, model: ValueModel, precision: float)->Tensor:
    with torch.no_grad():
        n = state.shape[0]
        actions0 = actions.repeat(n)
        actions1 = 0*actions0
        outcomes = torch.repeat_interleave(state,repeats=action_count,dim=0)
        outcomes = get_next(outcomes,actions0,actions1)
        values = model(outcomes)
        action_values = torch.reshape(values,shape=(n,action_count))
        probs = torch.softmax(precision*action_values,dim=1)
        choice0 = torch.multinomial(probs,num_samples=1).reshape(n)
        return choice0

def get_choice1(state: Tensor, model: ValueModel, precision: float)->Tensor:
    with torch.no_grad():
        n = state.shape[0]
        actions1 = actions.repeat(n)
        actions0 = 0*actions1
        outcomes = torch.repeat_interleave(state,repeats=action_count,dim=0)
        outcomes = get_next(outcomes,actions0,actions1)
        values = model(outcomes)
        action_values = torch.reshape(values,shape=(n,action_count))
        probs = torch.softmax(-precision*action_values,dim=1)
        choice0 = torch.multinomial(probs,num_samples=1).reshape(n)
        return choice0

def get_reward(state: Tensor)->Tensor:
    agent0 = state[:,0:4]
    agent1 = state[:,4:8]
    blade0 = state[:,8:12]
    blade1 = state[:,12:16]
    a1pos = agent1[:,2:4]
    a1dist = torch.sqrt(torch.sum(a1pos**2,dim=1,keepdim=True))
    hit0 = check_hit_pair(agent0,blade1)
    hit1 = check_hit_pair(agent1,blade0)
    return torch.where(hit1, 2*arena_radius, torch.where(hit0, -arena_radius, a1dist))

def advance(state: Tensor, model: ValueModel, precision:float)->Tensor:
    action0 = get_choice0(state,model,precision)
    action1 = get_choice1(state,model,precision)
    return get_next(state, action0, action1)

def generate(model: ValueModel, n: int, step_count: int, horizon: float, precision: float)->tuple[Tensor,Tensor]:
    start = get_random_states(n)
    end_prob = min(time_step/horizon,1.0) if horizon > 0 else 1.0
    state = start.clone()
    value = torch.zeros(n,1).float()
    for step in range(step_count):
        value += (1-end_prob)**step * end_prob * get_reward(state)
        state = advance(state, model, precision)
    value += (1-end_prob)**step_count * get_reward(state)
    return start, value

                                       
