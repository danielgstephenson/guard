import torch
from torch import Tensor
from physics import get_next
from sample import get_random_states
from value import ValueModel
from parameters import actions, action_count, time_step
from reward import get_reward


def get_choice0(state: Tensor, model: ValueModel, precision: float)->Tensor:
    with torch.no_grad():
        n = state.shape[0]
        actions0 = actions.repeat(n)
        actions1 = 0*actions0
        outcomes = torch.repeat_interleave(state,repeats=action_count,dim=0)
        outcomes = get_next(outcomes,actions0,actions1)
        values = model(outcomes)
        action_values = torch.reshape(values,shape=(n,action_count))
        probs = torch.softmax(precision*action_values/time_step,dim=1)
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
        probs = torch.softmax(-precision*action_values/time_step,dim=1)
        choice0 = torch.multinomial(probs,num_samples=1).reshape(n)
        return choice0

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
    with torch.no_grad():
        value += (1-end_prob)**step_count * model(state)
    return start, value

                                       
