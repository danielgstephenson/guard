import torch
from torch import Tensor
import torch.nn.functional as F
from parameters import \
    time_step, move_power, action_vectors, spring_power, \
    agent_drag, blade_drag, agent_radius, blade_radius, arena_radius

def integrate_entity(entity: Tensor, force: Tensor, drag: float)->Tensor:
    velocity = entity[:,0:2]
    position = entity[:,2:4]
    velocity = (1-drag*time_step)*velocity
    velocity += time_step*force
    position += time_step*velocity
    return torch.cat((velocity,position),1)

def integrate(state:Tensor, action0: Tensor, action1: Tensor)->Tensor:
    agent0 = state[:,0:4]
    agent1 = state[:,4:8]
    blade0 = state[:,8:12]
    blade1 = state[:,12:16]
    move0 = move_power*action_vectors[action0,:]
    move1 = move_power*action_vectors[action1,:]
    a0pos = agent0[:,2:4]
    a1pos = agent1[:,2:4]
    b0pos = blade0[:,2:4]
    b1pos = blade1[:,2:4]
    tension0 = spring_power*(a0pos-b0pos)
    tension1 = spring_power*(a1pos-b1pos)
    agent0 = integrate_entity(agent0,move0,agent_drag)
    agent1 = integrate_entity(agent1,move1,agent_drag)
    blade0 = integrate_entity(blade0,tension0,blade_drag)
    blade1 = integrate_entity(blade1,tension1,blade_drag)
    return torch.cat((agent0,agent1,blade0,blade1),1)

def collide_pair(entity0: Tensor, entity1, radius0: float, radius1: float)->tuple[Tensor,Tensor]:
    velocity0 = entity0[:,0:2]
    position0 = entity0[:,2:4]
    velocity1 = entity1[:,0:2]
    position1 = entity1[:,2:4]
    min_dist = radius0 + radius1
    vector = position1-position0
    dist = torch.sqrt(torch.sum(vector**2,dim=1,keepdim=True))
    overlap = min_dist - dist
    normal = torch.where(dist*overlap>0,vector/dist,0)
    relative_velocity = velocity0 - velocity1
    dot = torch.einsum('ij,ij->i',relative_velocity,normal).unsqueeze(1)
    impact_speed = torch.where(dot>0, dot, 0)
    impulse = 0.5*impact_speed*normal
    shift = 0.5*overlap*normal
    velocity0 -= impulse
    velocity1 += impulse
    position0 -= shift
    position1 += shift
    entity0 = torch.cat((velocity0,position0),1)
    entity1 = torch.cat((velocity1,position1),1)
    return entity0, entity1

def collide_boundary(entity: Tensor, radius: float)->Tensor:
    velocity = entity[:,0:2]
    position = entity[:,2:4]
    max_dist = arena_radius - radius
    dist = torch.sqrt(torch.sum(position**2,dim=1,keepdim=True))
    overlap = dist - max_dist
    normal = torch.where(dist>0,position/dist,0)
    position = torch.where(overlap>0,max_dist*normal,position)
    dotVelPos = torch.sum(velocity*position,dim=1,keepdim=True)
    dotPosPos = torch.sum(position*position,dim=1,keepdim=True)
    impact = torch.where(dotPosPos==0,0,position*dotVelPos/dotPosPos)
    velocity = torch.where(overlap>0,velocity-2*impact,velocity)
    return torch.cat((velocity,position),1)

def resolve(state:Tensor):
    agent0 = state[:,0:4]
    agent1 = state[:,4:8]
    blade0 = state[:,8:12]
    blade1 = state[:,12:16]
    agent0, agent1 = collide_pair(agent0,agent1,agent_radius,agent_radius)
    blade0, blade1 = collide_pair(blade0,blade1,blade_radius,blade_radius)
    agent0 = collide_boundary(agent0,agent_radius)
    agent1 = collide_boundary(agent1,agent_radius)
    blade0 = collide_boundary(blade0,blade_radius)
    blade1 = collide_boundary(blade1,blade_radius)
    return torch.cat((agent0,agent1,blade0,blade1),1)

def get_respawn(agent: Tensor)->Tensor:
    position = agent[:,2:4]
    position += 0.0001*(torch.rand_like(position)-0.5)
    position = -(arena_radius-agent_radius)*F.normalize(position,p=2,dim=1)
    velocity = 0*position
    return torch.cat((velocity,position),1)

def check_hit(agent: Tensor, blade: Tensor)->Tensor:
    agent_position = agent[:,2:4]
    blade_position = blade[:,2:4]
    vector = agent_position-blade_position
    dist = torch.sqrt(torch.sum(vector**2,dim=1,keepdim=True))
    min_dist = agent_radius + blade_radius
    return dist < min_dist

def strike(state: Tensor)->Tensor:
    agent0 = state[:,0:4]
    agent1 = state[:,4:8]
    blade0 = state[:,8:12]
    blade1 = state[:,12:16]
    hit0 = check_hit(agent0,blade1)
    hit1 = check_hit(agent1,blade0)
    respawn0 = get_respawn(agent0)
    respawn1 = get_respawn(agent1)
    agent0 = torch.where(hit0, respawn0, agent0)
    blade0 = torch.where(hit0, respawn0, blade0)
    agent1 = torch.where(hit1, respawn1, agent1)
    blade1 = torch.where(hit1, respawn1, blade1)
    return torch.cat((agent0,agent1,blade0,blade1),1)

def get_next(state:Tensor, action0: Tensor, action1: Tensor)->Tensor:
    state = integrate(state,action0,action1)
    state = resolve(state)
    state = strike(state)
    return state
