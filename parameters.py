import torch
import colorsys
from arcade.types import Color
from math import cos, pi, sin

def hsl_to_arcade_color(h: float, s: float, l: float, a=1.0)->Color:
    h_norm = h / 360.0
    s_norm = s / 100.0
    l_norm = l / 100.0
    r_f, g_f, b_f = colorsys.hls_to_rgb(h_norm, l_norm, s_norm)
    return Color(int(r_f*255), int(g_f*255), int(b_f*255), int(a*255))

wall_color = hsl_to_arcade_color(0, 0, 10)
guide_color = hsl_to_arcade_color(0, 0, 10)
player_color = hsl_to_arcade_color(220, 100, 40)
bot_color = hsl_to_arcade_color(120, 100, 27)
player_blade_color = hsl_to_arcade_color(195, 100, 50)
bot_blade_color = hsl_to_arcade_color(140, 100, 45)
floor_color = (0,0,0,255)

window_size = 1000
time_step = 0.02

arena_radius = 500
agent_radius = 15
target_radius = 40
blade_radius = 25

agent_drag = 0.4
blade_drag = 0.1
spring_power = 2
move_power = 50

action_vector_list = [[0.0,0.0]]
for i in range(8):
    angle = 2 * pi * i / 8
    vision_dir = [cos(angle), sin(angle)]
    action_vector_list.append(vision_dir)
action_vectors = torch.tensor(action_vector_list,dtype=torch.float)
actions = torch.tensor([i for i in range(9)])

action_count = len(action_vector_list)
state_size = 16