import colorsys
from arcade.types import Color

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

time_step = 0.02

arena_radius = 500
agent_radius = 15
target_radius = 30
blade_radius = 25

window_size = 1000
