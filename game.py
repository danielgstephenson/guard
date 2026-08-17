import torch
from torch import Tensor
import arcade
from parameters import \
    window_size, time_step, arena_radius, agent_radius, blade_radius, \
    wall_color, floor_color, bot_color, player_color, bot_blade_color, player_blade_color

class Game(arcade.Window):
    def __init__(self):
        super().__init__(window_size, window_size, 'game')
        arcade.set_background_color(wall_color)
        self.set_update_rate(time_step)
        self.camera = arcade.Camera2D()
        self.camera.zoom = 0.9
        self.camera.position = (0,0)
        self.state = torch.tensor([
            0,0,+100,0,
            0,0,-100,0,
            0,0,+200,0,
            0,0,-200,0
        ],dtype=torch.float)

    def on_mouse_scroll(self, x: int, y: int, scroll_x: float, scroll_y: float):
       self.camera.zoom *= 1 + 0.1*scroll_y
       print('zoom',self.camera.zoom)

    def on_draw(self):
        self.clear()
        self.camera.use()
        bot_x = self.state[2].item()
        bot_y = self.state[3].item()
        player_x = self.state[6].item()
        player_y = self.state[7].item()
        bot_blade_x = self.state[10].item()
        bot_blade_y = self.state[11].item()
        player_blade_x = self.state[14].item()
        player_blade_y = self.state[15].item()
        arcade.draw_circle_filled(0,0,arena_radius,floor_color)
        arcade.draw_line(bot_blade_x,bot_blade_y,bot_x,bot_y,bot_blade_color,0.1*agent_radius)
        arcade.draw_line(player_blade_x,player_blade_y,player_x,player_y,player_blade_color,0.1*agent_radius)
        arcade.draw_circle_filled(bot_blade_x,bot_blade_y,blade_radius,bot_blade_color)
        arcade.draw_circle_filled(player_blade_x,player_blade_y,blade_radius,player_blade_color)
        arcade.draw_circle_filled(bot_x,bot_y,agent_radius,bot_color)
        arcade.draw_circle_filled(player_x,player_y,agent_radius,player_color)
        

game = Game()
game.run()