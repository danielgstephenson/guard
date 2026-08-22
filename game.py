from collections import defaultdict
from requests import get
import torch
import arcade
from physics import get_next
from sample import get_random_states
from parameters import \
    window_size, time_step, arena_radius, agent_radius, blade_radius, \
    wall_color, floor_color, bot_color, player_color, bot_blade_color, \
    player_blade_color, guide_color, action_vectors, target_radius

class Game(arcade.Window):
    def __init__(self):
        super().__init__(window_size, window_size, 'game')
        arcade.set_background_color(wall_color)
        self.set_update_rate(time_step)
        self.camera = arcade.Camera2D()
        self.camera.zoom = 0.9
        self.camera.position = (0,0)
        self.state = get_random_states(1)
        self.pressed = defaultdict(lambda: False)
        self.paused = True

    def on_mouse_scroll(self, x: int, y: int, scroll_x: float, scroll_y: float):
       self.camera.zoom *= 1 + 0.1*scroll_y

    def on_draw(self):
        self.clear()
        self.camera.use()
        bot_x = self.state[0,2].item()
        bot_y = self.state[0,3].item()
        player_x = self.state[0,6].item()
        player_y = self.state[0,7].item()
        bot_blade_x = self.state[0,10].item()
        bot_blade_y = self.state[0,11].item()
        player_blade_x = self.state[0,14].item()
        player_blade_y = self.state[0,15].item()
        arcade.draw_circle_filled(0,0,arena_radius,floor_color)
        arcade.draw_line(0,arena_radius,0,-arena_radius,guide_color,0.2*agent_radius)
        arcade.draw_line(arena_radius,0,-arena_radius,0,guide_color,0.2*agent_radius)
        arcade.draw_circle_filled(0,0,target_radius,floor_color)
        arcade.draw_circle_outline(0,0,target_radius,guide_color,0.2*agent_radius)
        arcade.draw_line(bot_blade_x,bot_blade_y,bot_x,bot_y,bot_blade_color,0.1*agent_radius)
        arcade.draw_line(player_blade_x,player_blade_y,player_x,player_y,player_blade_color,0.1*agent_radius)
        arcade.draw_circle_filled(bot_blade_x,bot_blade_y,blade_radius,bot_blade_color)
        arcade.draw_circle_filled(player_blade_x,player_blade_y,blade_radius,player_blade_color)
        arcade.draw_circle_filled(bot_x,bot_y,agent_radius,bot_color)
        arcade.draw_circle_filled(player_x,player_y,agent_radius,player_color)

    def on_update(self, delta_time: float) -> bool | None:
        if self.paused: return
        action0 = torch.tensor([0])
        action1 = torch.tensor([self.get_user_action()])
        self.state = get_next(self.state,action0,action1)

    def on_key_press(self, symbol: int, modifiers: int):
        self.pressed[symbol] = True

    def on_key_release(self, symbol: int, modifiers: int):
        self.pressed[symbol] = False
        if symbol == arcade.key.SPACE:
            self.paused = not self.paused
        if symbol == arcade.key.ENTER:
            self.state = get_random_states(1)

    def get_user_action(self)->int:
        dx = 0.0
        dy = 0.0
        if self.pressed[arcade.key.W] or self.pressed[arcade.key.UP]:
            dy += 1
        if self.pressed[arcade.key.S] or self.pressed[arcade.key.DOWN]:
            dy -= 1
        if self.pressed[arcade.key.A] or self.pressed[arcade.key.LEFT]:
            dx -= 1
        if self.pressed[arcade.key.D] or self.pressed[arcade.key.RIGHT]:
            dx += 1
        action = 0
        if dx != 0.0 or dy != 0.0:
            vector = torch.tensor([dx,dy])
            dots = torch.einsum('ij,j->i',action_vectors, vector)
            action = int(torch.argmax(dots).item())
        return action
        
game = Game()
game.run()