import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pygame
import sys

class BarefootCoachEnv(gym.Env):
    """
    Custom Environment that simulates a runner falling from 2 meters.
    The agent chooses leg stiffness (0 = jelly, 1 = stiff) to land softly.
    """
    metadata = {"render_modes": ["human"], "render_fps": 30}

    def __init__(self, render_mode=None):
        super(BarefootCoachEnv, self).__init__()

        # Action: leg stiffness [0, 1]
        self.action_space = spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32)

        # Observation: [height (m), vertical velocity (m/s)]
        self.observation_space = spaces.Box(
            low=np.array([-1.0, -20.0], dtype=np.float32),
            high=np.array([10.0, 20.0], dtype=np.float32),
            dtype=np.float32
        )

        # State
        self.state = None

        # Rendering
        self.render_mode = render_mode
        if self.render_mode == "human":
            pygame.init()
            self.screen = pygame.display.set_mode((400, 600))
            pygame.display.set_caption("Barefoot Coach")
            self.clock = pygame.time.Clock()
            self.font = pygame.font.SysFont(None, 24)

    def step(self, action):
        height, velocity = self.state
        stiffness = action[0]

        # Physics: gravity
        gravity = -9.81
        time_step = 0.05

        new_velocity = velocity + (gravity * time_step)
        new_height = height + (new_velocity * time_step)

        # Reward logic
        reward = 0.0
        terminated = False
        truncated = False
        info = {}

        if new_height <= 0:          # runner hits ground
            new_height = 0.0
            impact_force = abs(new_velocity) * (stiffness + 0.1)
            if impact_force > 5.0:
                reward = -impact_force * 10.0   # hard landing penalty
            else:
                reward = 50.0                    # perfect soft landing
            terminated = True
        else:
            reward = -0.1             # small penalty while in air

        self.state = np.array([new_height, new_velocity], dtype=np.float32)

        # Optional rendering
        if self.render_mode == "human":
            self.render()

        return self.state, float(reward), terminated, truncated, info

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        # Start 2 m above ground with zero velocity
        self.state = np.array([2.0, 0.0], dtype=np.float32)

        if self.render_mode == "human":
            self.render()   # show initial state

        return self.state, {}

    def render(self):
        if self.render_mode == "human":
            # Handle Pygame window close
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()

            height = self.state[0]

            # Clear screen
            self.screen.fill((255, 255, 255))

            # Draw ground (brown line)
            ground_y = 550
            pygame.draw.line(self.screen, (139, 69, 19), (0, ground_y), (400, ground_y), 5)

            # Draw runner (red circle)
            # Scale: 1 m = 50 pixels; higher height → lower y (because y=0 at top)
            runner_y = ground_y - int(height * 50)
            pygame.draw.circle(self.screen, (255, 0, 0), (200, runner_y), 15)

            # Display height text
            text = self.font.render(f"Height: {height:.2f} m", True, (0, 0, 0))
            self.screen.blit(text, (10, 10))

            pygame.display.flip()
            self.clock.tick(self.metadata["render_fps"])

    def close(self):
        if self.render_mode == "human":
            pygame.quit()