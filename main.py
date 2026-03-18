import pygame
import random
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO

# ─── Constants ──────────────────────────────────────────────────────────────
WIDTH = 288
HEIGHT = 512
BIRD_X = 60
GROUND_Y = 450
PIPE_WIDTH = 52
PIPE_GAP = 130
PIPE_SPEED = 2.0
GRAVITY = 0.4
JUMP = -8.0
PIPE_SPACING = 180

class FlappyBirdEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 30}

    def __init__(self, render_mode=None):
        super().__init__()
        self.render_mode = render_mode

        self.action_space = spaces.Discrete(2)

        self.observation_space = spaces.Box(
            low=np.array([0.0, -20.0, -100.0, 0.0], dtype=np.float32),
            high=np.array([HEIGHT, 20.0, WIDTH + 200.0, HEIGHT], dtype=np.float32),
            shape=(4,),
            dtype=np.float32,
        )

        self.screen = None
        self.clock = None
        if self.render_mode == "human":
            pygame.init()
            self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
            pygame.display.set_caption("Flappy Bird - Live Training")
            self.clock = pygame.time.Clock()

        self.bird_y = 0.0
        self.bird_vel = 0.0
        self.pipes = []
        self.score = 0

    def _add_new_pipe(self):
        if self.pipes:
            new_x = self.pipes[-1]["x"] + PIPE_SPACING
        else:
            new_x = WIDTH + 100
        top = random.randint(60, HEIGHT - PIPE_GAP - 120)
        self.pipes.append({"x": new_x, "top": top, "passed": False})

    def _get_next_pipe_info(self):
        for pipe in self.pipes:
            if pipe["x"] + PIPE_WIDTH > BIRD_X:
                return pipe["x"] - BIRD_X, pipe["top"] + PIPE_GAP // 2
        return 300.0, HEIGHT / 2

    def _get_observation(self):
        dist, gap_center = self._get_next_pipe_info()
        return np.array([self.bird_y, self.bird_vel, dist, gap_center], dtype=np.float32)

    def _check_collision(self):
        bird_rect = pygame.Rect(BIRD_X - 12, int(self.bird_y) - 12, 24, 24)
        if self.bird_y - 12 < 0 or self.bird_y + 12 > GROUND_Y:
            return True
        for pipe in self.pipes:
            top_rect = pygame.Rect(pipe["x"], 0, PIPE_WIDTH, pipe["top"])
            bottom_y = pipe["top"] + PIPE_GAP
            bottom_rect = pygame.Rect(pipe["x"], bottom_y, PIPE_WIDTH, HEIGHT - bottom_y)
            if bird_rect.colliderect(top_rect) or bird_rect.colliderect(bottom_rect):
                return True
        return False

    def reset(self, seed=None, options=None):
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        self.bird_y = HEIGHT // 2
        self.bird_vel = 0.0
        self.pipes = []
        self.score = 0
        self._add_new_pipe()
        return self._get_observation(), {}

    def step(self, action):
        if action == 1:
            self.bird_vel = JUMP
        self.bird_vel += GRAVITY
        self.bird_y += self.bird_vel

        for pipe in self.pipes:
            pipe["x"] -= PIPE_SPEED

        if self.pipes and self.pipes[0]["x"] + PIPE_WIDTH < 0:
            self.pipes.pop(0)

        if not self.pipes or self.pipes[-1]["x"] < WIDTH - PIPE_SPACING:
            self._add_new_pipe()

        reward = -0.01
        for pipe in self.pipes:
            if not pipe["passed"] and BIRD_X > pipe["x"] + PIPE_WIDTH:
                reward += 1.0
                pipe["passed"] = True
                self.score += 1

        terminated = self._check_collision()
        if terminated:
            reward = -100.0

        obs = self._get_observation()
        truncated = False
        info = {"score": self.score}

        if self.render_mode == "human":
            self.render()

        return obs, reward, terminated, truncated, info

    def render(self):
        if self.screen is None:
            return
        pygame.event.pump()

        self.screen.fill((135, 206, 235))
        for pipe in self.pipes:
            pygame.draw.rect(self.screen, (0, 128, 0), (pipe["x"], 0, PIPE_WIDTH, pipe["top"]))
            bottom_y = pipe["top"] + PIPE_GAP
            pygame.draw.rect(self.screen, (0, 128, 0), (pipe["x"], bottom_y, PIPE_WIDTH, HEIGHT - bottom_y))

        pygame.draw.circle(self.screen, (255, 255, 0), (BIRD_X, int(self.bird_y)), 12)
        pygame.draw.rect(self.screen, (139, 69, 19), (0, GROUND_Y, WIDTH, HEIGHT - GROUND_Y))

        font = pygame.font.SysFont(None, 36)
        text = font.render(f"Score: {self.score}", True, (255, 255, 255))
        self.screen.blit(text, (10, 10))

        pygame.display.flip()
        # self.clock.tick(30)

    def close(self):
        if self.screen is not None:
            pygame.quit()


# ─── Live Training + Watching ───────────────────────────────────────────────
if __name__ == "__main__":
    env = FlappyBirdEnv(render_mode="human")
    print("Window should open now. Training starts soon...")

    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        verbose=1,
        device="cpu",           # change to "cuda" if you have GPU + want to try
    )

    print("\nTraining in progress — watch the bird improve over time")
    print("  • Press ESC or close window to stop early\n")

    total_steps = 0
    target_steps = 400_000   # ← change this number if you want longer/shorter training

    try:
        while total_steps < target_steps:
            model.learn(total_timesteps=2048, reset_num_timesteps=False)
            total_steps += 2048

            # Short demo period with current policy (makes progress visible)
            obs, _ = env.reset()
            demo_steps = 0
            done = False
            while not done and demo_steps < 1200:   # ~40 seconds at 30 fps
                demo_steps += 1
                if demo_steps % 60 == 0:  # every ~2 seconds
                     print(f"  [demo] step {demo_steps} | bird y={env.bird_y:.0f} vel={env.bird_vel:.1f} | score={env.score}")
                action, _ = model.predict(obs, deterministic=False)  # stochastic = more variety
                obs, r, term, trunc, info = env.step(action)
                done = term or trunc

                # Allow graceful exit during demo
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        raise KeyboardInterrupt
                    if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                        raise KeyboardInterrupt

            print(f"  Steps: {total_steps:>7,}   |   Score this demo: {info.get('score', 0)}")

    except KeyboardInterrupt:
        print("\nTraining stopped by user")

    except Exception as e:
        print(f"Error during training: {e}")

    finally:
        print(f"\nTraining ended at {total_steps} steps")
        print("You can close the window now.")
        env.close()