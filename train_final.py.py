from stable_baselines3 import PPO

# Train (no rendering = super fast)
env = FlappyBirdEnv(render_mode=None)
model = PPO("MlpPolicy", env, verbose=1, learning_rate=0.0003)
model.learn(total_timesteps=200_000)   # ~5-10 minutes on a normal laptop
model.save("flappy_ppo")

# Watch your trained agent!
env = FlappyBirdEnv(render_mode="human")
obs, _ = env.reset()
done = False
while not done:
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            done = True
env.close()