from barefoot_env import BarefootCoachEnv
from stable_baselines3 import PPO
import pygame
# ---------------------------
# 1. TRAINING (no graphics)
# ---------------------------
print("Initializing environment for training...")
env = BarefootCoachEnv(render_mode=None)   # render_mode=None → no window

print("Creating PPO model...")
model = PPO("MlpPolicy", env, verbose=1, device="cpu")

print("Starting training (20,000 timesteps)...")
model.learn(total_timesteps=20000)

# Save the trained model
model.save("barefoot_coach_model")
print("Model saved as 'barefoot_coach_model.zip'")

# Close the training environment
env.close()

# ---------------------------
# 2. TESTING WITH VISUALISATION
# ---------------------------
print("\n--- Testing the trained AI with graphics ---")
env = BarefootCoachEnv(render_mode="human")   # now with graphics

obs, info = env.reset()
terminated = False
total_reward = 0

while not terminated:
    # Let the trained model decide the leg stiffness
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = env.step(action)
    total_reward += reward

    # Small delay so the animation is watchable (render is already called inside step)
    pygame.time.wait(50)   # optional extra delay

print(f"\nFinal Test Episode Reward: {total_reward:.2f}")
if total_reward > 0:
    print("Success! The agent learned a soft landing.")
else:
    print("Ouch! The agent landed too hard. (Try training for more timesteps)")

# Clean up
env.close()