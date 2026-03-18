# Windy Courier RL

Windy Courier is a one-day reinforcement learning project built to be demo-friendly in an interview: a courier drone learns to pick up a package and deliver it across a windy 2D map while avoiding a moving obstacle. The project is intentionally small enough to complete in a day, but polished enough to show visible learning through videos, GIFs, plots, and a free static showcase page.

## Why This Is A Good RL Demo

This environment simulates a courier drone navigating uncertain dynamics. The drone cannot rely on a fixed shortest path because wind zones push it off course and a moving obstacle changes the safe route over time. That makes it a natural RL problem: the agent must learn a sequential control policy that balances short-term movement against long-term delivery success.

RL is a better fit than supervised learning here because there is no single correct action label for every state. Good behavior depends on future consequences, exploration, and delayed reward. The agent has to discover a strategy, not imitate a provided path.

## What Learning Looks Like

Early in training, the drone drifts into wind corridors, collides with the moving obstacle, or runs out of time before completing delivery. As PPO improves the policy, behavior becomes more structured: the agent learns to stabilize near the package, use calm zones to reposition, and approach the delivery target with less wasteful motion.

The saved artifacts make this visible:
- baseline random-policy video
- 25%, 50%, and 75% progress videos during training
- final trained-policy video
- one combined learning-progression GIF
- reward and rolling success-rate plots

## Environment Design

### State / Observation
The observation vector contains:
- agent position `(x, y)`
- agent velocity `(vx, vy)`
- goal position
- package picked flag
- package position
- moving obstacle position
- local wind vector at the agent location

### Action Space
The action space is discrete with 5 actions:
- stay
- thrust up
- thrust down
- thrust left
- thrust right

This keeps training robust and fast enough for a one-day build while still showing interesting control behavior.

### Rewards
- positive reward for package pickup
- larger positive reward for successful delivery
- shaping reward for reducing distance to the current target
- small per-step penalty to encourage efficiency
- penalties for collision and out-of-bounds failure

### Episode Ends When
- the package is delivered successfully
- the drone collides with the obstacle
- the drone goes out of bounds
- max steps are reached

## Repo Structure

```text
windy-courier-rl/
  README.md
  requirements.txt
  .gitignore
  LICENSE
  src/
    envs/
      windy_courier_env.py
      __init__.py
    train.py
    evaluate.py
    record_progress.py
    utils.py
  assets/
    videos/
    gifs/
    plots/
    checkpoints/
    final_model/
  web/
    index.html
    style.css
    app.js
  huggingface_space/
    app.py
    requirements.txt
```

## Install

```bash
pip install -r requirements.txt
```

## Commands That Must Work

Train:

```bash
python -m src.train --total-timesteps 300000 --save-dir assets
```

Evaluate and record:

```bash
python -m src.evaluate --model-path assets/final_model/model.zip --record
```

Generate plots and GIF:

```bash
python -m src.record_progress --log-dir assets --out-dir assets
```

## Practical Training Notes

Recommended local run:
- `300000` timesteps
- 8 vectorized environments
- target runtime: roughly 1 to 4 hours depending on CPU/GPU setup and Python environment

Artifacts are saved automatically to:
- `assets/videos/`
- `assets/gifs/`
- `assets/plots/`
- `assets/checkpoints/`
- `assets/final_model/`

## What To Expect In Assets

After training and evaluation, you should have:
- `assets/final_model/model.zip`
- `assets/logs/metrics.csv`
- `assets/videos/baseline_random.mp4`
- `assets/videos/training_25.mp4`
- `assets/videos/training_50.mp4`
- `assets/videos/training_75.mp4`
- `assets/videos/trained_final.mp4`
- `assets/videos/eval_episode_1.mp4`
- `assets/gifs/learning_progression.gif`
- `assets/plots/reward_curve.png`
- `assets/plots/success_rate_curve.png`
- `assets/plots/architecture_diagram.png`

The `record_progress.py` script also copies the main media outputs into `web/media/` so the GitHub Pages site can display the latest results.

## How To Host On GitHub Pages

1. Push the repo to GitHub.
2. Run `python -m src.record_progress --log-dir assets --out-dir assets` so `web/media/` is populated.
3. In the GitHub repo, open `Settings -> Pages`.
4. Under `Build and deployment`, choose `Deploy from a branch`.
5. Select your main branch and the `/web` folder.
6. Save.
7. Your showcase page will be published from `web/index.html`.

If you prefer `gh-pages`, copy the `web/` folder contents into that branch root instead.

## Optional Hugging Face Space

The included Gradio app is inference-only and CPU-friendly.

Deploy steps:
1. Create a new Gradio Space on Hugging Face.
2. Upload `huggingface_space/app.py`, `huggingface_space/requirements.txt`, and the trained model in `assets/final_model/model.zip`.
3. Keep the repo structure or update the model path inside `app.py`.
4. Launch the Space.

The app renders one evaluation episode and returns an MP4 plus summary stats.

## Failure Modes And Next Improvements

Current failure modes:
- the drone can still over-commit inside strong wind corridors
- moving obstacle timing may cause occasional last-second collisions
- policy quality is sensitive to reward shaping and episode length

Natural next improvements:
- curriculum training with simpler wind maps first
- continuous control instead of discrete thrust
- multiple delivery tasks per episode
- domain randomization over wind layouts
- richer observation encoding such as obstacle velocity and local occupancy

## Talking Points

### 30-Second Pitch
I built a custom RL environment called Windy Courier where a drone learns to pick up and deliver a package across a 2D map with wind drift and a moving obstacle. I trained a PPO agent, automatically saved videos throughout training, and turned the outputs into a GitHub Pages portfolio so learning is visually obvious even without running code live.

### 2-Minute Technical Deep Dive
The environment is a custom Gymnasium task with a compact observation vector containing position, velocity, package state, obstacle position, and local wind. I used a discrete action space to keep the project stable and trainable within one day. PPO works well here because the agent has to learn a sequential policy under delayed reward and nontrivial dynamics. I added distance-based shaping so the agent first learns to reach the package and then transition to delivery behavior. During training, a callback records milestone videos and checkpoints, while evaluation and post-processing scripts generate a final demo video, a progression GIF, and reward and success plots. That makes it easy to discuss both the RL design and the engineering polish.

## Interview Narrative

### Why RL
This problem involves sequential decision-making under uncertainty, delayed outcomes, and interaction with the environment. A static labeled dataset would not capture the consequences of actions under different wind conditions and obstacle timings.

### How Learning Emerges
The policy starts with exploration and mostly chaotic drift. Then it learns a useful sub-goal structure: stabilize, acquire package, route around danger, and deliver. You can point to the saved milestone videos to show each stage.

### What Problem This Simulates
It loosely simulates a last-mile autonomous drone that must operate under environmental disturbances and dynamic hazards while trying to complete a mission efficiently.

## Notes On Reproducibility
- all scripts expose a `--seed` argument
- training, evaluation, and recordings use deterministic paths where appropriate
- outputs are written to predictable asset directories

## License
MIT
