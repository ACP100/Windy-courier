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

## Live Play / Demo Mode

After training, run `python -m src.play` to open the pygame render:

```bash
python -m src.play --mode agent --model-path assets/final_model/model.zip --deterministic
python -m src.play --mode human
```

Human mode keeps moving even when you release keys, so hold `WASD`/arrows to steer, `Space` to brake, and `Esc` to quit. The script now counts drops and score, and hitting the outer walls automatically restarts you from spawn for a new delivery attempt (`--auto-restart-on-wall` is on by default). The agent mode also reports score/delivery counts every time the policy succeeds while keeping the cycling demo running.

## Reward & Penalty Tuning

All reward terms are named constructor arguments in `src/envs/windy_courier_env.py`, which makes the shaping easy to adjust:

- `step_penalty` (default -0.04) – encourages speed.
- `pickup_reward` (default +8.0) – positive reward for grabbing the package.
- `delivery_reward` (default +30.0) – reward for successful delivery.
- `collision_penalty` (default -18.0) – punishes obstacle hits.
- `out_of_bounds_penalty` (default -20.0) – punishes wall crossings.
- `distance_reward_scale` (default 1.1) – shapes movement toward the current target.

If you change any of these values, retrain with `python -m src.train --save-dir assets` and rerun `src.record_progress` so the metrics/plotted artifacts reflect the new tuning. The live `play.py` demo uses `terminate_on_delivery=False` so you can keep stacking deliveries and watching the score grow even after a drop.

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


