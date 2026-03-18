from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import imageio.v2 as imageio
import matplotlib
import numpy as np
import pandas as pd
import torch

matplotlib.use('Agg')
import matplotlib.pyplot as plt

from src.envs import WindyCourierEnv


@dataclass
class EpisodeResult:
    reward: float
    steps: int
    success: bool
    collision: bool
    out_of_bounds: bool


def ensure_asset_dirs(save_dir: Path) -> dict[str, Path]:
    paths = {
        'root': save_dir,
        'videos': save_dir / 'videos',
        'gifs': save_dir / 'gifs',
        'plots': save_dir / 'plots',
        'checkpoints': save_dir / 'checkpoints',
        'final_model': save_dir / 'final_model',
        'logs': save_dir / 'logs',
    }
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    return paths


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_env(
    render_mode: str | None = None,
    seed: int = 0,
    max_steps: int = 180,
    terminate_on_delivery: bool = True,
) -> WindyCourierEnv:
    return WindyCourierEnv(
        render_mode=render_mode,
        seed=seed,
        max_steps=max_steps,
        terminate_on_delivery=terminate_on_delivery,
    )


def record_episode_video(
    model,
    save_path: Path,
    seed: int,
    deterministic: bool = True,
    max_steps: int = 180,
    fps: int = 20,
) -> EpisodeResult:
    env = make_env(render_mode='rgb_array', seed=seed, max_steps=max_steps)
    obs, _ = env.reset(seed=seed)
    frames: list[np.ndarray] = []
    total_reward = 0.0
    info = {}
    steps = 0

    while True:
        frame = env.render()
        if frame is not None:
            frames.append(frame)

        if model is None:
            action = env.action_space.sample()
        else:
            action, _ = model.predict(obs, deterministic=deterministic)

        obs, reward, terminated, truncated, info = env.step(int(action))
        total_reward += reward
        steps += 1
        if terminated or truncated:
            final_frame = env.render()
            if final_frame is not None:
                frames.append(final_frame)
            break

    env.close()
    if not frames:
        raise RuntimeError('No frames were collected for the video.')

    imageio.mimsave(save_path, frames, fps=fps, macro_block_size=None)
    return EpisodeResult(
        reward=float(total_reward),
        steps=steps,
        success=bool(info.get('is_success', False)),
        collision=bool(info.get('collision', False)),
        out_of_bounds=bool(info.get('out_of_bounds', False)),
    )


def load_metrics(log_path: Path) -> pd.DataFrame:
    if not log_path.exists():
        raise FileNotFoundError(f'Metrics log not found: {log_path}')
    return pd.read_csv(log_path)


def plot_training_metrics(metrics: pd.DataFrame, out_dir: Path, window: int = 25) -> tuple[Path, Path]:
    plots_dir = out_dir / 'plots'
    plots_dir.mkdir(parents=True, exist_ok=True)
    reward_path = plots_dir / 'reward_curve.png'
    success_path = plots_dir / 'success_rate_curve.png'

    metrics = metrics.copy()
    metrics['reward_ma'] = metrics['episode_reward'].rolling(window=window, min_periods=1).mean()
    metrics['success_ma'] = metrics['success'].rolling(window=window, min_periods=1).mean()

    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.plot(metrics['episode'], metrics['episode_reward'], alpha=0.25, color='#3f51b5', label='Episode reward')
    ax.plot(metrics['episode'], metrics['reward_ma'], color='#d84315', linewidth=2, label=f'{window}-ep mean')
    ax.set_title('Windy Courier Reward Curve')
    ax.set_xlabel('Episode')
    ax.set_ylabel('Reward')
    ax.legend()
    fig.tight_layout()
    fig.savefig(reward_path, dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.plot(metrics['episode'], metrics['success_ma'], color='#2e7d32', linewidth=2)
    ax.fill_between(metrics['episode'], 0, metrics['success_ma'], color='#81c784', alpha=0.35)
    ax.set_ylim(0, 1.05)
    ax.set_title('Rolling Success Rate')
    ax.set_xlabel('Episode')
    ax.set_ylabel('Success rate')
    fig.tight_layout()
    fig.savefig(success_path, dpi=180)
    plt.close(fig)

    return reward_path, success_path


def write_summary(summary: dict, path: Path) -> None:
    path.write_text(json.dumps(summary, indent=2), encoding='utf-8')


def merge_videos_to_gif(video_paths: Iterable[Path], gif_path: Path, fps: int = 10) -> Path:
    frames: list[np.ndarray] = []
    for video_path in video_paths:
        if not video_path.exists():
            continue
        reader = imageio.get_reader(video_path)
        try:
            for frame in reader:
                frames.append(frame)
        finally:
            reader.close()
    if not frames:
        raise RuntimeError('No frames available to build GIF.')
    imageio.mimsave(gif_path, frames, fps=fps)
    return gif_path
