from __future__ import annotations

import argparse
import csv
from pathlib import Path

from gymnasium.wrappers import RecordEpisodeStatistics
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

from src.utils import ensure_asset_dirs, make_env, record_episode_video, set_global_seed, write_summary


class ProgressArtifactsCallback(BaseCallback):
    def __init__(
        self,
        total_timesteps: int,
        save_dir: Path,
        seed: int,
        record_points: list[float],
        checkpoint_every: int,
        verbose: int = 1,
    ) -> None:
        super().__init__(verbose)
        self.total_timesteps = total_timesteps
        self.save_dir = save_dir
        self.seed = seed
        self.record_points = sorted(record_points)
        self.checkpoint_every = checkpoint_every
        self.next_checkpoint = checkpoint_every
        self.metrics_path = save_dir / 'logs' / 'metrics.csv'
        self.metrics_file = None
        self.metrics_writer = None
        self.episode_idx = 0
        self.completed_markers: set[int] = set()

    def _on_training_start(self) -> None:
        self.save_dir.joinpath('logs').mkdir(parents=True, exist_ok=True)
        self.metrics_file = self.metrics_path.open('w', newline='', encoding='utf-8')
        self.metrics_writer = csv.DictWriter(
            self.metrics_file,
            fieldnames=['episode', 'timesteps', 'episode_reward', 'episode_length', 'success', 'collision', 'out_of_bounds'],
        )
        self.metrics_writer.writeheader()

    def _on_step(self) -> bool:
        infos = self.locals.get('infos', [])
        dones = self.locals.get('dones', [])
        for done, info in zip(dones, infos):
            if not done:
                continue
            ep_info = info.get('episode')
            if ep_info is None:
                continue
            self.episode_idx += 1
            self.metrics_writer.writerow(
                {
                    'episode': self.episode_idx,
                    'timesteps': self.num_timesteps,
                    'episode_reward': float(ep_info['r']),
                    'episode_length': int(ep_info['l']),
                    'success': int(info.get('is_success', False)),
                    'collision': int(info.get('collision', False)),
                    'out_of_bounds': int(info.get('out_of_bounds', False)),
                }
            )
            self.metrics_file.flush()

        while self.num_timesteps >= self.next_checkpoint:
            checkpoint_path = self.save_dir / 'checkpoints' / f'ppo_step_{self.next_checkpoint}.zip'
            self.model.save(checkpoint_path)
            self.next_checkpoint += self.checkpoint_every

        for fraction in self.record_points:
            marker = int(fraction * 100)
            threshold = int(self.total_timesteps * fraction)
            if self.num_timesteps >= threshold and marker not in self.completed_markers:
                video_path = self.save_dir / 'videos' / f'training_{marker:02d}.mp4'
                result = record_episode_video(self.model, video_path, seed=self.seed + marker, deterministic=True)
                if self.verbose:
                    print(f'[record] saved {video_path} success={result.success} reward={result.reward:.2f}')
                self.completed_markers.add(marker)
        return True

    def _on_training_end(self) -> None:
        if self.metrics_file is not None:
            self.metrics_file.close()


def build_vec_env(seed: int, max_steps: int, n_envs: int) -> DummyVecEnv:
    def factory(rank: int):
        def _init():
            env = make_env(seed=seed + rank, max_steps=max_steps)
            env = RecordEpisodeStatistics(env)
            env = Monitor(env)
            return env
        return _init

    return DummyVecEnv([factory(rank) for rank in range(n_envs)])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Train PPO on the Windy Courier environment.')
    parser.add_argument('--total-timesteps', type=int, default=300_000)
    parser.add_argument('--save-dir', type=Path, default=Path('assets'))
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--record-frequency', type=float, default=0.25, help='Fractional interval used for progress videos.')
    parser.add_argument('--n-envs', type=int, default=8)
    parser.add_argument('--max-steps', type=int, default=180)
    parser.add_argument('--learning-rate', type=float, default=3e-4)
    parser.add_argument('--device', type=str, default='auto')
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_global_seed(args.seed)
    asset_dirs = ensure_asset_dirs(args.save_dir)

    print('[setup] recording baseline random policy video...')
    baseline_path = asset_dirs['videos'] / 'baseline_random.mp4'
    baseline_result = record_episode_video(None, baseline_path, seed=args.seed)
    print(f'[setup] baseline saved to {baseline_path} success={baseline_result.success}')

    env = build_vec_env(args.seed, args.max_steps, args.n_envs)
    checkpoint_every = max(args.total_timesteps // 4, 10_000)
    record_points = [args.record_frequency, 0.5, 0.75]

    model = PPO(
        'MlpPolicy',
        env,
        verbose=1,
        learning_rate=args.learning_rate,
        n_steps=1024,
        batch_size=256,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        seed=args.seed,
        device=args.device,
    )

    callback = ProgressArtifactsCallback(
        total_timesteps=args.total_timesteps,
        save_dir=args.save_dir,
        seed=args.seed,
        record_points=record_points,
        checkpoint_every=checkpoint_every,
    )

    model.learn(total_timesteps=args.total_timesteps, callback=callback, progress_bar=True)

    final_model_path = asset_dirs['final_model'] / 'model.zip'
    model.save(final_model_path)
    final_video_path = asset_dirs['videos'] / 'trained_final.mp4'
    final_result = record_episode_video(model, final_video_path, seed=args.seed + 999, deterministic=True)
    print(f'[done] final model saved to {final_model_path}')
    print(f'[done] final video saved to {final_video_path} success={final_result.success}')

    summary = {
        'seed': args.seed,
        'total_timesteps': args.total_timesteps,
        'n_envs': args.n_envs,
        'max_steps': args.max_steps,
        'baseline_video': str(baseline_path),
        'final_video': str(final_video_path),
        'final_model': str(final_model_path),
        'baseline_success': baseline_result.success,
        'final_success': final_result.success,
        'final_reward': final_result.reward,
    }
    write_summary(summary, asset_dirs['final_model'] / 'training_summary.json')
    env.close()


if __name__ == '__main__':
    main()
