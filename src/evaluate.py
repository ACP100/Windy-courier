from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from stable_baselines3 import PPO

from src.utils import ensure_asset_dirs, record_episode_video, write_summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Evaluate a trained Windy Courier PPO agent.')
    parser.add_argument('--model-path', type=Path, default=Path('assets/final_model/model.zip'))
    parser.add_argument('--record', action='store_true')
    parser.add_argument('--episodes', type=int, default=3)
    parser.add_argument('--seed', type=int, default=123)
    parser.add_argument('--save-dir', type=Path, default=Path('assets'))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    asset_dirs = ensure_asset_dirs(args.save_dir)

    if not args.model_path.exists():
        raise FileNotFoundError(f'Model not found: {args.model_path}')

    model = PPO.load(args.model_path)
    rows = []
    for episode in range(args.episodes):
        video_path = asset_dirs['videos'] / f'eval_episode_{episode + 1}.mp4'
        result = record_episode_video(
            model,
            video_path,
            seed=args.seed + episode,
            deterministic=True,
        )
        rows.append(
            {
                'episode': episode + 1,
                'reward': result.reward,
                'steps': result.steps,
                'success': int(result.success),
                'collision': int(result.collision),
                'out_of_bounds': int(result.out_of_bounds),
                'video_path': str(video_path),
            }
        )
        print(f'[eval] episode={episode + 1} reward={result.reward:.2f} success={result.success} video={video_path}')
        if not args.record:
            break

    df = pd.DataFrame(rows)
    csv_path = asset_dirs['final_model'] / 'evaluation_metrics.csv'
    df.to_csv(csv_path, index=False)

    summary = {
        'episodes': len(rows),
        'mean_reward': float(df['reward'].mean()),
        'success_rate': float(df['success'].mean()),
        'metrics_csv': str(csv_path),
    }
    write_summary(summary, asset_dirs['final_model'] / 'evaluation_summary.json')
    print(f'[eval] metrics saved to {csv_path}')


if __name__ == '__main__':
    main()
