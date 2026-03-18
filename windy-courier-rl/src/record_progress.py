from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont

from src.utils import load_metrics, merge_videos_to_gif, plot_training_metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Generate Windy Courier plots, GIFs, and web assets.')
    parser.add_argument('--log-dir', type=Path, default=Path('assets'))
    parser.add_argument('--out-dir', type=Path, default=Path('assets'))
    return parser.parse_args()


def create_architecture_diagram(out_path: Path) -> None:
    width, height = 1200, 700
    image = Image.new('RGB', (width, height), '#f4f6f9')
    draw = ImageDraw.Draw(image)
    try:
        title_font = ImageFont.truetype('arial.ttf', 34)
        font = ImageFont.truetype('arial.ttf', 24)
    except OSError:
        title_font = ImageFont.load_default()
        font = ImageFont.load_default()

    draw.text((50, 30), 'Windy Courier RL System', fill='#1f2937', font=title_font)
    boxes = [
        ((60, 130, 330, 290), '#dbeafe', 'Custom Gymnasium Env', 'Drone + wind zones + moving obstacle'),
        ((430, 130, 770, 290), '#fde68a', 'PPO Agent', 'Learns pickup and delivery under drift'),
        ((850, 130, 1140, 290), '#dcfce7', 'Artifacts', 'Videos, GIFs, reward curve, success plot'),
        ((240, 420, 560, 590), '#ede9fe', 'Evaluation', 'Deterministic rollout with MP4 recording'),
        ((680, 420, 1060, 590), '#fee2e2', 'Showcase', 'GitHub Pages + optional HF Space demo'),
    ]

    for coords, color, title, subtitle in boxes:
        draw.rounded_rectangle(coords, radius=24, fill=color, outline='#94a3b8', width=3)
        draw.text((coords[0] + 20, coords[1] + 22), title, fill='#111827', font=font)
        draw.text((coords[0] + 20, coords[1] + 78), subtitle, fill='#334155', font=font)

    arrows = [((330, 210), (430, 210)), ((770, 210), (850, 210)), ((600, 290), (400, 420)), ((780, 290), (860, 420))]
    for start, end in arrows:
        draw.line([start, end], fill='#475569', width=6)
        draw.polygon([(end[0], end[1]), (end[0] - 18, end[1] - 10), (end[0] - 18, end[1] + 10)], fill='#475569')

    image.save(out_path)


def copy_to_web(media_paths: dict[str, Path], web_dir: Path) -> None:
    media_dir = web_dir / 'media'
    media_dir.mkdir(parents=True, exist_ok=True)
    for name, src in media_paths.items():
        if src.exists():
            shutil.copy2(src, media_dir / src.name)
            print(f'[web] copied {name}: {src.name}')


def main() -> None:
    args = parse_args()
    metrics_path = args.log_dir / 'logs' / 'metrics.csv'
    videos_dir = args.log_dir / 'videos'
    gifs_dir = args.out_dir / 'gifs'
    gifs_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = args.out_dir / 'plots'
    plots_dir.mkdir(parents=True, exist_ok=True)

    metrics = load_metrics(metrics_path)
    reward_plot, success_plot = plot_training_metrics(metrics, args.out_dir)

    progression_candidates = [
        videos_dir / 'baseline_random.mp4',
        videos_dir / 'training_25.mp4',
        videos_dir / 'training_50.mp4',
        videos_dir / 'training_75.mp4',
        videos_dir / 'trained_final.mp4',
    ]
    progression_gif = gifs_dir / 'learning_progression.gif'
    merge_videos_to_gif(progression_candidates, progression_gif, fps=10)
    print(f'[gif] saved {progression_gif}')

    architecture_path = plots_dir / 'architecture_diagram.png'
    create_architecture_diagram(architecture_path)
    print(f'[plot] saved {architecture_path}')

    copy_to_web(
        {
            'progression_gif': progression_gif,
            'reward_plot': reward_plot,
            'success_plot': success_plot,
            'architecture': architecture_path,
            'baseline_video': videos_dir / 'baseline_random.mp4',
            'final_video': videos_dir / 'trained_final.mp4',
        },
        Path('web'),
    )


if __name__ == '__main__':
    main()
