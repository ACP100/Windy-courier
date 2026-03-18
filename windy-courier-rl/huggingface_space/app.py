from __future__ import annotations

import tempfile
from pathlib import Path

import gradio as gr
from stable_baselines3 import PPO

from src.utils import record_episode_video

MODEL_PATH = Path('assets/final_model/model.zip')


def run_demo(seed: int) -> tuple[str, str]:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f'Model not found at {MODEL_PATH}. Upload the trained model before launching the Space.')

    model = PPO.load(MODEL_PATH)
    tmp_dir = Path(tempfile.mkdtemp())
    video_path = tmp_dir / 'windy_courier_demo.mp4'
    result = record_episode_video(model, video_path, seed=seed, deterministic=True)
    stats = (
        f'Success: {result.success}\n'
        f'Reward: {result.reward:.2f}\n'
        f'Steps: {result.steps}\n'
        f'Collision: {result.collision}\n'
        f'Out of bounds: {result.out_of_bounds}'
    )
    return str(video_path), stats


demo = gr.Interface(
    fn=run_demo,
    inputs=gr.Slider(minimum=0, maximum=9999, step=1, value=123, label='Seed'),
    outputs=[gr.Video(label='Inference Episode'), gr.Textbox(label='Summary Stats')],
    title='Windy Courier RL Demo',
    description='Runs one inference-only episode of the trained PPO courier policy and returns an MP4.',
    allow_flagging='never',
)


if __name__ == '__main__':
    demo.launch()
