from __future__ import annotations

import argparse
import time
from pathlib import Path

import pygame
from stable_baselines3 import PPO

from src.utils import make_env

KEY_PRIORITY = [
    (pygame.K_UP, 1),
    (pygame.K_w, 1),
    (pygame.K_DOWN, 2),
    (pygame.K_s, 2),
    (pygame.K_LEFT, 3),
    (pygame.K_a, 3),
    (pygame.K_RIGHT, 4),
    (pygame.K_d, 4),
    (pygame.K_SPACE, 0),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Play Windy Courier live in a pygame window as the agent or as a human.')
    parser.add_argument('--mode', choices=['agent', 'human'], default='agent')
    parser.add_argument('--model-path', type=Path, default=Path('assets/final_model/model.zip'))
    parser.add_argument('--seed', type=int, default=123)
    parser.add_argument('--max-steps', type=int, default=500)
    parser.add_argument('--deterministic', action='store_true', help='Use deterministic policy actions in agent mode.')
    parser.add_argument('--sleep', type=float, default=0.0, help='Optional extra delay between steps in seconds.')
    parser.add_argument('--auto-restart-on-wall', action='store_true', default=True)
    return parser.parse_args()


def run_agent_mode(args: argparse.Namespace) -> None:
    if not args.model_path.exists():
        raise FileNotFoundError(f'Model not found: {args.model_path}')

    model = PPO.load(args.model_path)
    env = make_env(render_mode='human', seed=args.seed, max_steps=args.max_steps, terminate_on_delivery=False)
    obs, _ = env.reset(seed=args.seed)
    total_reward = 0.0
    steps = 0
    restarts = 0

    try:
        while True:
            action, _ = model.predict(obs, deterministic=args.deterministic)
            obs, reward, terminated, truncated, info = env.step(int(action))
            total_reward += reward
            steps += 1

            if info.get('is_success', False):
                print(f"[play-agent] delivery #{info.get('deliveries', 0)} score={info.get('score', total_reward):.2f}")

            if info.get('out_of_bounds', False) and args.auto_restart_on_wall:
                restarts += 1
                print(f'[play-agent] hit wall, restarting run #{restarts} from spawn.')
                obs, _ = env.reset(seed=args.seed + restarts)
                continue

            if args.sleep > 0:
                time.sleep(args.sleep)

            if terminated or truncated:
                print(
                    f'[play-agent] score={info.get("score", total_reward):.2f} steps={steps} '
                    f"deliveries={info.get('deliveries', 0)} restarts={restarts} "
                    f"collision={info.get('collision', False)} "
                    f"out_of_bounds={info.get('out_of_bounds', False)}"
                )
                break
    finally:
        env.close()


def read_continuous_action() -> tuple[int | None, bool]:
    quit_requested = False
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            quit_requested = True
        elif event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
            quit_requested = True

    if quit_requested:
        return None, True

    keys = pygame.key.get_pressed()
    for key, action in KEY_PRIORITY:
        if keys[key]:
            return action, False
    return 0, False


def run_human_mode(args: argparse.Namespace) -> None:
    env = make_env(render_mode='human', seed=args.seed, max_steps=args.max_steps, terminate_on_delivery=False)
    _, info = env.reset(seed=args.seed)
    total_reward = 0.0
    steps = 0
    last_action = 0
    clock = pygame.time.Clock()
    restarts = 0

    print('[play-human] Controls: hold arrow keys or WASD to move, space to brake, Esc to quit.')
    print('[play-human] Movement is continuous. Deliveries do not end the run.')

    try:
        while True:
            action, quit_requested = read_continuous_action()
            if quit_requested or action is None:
                print('[play-human] Exiting.')
                break

            last_action = action
            _, reward, terminated, truncated, info = env.step(last_action)
            total_reward += reward
            steps += 1

            if info.get('is_success', False):
                print(f"[play-human] delivery #{info.get('deliveries', 0)} complete, score={info.get('score', total_reward):.2f}")

            if info.get('out_of_bounds', False) and args.auto_restart_on_wall:
                restarts += 1
                print(f'[play-human] hit wall, restarting from spawn. restarts={restarts}')
                _, info = env.reset(seed=args.seed + restarts)
                continue

            if terminated or truncated:
                print(
                    f'[play-human] final_score={info.get("score", total_reward):.2f} steps={steps} '
                    f"deliveries={info.get('deliveries', 0)} restarts={restarts} "
                    f"collision={info.get('collision', False)} "
                    f"out_of_bounds={info.get('out_of_bounds', False)}"
                )
                break

            if args.sleep > 0:
                time.sleep(args.sleep)
            else:
                clock.tick(20)
    finally:
        env.close()


def main() -> None:
    args = parse_args()
    if args.mode == 'agent':
        run_agent_mode(args)
    else:
        run_human_mode(args)


if __name__ == '__main__':
    main()
