import os
import sys
import time
import argparse
import numpy as np
import matplotlib.pyplot as plt

# 3rd party
from stable_baselines3 import PPO, A2C

# local imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from envs.game_2048_env import Game2048Env  # noqa


ALGOS = {"ppo": PPO, "a2c": A2C}


def load_model(model_path: str, algo_name: str):
    """Load a model, allowing model_path with or without '.zip'."""
    if model_path.endswith(".zip"):
        model_path = model_path[:-4]  # SB3 appends ".zip" automatically
    cls = ALGOS[algo_name]
    return cls.load(model_path)


def visualize(model, persona: str, fps: int = 2):
    """Run a short rendering loop with OpenCV."""
    import cv2  # import local to avoid hard dep when not visualizing

    env = Game2048Env(persona=persona, seed=42)
    obs, info = env.reset()

    win_title = f"2048 — {model.__class__.__name__} ({persona})"
    delay_ms = max(1, int(1000 / max(1, fps)))

    while True:
        action, _ = model.predict(obs, deterministic=True)
        step_out = env.step(int(action))
        if len(step_out) == 5:
            obs, reward, terminated, truncated, info = step_out
            done = terminated or truncated
        else:
            obs, reward, done, info = step_out

        # Render a frame (expects 'rgb_array')
        frame = env.render(mode="rgb_array") if hasattr(env, "render") else np.zeros((400, 400, 3), dtype=np.uint8)

        # Show with OpenCV
        frame = cv2.resize(frame, (512, 512), interpolation=cv2.INTER_CUBIC)
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        cv2.imshow(win_title, frame_bgr)

        key = cv2.waitKey(delay_ms)
        if key != -1:
            break
        if done:
            print(f"Episode finished. score={info.get('score')}, max_tile={info.get('max_tile')}")
            break

    env.close()
    try:
        import cv2
        cv2.destroyWindow(win_title)
    except Exception:
        pass


def main():
    parser = argparse.ArgumentParser(description="Visualize a trained 2048 model")
    parser.add_argument("--algo", choices=["ppo", "a2c"], required=True, help="Algorithm used to train the model")
    parser.add_argument("--model", required=True, help="Model path WITHOUT .zip (SB3 adds it) or with .zip (both work)")
    parser.add_argument("--persona", choices=["maximizer", "efficiency"], default="efficiency")
    parser.add_argument("--fps", type=int, default=2, help="Frames per second for the viewer")
    args = parser.parse_args()

    model = load_model(args.model, args.algo)
    visualize(model, persona=args.persona, fps=args.fps)


if __name__ == "__main__":
    main()
