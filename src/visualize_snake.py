import argparse
import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import cv2
from stable_baselines3 import PPO, A2C
from envs.snake_env import SnakeEnv

ALGOS = {"ppo": PPO, "a2c": A2C}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--algo", choices=["ppo", "a2c"], required=True)
    parser.add_argument("--model", required=True, help="Path to .zip model (e.g., models/ppo_snake_efficiency_seed7.zip)")
    parser.add_argument("--persona", choices=["maximizer", "efficiency"], default="efficiency", help="Snake personas: maximizer or efficiency")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-steps", type=int, default=3000)
    parser.add_argument("--fps", type=int, default=8)
    parser.add_argument("--stochastic", action="store_true", help="Sample actions stochastically")
    args = parser.parse_args()

    # Load model
    Model = ALGOS[args.algo]
    model = Model.load(args.model)

    # Build environment (render to RGB arrays for display)
    env = SnakeEnv(persona=args.persona, seed=args.seed, render_mode="rgb_array")
    obs, info = env.reset(seed=args.seed)

    # UI
    win = f"{args.algo.upper()} playing Snake ({args.persona})"
    cv2.namedWindow(win, cv2.WINDOW_AUTOSIZE)
    delay = max(1, int(1000 / max(args.fps, 1)))

    deterministic = not args.stochastic
    score = info.get("score", 0)

    for step in range(1, args.max_steps + 1):
        action, _ = model.predict(obs, deterministic=deterministic)
        obs, reward, terminated, truncated, info = env.step(int(action))
        score = info.get("score", score)

        # NOTE: Since env was created with render_mode='rgb_array', render() returns an RGB frame.
        frame = env.render(mode="rgb_array")  # keep signature as in your env
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        frame = cv2.resize(frame, (600, 600), interpolation=cv2.INTER_NEAREST)

        hud = f"step {step} | apples {score} | reward {reward:.3f} | {'det' if deterministic else 'stoch'}"
        cv2.putText(frame, hud, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (220,220,220), 2, cv2.LINE_AA)
        cv2.imshow(win, frame)

        key = cv2.waitKey(delay)
        if key != -1:
            print("Stopped by user.")
            break

        if terminated or truncated:
            print(f"Episode finished at step {step} — apples: {score}")
            break

    cv2.destroyWindow(win)
    env.close()

if __name__ == "__main__":
    main()
