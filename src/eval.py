import argparse
import os
import pandas as pd
from stable_baselines3 import PPO, A2C
from envs.game_2048_env import Game2048Env

def load_model(path: str):
    if "ppo" in os.path.basename(path).lower():
        return PPO.load(path)
    else:
        return A2C.load(path)

def run_episode(model, persona: str, seed: int = 42):
    env = Game2048Env(persona=persona, seed=seed)
    obs, info = env.reset()
    done = False
    total_reward = 0.0
    steps = 0
    max_tile = 0
    final_score = 0
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, trunc, info = env.step(int(action))
        total_reward += reward
        steps += 1
        max_tile = max(max_tile, int(env.board.max()))
        final_score = info.get("score", final_score)
    return {"reward": total_reward, "steps": steps, "score": final_score, "max_tile": max_tile}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--persona", choices=["maximizer","efficiency"], default="maximizer")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    model = load_model(args.model)
    rows = []
    for ep in range(args.episodes):
        rows.append(run_episode(model, persona=args.persona, seed=args.seed + ep))

    df = pd.DataFrame(rows)
    os.makedirs("logs", exist_ok=True)
    out_csv = "logs/2048_metrics.csv"
    df.to_csv(out_csv, index=False)
    print(df.describe())
    print(f"Wrote {out_csv}")

if __name__ == "__main__":
    main()