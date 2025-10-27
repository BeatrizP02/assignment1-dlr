import argparse
import os, sys, time
import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from stable_baselines3 import PPO, A2C
from envs.game_2048_env import Game2048Env
from envs.snake_env import SnakeEnv



ALGOS = {"ppo": PPO, "a2c": A2C}

def load_model(path: str):
    base = os.path.basename(path).lower()
    if base.startswith("ppo_") or "ppo" in base:
        return PPO.load(path), "ppo"
    elif base.startswith("a2c_") or "a2c" in base:
        return A2C.load(path), "a2c"
    return PPO.load(path), "ppo"  # Default to PPO

def make_env(env_name: str, persona: str, seed: int):
    if env_name == "2048":
        return Game2048Env(persona=persona, seed=seed)
    elif env_name == "snake":
        return SnakeEnv(persona=persona, seed=seed)
    else:
        raise ValueError(f"Unknown env: {env_name}")

def run_episode(model, env_name: str, persona: str, seed: int, max_steps: int, ep_idx: int):
    env = make_env(env_name, persona, seed)
    obs, info = env.reset()
    total_reward = 0.0
    steps = 0
    final_score = info.get("score", 0)
    death_type = info.get("death_type", None)
    max_tile = 0
    empty_tiles = 0
    done = False
    start = time.time()

    while not done and steps < max_steps:
        action, _ = model.predict(obs, deterministic=True)
        out = env.step(int(action))
        if len(out) == 5:
            obs, reward, terminated, truncated, info = out
            done = terminated or truncated
        else:
            obs, reward, done, info = out

        total_reward += float(reward)
        steps += 1
        final_score = info.get("score", final_score)
        death_type = info.get("death_type", death_type)

        if env_name == "2048":
            max_tile = max(max_tile, info.get("max_tile", 0))
            empty_tiles = info.get("empty_tiles", empty_tiles)

        if time.time() - start > 20:
            print("Timeout (20s) — ending episode early.")
            break

    env.close()

    row = {
        "episode": ep_idx + 1,
        "env": env_name,
        "persona": persona,
        "reward": round(total_reward, 3),
        "steps": steps,
        "score": final_score,
    }
    if env_name == "snake":
        row["death_type"] = death_type
    if env_name == "2048":
        row["max_tile"] = max_tile
        row["empty_tiles"] = empty_tiles
    return row

def main():
    parser = argparse.ArgumentParser(description="Evaluate DRL models for Snake or 2048")
    parser.add_argument("--env", choices=["2048", "snake"], required=True)  # CHANGED: allow both envs
    parser.add_argument("--model", required=True, help="Path to trained model (.zip)")
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--persona", choices=["maximizer", "efficiency"], default="efficiency")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-steps", type=int, default=1000, help="Per-episode step cap")
    parser.add_argument("--log-dir", type=str, default="logs", help="Directory for evaluation logs")
    args = parser.parse_args()

    model, algo_name = load_model(args.model)

    os.makedirs(args.log_dir, exist_ok=True)
    rows = []
    for ep in range(args.episodes):
        row = run_episode(model, args.env, args.persona, args.seed + ep, args.max_steps, ep_idx=ep)
        rows.append(row)

    df = pd.DataFrame(rows)
    out_path = os.path.join(args.log_dir, f"eval_{args.env}_{algo_name}_{args.persona}.csv")
    df.to_csv(out_path, index=False)
    print(f"Saved evaluation results to {out_path}")
    print(df.describe(include="all"))

if __name__ == "__main__":
    main()
