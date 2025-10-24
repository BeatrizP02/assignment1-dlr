import argparse
import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd
from stable_baselines3 import PPO, A2C
from envs.game_2048_env import Game2048Env

def load_model(path: str):
    if "ppo" in os.path.basename(path).lower():
        return PPO.load(path)
    else:
        return A2C.load(path)

def run_episode(model, persona: str, seed: int = 42):
    import time
    env = Game2048Env(persona=persona, seed=seed)
    obs, info = env.reset()
    done = False
    total_reward = 0.0
    steps = 0
    max_tile = 0
    final_score = 0
    max_steps = 300  # stop early
    start_time = time.time()

    print(f"Running episode with persona='{persona}'...", flush=True)

    while not done and steps < max_steps:
        # Predict next action
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, trunc, info = env.step(int(action))
        total_reward += reward
        steps += 1
        max_tile = max(max_tile, int(env.board.max()))
        final_score = info.get("score", final_score)

        # Print progress every 50 steps
        if steps % 50 == 0:
            print(f"  Step {steps}: Score={final_score}, Max Tile={max_tile}", flush=True)

        # Hard stop if it runs more than 10 seconds
        if time.time() - start_time > 10:
            print("Timeout reached (10s) — ending early.")
            break

    print(f"Episode done after {steps} steps — Final Score: {final_score}\n", flush=True)
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
        print(f"Running episode {ep+1}/{args.episodes}...", flush=True)
        result = run_episode(model, persona=args.persona, seed=args.seed + ep)
        print(f"Finished episode {ep+1}: score={result['score']}, reward={result['reward']}")
        rows.append(result)


    df = pd.DataFrame(rows)
    algo_name = os.path.basename(args.model).split('_')[0].upper()
    df["algorithm"] = algo_name
    df["persona"] = args.persona 
    df["reward"] = df["reward"].round(2)

    os.makedirs("logs", exist_ok=True)
    out_csv = "logs/2048_metrics.csv"

    #  Append if file exists, otherwise create it
    #if os.path.exists(out_csv):
    #    existing = pd.read_csv(out_csv)
    #    df = pd.concat([existing, df], ignore_index=True)

    # check if file exists
    file_exists = os.path.exists(out_csv)

# append new data
    if file_exists:
        existing = pd.read_csv(out_csv)
        df = pd.concat([existing, df], ignore_index=True)

    df.to_csv(out_csv, index=False)
    print(df.describe())
    print(f"Appended results to {out_csv}")


if __name__ == "__main__":
    main()