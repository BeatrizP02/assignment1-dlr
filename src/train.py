from __future__ import annotations
import argparse
import os, sys
import pandas as pd
from stable_baselines3 import PPO, A2C
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.callbacks import BaseCallback

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from envs.snake_env import SnakeEnv          
from envs.game_2048_env import Game2048Env   

ALGOS = {"ppo": PPO, "a2c": A2C}

# 
class MetricsCallback(BaseCallback):
    def __init__(self, log_dir: str, env_name: str, persona: str, algo: str, verbose: int = 0):
        super().__init__(verbose)
        self.log_dir = log_dir
        self.env_name = env_name
        self.persona = persona
        self.algo = algo
        self.metrics = []
        os.makedirs(log_dir, exist_ok=True)

    def _on_step(self) -> bool:
        for env_idx in range(self.training_env.num_envs):
            if self.locals["dones"][env_idx]:
                info = self.locals["infos"][env_idx]
                metrics = {
                    "episode": len(self.metrics) + 1,
                    "timestep": self.num_timesteps,
                    "score": info.get("score", 0),
                    "episode_length": info.get("episode_length", 0)
                }
                if self.env_name == "snake":
                    metrics["death_type"] = info.get("death_type", None)
                if self.env_name == "2048":
                    metrics["max_tile"] = info.get("max_tile", 0)
                    metrics["empty_tiles"] = info.get("empty_tiles", 0)
                self.metrics.append(metrics)
                pd.DataFrame(self.metrics).to_csv(
                    os.path.join(self.log_dir, f"training_metrics_{self.env_name}_{self.algo}_{self.persona}.csv"),
                    index=False
                )
        return True

def main():
    parser = argparse.ArgumentParser(description="Train DRL models for Snake or 2048")
    parser.add_argument("--algo", choices=["ppo", "a2c"], required=True, help="Algorithm to use (ppo or a2c)")
    parser.add_argument("--env", choices=["2048", "snake"], required=True, help="Environment to train on")  # CHANGED: added snake
    parser.add_argument("--timesteps", type=int, default=200_000, help="Total training timesteps")
    parser.add_argument("--seed", type=int, default=7, help="Random seed for reproducibility")
    parser.add_argument("--persona", choices=["maximizer", "efficiency"], default="efficiency", help="Reward persona")
    parser.add_argument("--log-dir", type=str, default="logs", help="Directory for training logs")
    args = parser.parse_args()

    set_random_seed(args.seed)
    algo_cls = ALGOS[args.algo]

    if args.env == "2048":
        vec_env = make_vec_env(lambda: Game2048Env(persona=args.persona, seed=args.seed), n_envs=4)
    elif args.env == "snake":
        vec_env = make_vec_env(lambda: SnakeEnv(persona=args.persona, seed=args.seed), n_envs=4)
    else:
        raise NotImplementedError("Only snake and 2048 are supported.")

    model_name = f"{args.algo}_{args.env}_{args.persona}_seed{args.seed}.zip"  # CHANGED: persona included
    model = algo_cls("MlpPolicy", vec_env, verbose=1, seed=args.seed)

    os.makedirs(args.log_dir, exist_ok=True)
    callback = MetricsCallback(log_dir=args.log_dir, env_name=args.env, persona=args.persona, algo=args.algo)
    model.learn(total_timesteps=args.timesteps, callback=callback)

    os.makedirs("models", exist_ok=True)
    out = os.path.join("models", model_name)
    model.save(out)
    print(f"Saved model to {out}")

    if callback.metrics:
        df = pd.DataFrame(callback.metrics)
        aggregate_stats = {
            "mean_score": df["score"].mean(),
            "std_score": df["score"].std(),
            "mean_episode_length": df["episode_length"].mean(),
            "std_episode_length": df["episode_length"].std(),
        }
        if args.env == "snake":
            aggregate_stats.update({
                "death_rate": df["death_type"].notnull().mean(),
                "wall_death_rate": (df["death_type"] == "wall").mean(),
                "self_death_rate": (df["death_type"] == "self").mean()
            })
        if args.env == "2048":
            aggregate_stats.update({
                "mean_max_tile": df["max_tile"].mean(),
                "mean_empty_tiles": df["empty_tiles"].mean()
            })
        pd.Series(aggregate_stats).to_csv(
            os.path.join(args.log_dir, f"training_aggregate_stats_{args.env}_{args.persona}.csv")
        )
        print(f"Saved aggregate stats to {args.log_dir}/training_aggregate_stats_{args.env}_{args.persona}.csv")

if __name__ == "__main__":
    main()