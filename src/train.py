import argparse
import os
from stable_baselines3 import PPO, A2C
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.utils import set_random_seed
from envs.game_2048_env import Game2048Env

ALGOS = {"ppo": PPO, "a2c": A2C}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--algo", choices=["ppo","a2c"], required=True)
    parser.add_argument("--env", choices=["2048"], required=True, help="For now, train on 2048")
    parser.add_argument("--timesteps", type=int, default=200_000)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--persona", choices=["maximizer","efficiency"], default="maximizer")
    args = parser.parse_args()

    set_random_seed(args.seed)
    algo_cls = ALGOS[args.algo]

    if args.env == "2048":
        vec_env = make_vec_env(lambda: Game2048Env(persona=args.persona, seed=args.seed), n_envs=4)
    else:
        raise NotImplementedError("Only 2048 training is enabled in this starter.")

    model = algo_cls("MlpPolicy", vec_env, verbose=1, seed=args.seed)
    model.learn(total_timesteps=args.timesteps)

    os.makedirs("models", exist_ok=True)
    out = f"models/{args.algo}_2048_seed{args.seed}.zip"
    model.save(out)
    print(f"Saved model to {out}")

if __name__ == "__main__":
    main()