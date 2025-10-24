import os, sys
import imageio
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import A2C, PPO


# Add parent directory to import  custom env
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from envs.game_2048_env import Game2048Env

def visualize_model(model_path, model_class, algo_name):
    print(f"\n Visualizing {algo_name} ...")
    model = model_class.load(model_path)
    env = Game2048Env(persona="maximizer", seed=42)
    obs, info = env.reset()
    frames = []

    for _ in range(300):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, trunc, info = env.step(int(action))
        frame = env.render(mode="rgb_array") if hasattr(env, "render") else np.zeros((400, 400, 3), dtype=np.uint8)
        frames.append(frame)
        if done:
            break

    env.close()
    os.makedirs("outputs", exist_ok=True)
    gif_path = f"outputs/2048_agent_{algo_name}.gif"
    imageio.mimsave(gif_path, frames, fps=4)
    print(f" Saved visualization: {gif_path}")

print("Select which model to visualize:")
print("1. PPO")
print("2. A2C")
choice = input("Enter 1 or 2: ").strip()

if choice == "1":
    visualize_model("models/ppo_2048_seed7.zip", PPO, "PPO")
elif choice == "2":
    visualize_model("models/a2c_2048_seed7.zip", A2C, "A2C")
else:
    print("Invalid choice. Please run again and enter 1 or 2.")