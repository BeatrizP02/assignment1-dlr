import imageio
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import A2C
from gymnasium import make

# --- 1. Load your trained model ---
model = A2C.load("models/a2c_2048")

# --- 2. Create the environment ---
env = make("gym_2048:2048-v0", render_mode="rgb_array")  # Change ID if custom

frames = []
obs, _ = env.reset()
done = False

# --- 3. Run one full episode ---
while not done:
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, done, truncated, info = env.step(action)
    frame = env.render()  # RGB frame (numpy array)
    frames.append(frame)

env.close()

# --- 4. Save frames as a GIF ---
gif_path = "outputs/2048_agent_play.gif"
imageio.mimsave(gif_path, frames, fps=4)
print(f"GIF saved at: {gif_path}")
