import os, sys
import imageio
import numpy as np
import time
import matplotlib.pyplot as plt
from stable_baselines3 import A2C, PPO


# Add parent directory to import  custom env
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from envs.game_2048_env import Game2048Env

def visualize_model(model_path, model_class, algo_name):
    import cv2  # Add here for OpenCV visualization
    print(f"\n Visualizing {algo_name} ...")
    model = model_class.load(model_path)
    env = Game2048Env(persona="maximizer", seed=42)
    obs, info = env.reset()
    frames = []

    for step in range(300):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, trunc, info = env.step(int(action))

        # Get rendered frame
        frame = env.render(mode="rgb_array") if hasattr(env, "render") else np.zeros((400, 400, 3), dtype=np.uint8)
        frame = cv2.resize(frame, (512, 512), interpolation=cv2.INTER_CUBIC)
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        # Show frame in an OpenCV window
        cv2.imshow(f"{algo_name} playing 2048", frame_bgr)
        time.sleep(0.5)
        # Wait ~300ms or until a key press; if window closed, break cleanly
        key = cv2.waitKey(1000)
        if key != -1 or cv2.getWindowProperty(f"{algo_name} playing 2048", cv2.WND_PROP_VISIBLE) < 1:
            print("\n Visualization manually closed.")
            break

        frames.append(frame)
        if done:
            print(f"\nEpisode finished with score: {info.get('score', 0)}")
            break

    env.close()
    cv2.destroyAllWindows()

if __name__ == "__main__":
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