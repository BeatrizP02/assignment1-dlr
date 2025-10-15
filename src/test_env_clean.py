import sys, os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from envs.game_2048_env import Game2048Env

print(">>> Script started", flush=True)

# Initialize the 2048 environment
env = Game2048Env(persona="maximizer", seed=7)
obs, info = env.reset()
print(">>> Environment reset complete", flush=True)

done = False
steps = 0

# Run a random episode
while not done:
    action = env.action_space.sample()
    obs, reward, done, trunc, info = env.step(action)
    steps += 1
    if steps % 50 == 0:
        print(f"Step {steps}: score={info['score']}", flush=True)

print(">>> Finished! Final score:", info["score"], flush=True)
