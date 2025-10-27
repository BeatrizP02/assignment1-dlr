from __future__ import annotations
import gymnasium as gym
from gymnasium import spaces
import numpy as np
from dataclasses import dataclass
from typing import Optional, List
import random

@dataclass
class SnakeConfig:
    grid_n: int = 12
    death_penalty: float = 1.0 
    max_steps: int = 2000        

# Directions: R, D, L, U
DIRS = np.array([[0, 1], [1, 0], [0, -1], [-1, 0]], dtype=np.int32)

class SnakeEnv(gym.Env):
    """
    Actions (Discrete 3, relative to current heading):
      0 = turn left, 1 = go straight, 2 = turn right

    Observation (float32, shape (12,)):
      [ head_x_norm, head_y_norm, dir_idx_norm,
        apple_dx_norm, apple_dy_norm,
        dist_R_norm, dist_D_norm, dist_L_norm, dist_U_norm,
        danger_ahead, danger_left, danger_right ]

    Personas rewards:
      - 'maximizer': +2.0 per apple, -0.05 per step, -0.001*length (hunger),
                     death penalty scaled by score
      - 'efficiency': +1.0 per apple, -0.04 per step,
                      same scaled death penalty
      Both: +0.05/-0.05 for moving closer/farther from apple, -0.1 for danger ahead

    Metrics in info:
      {"score": apples_eaten, "apples": apples_eaten, "episode_length": steps, "death_type": "wall"/"self"/None}
    """
    metadata = {"render_modes": ["rgb_array"]}

    def __init__(self,
                 persona: str = "efficiency",
                 seed: Optional[int] = None,
                 cfg: SnakeConfig = SnakeConfig(),
                 render_mode: Optional[str] = None):
        super().__init__()
        assert persona in ("maximizer", "efficiency")
        self.persona = persona
        self.cfg = cfg
        self.render_mode = render_mode

        self.rng = random.Random(seed)
        self.np_rng = np.random.default_rng(seed)

        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(low=-1.0, high=1.0, shape=(12,), dtype=np.float32)

        self.snake: List[np.ndarray] = []
        self.dir_idx: int = 0
        self.apple: np.ndarray = np.zeros(2, dtype=np.int32)

        self.steps = 0
        self.score = 0

    def seed(self, seed: Optional[int] = None):
        self.rng = random.Random(seed)
        self.np_rng = np.random.default_rng(seed)

    def reset(self, *, seed: Optional[int] = None, options=None):
        if seed is not None:
            self.seed(seed)
        self.steps = 0
        self.score = 0
        self._spawn_snake()
        self._spawn_apple()
        obs = self._obs()
        info = {"score": self.score, "apples": self.score, "episode_length": 0, "death_type": None}
        return obs, info

    def step(self, action: int):
        self.steps += 1
        death_type = None

        # Distance to apple before moving (Manhattan)
        old_dist = self._manhattan_distance(self.snake[0], self.apple)

        # Action: 0 left, 1 straight, 2 right
        if action == 0:
            self.dir_idx = (self.dir_idx - 1) % 4
        elif action == 2:
            self.dir_idx = (self.dir_idx + 1) % 4

        head = self.snake[0].copy()
        head = head + DIRS[self.dir_idx]

        n = self.cfg.grid_n
        out_of_bounds = (head[0] < 0 or head[0] >= n or head[1] < 0 or head[1] >= n)

        snake_set = {tuple(p) for p in self.snake}
        self_collision = (not out_of_bounds) and (tuple(head) in snake_set)
        dead = out_of_bounds or self_collision

        reward = 0.0
        if dead:
            # Dynamic death penalty: less severe for higher scores
            reward = -self.cfg.death_penalty * (1 - min(self.score / 50, 0.9))
            death_type = "wall" if out_of_bounds else "self"
            terminated = True
            truncated = False
            info = {
                "score": self.score,
                "apples": self.score,
                "episode_length": self.steps,
                "death_type": death_type
            }
            return self._obs(), float(reward), terminated, truncated, info

        # Move forward
        self.snake.insert(0, head)

        ate = np.array_equal(head, self.apple)
        if ate:
            self.score += 1
            # Persona-specific apple reward
            reward += 2.0 if self.persona == "maximizer" else 1.0
            self._spawn_apple()
        else:
            self.snake.pop()  # remove tail

        # Distance-based reward
        new_dist = self._manhattan_distance(self.snake[0], self.apple)
        if new_dist < old_dist:
            reward += 0.05   # moved closer
        elif new_dist > old_dist:
            reward -= 0.05   # moved farther

        # Danger penalty (both personas)
        ahead = self.snake[0] + DIRS[self.dir_idx]
        ahead_is_danger = (
            ahead[0] < 0 or ahead[0] >= n or
            ahead[1] < 0 or ahead[1] >= n or
            tuple(ahead) in {tuple(p) for p in self.snake}
        )
        if ahead_is_danger:
            reward -= 0.1

        # Persona-specific step/hunger rewards
        if self.persona == "maximizer":
            reward -= 0.05
            reward -= 0.001 * len(self.snake)  # hunger scales with length
        else:  # efficiency
            reward -= 0.04

        # Clip rewards to stabilize training
        reward = np.clip(reward, -1.0, 2.0)  # allow +2.0 for maximizer apples

        terminated = False
        truncated = self.steps >= self.cfg.max_steps
        info = {
            "score": self.score,
            "apples": self.score,
            "episode_length": self.steps,
            "death_type": None
        }
        return self._obs(), float(reward), terminated, truncated, info

    # -- internals --
    @staticmethod
    def _manhattan_distance(pos1: np.ndarray, pos2: np.ndarray) -> int:
        return abs(int(pos1[0]) - int(pos2[0])) + abs(int(pos1[1]) - int(pos2[1]))

    def _spawn_snake(self):
        n = self.cfg.grid_n
        start = np.array([self.rng.randrange(2, n - 2), self.rng.randrange(2, n - 2)], dtype=np.int32)
        self.snake = [start.copy(), start - np.array([0, 1], dtype=np.int32)]
        self.dir_idx = 0  # start moving Right

    def _spawn_apple(self):
        n = self.cfg.grid_n
        snake_set = {tuple(p) for p in self.snake}
        free = {(r, c) for r in range(n) for c in range(n)} - snake_set
        if not free:
            self.apple = self.snake[0].copy()
            return False
        self.apple = np.array(self.rng.choice(list(free)), dtype=np.int32)
        return True

    def _obs(self):
        n = self.cfg.grid_n
        head = self.snake[0]
        hx, hy = head.tolist()
        dx, dy = (self.apple - head).tolist()

        # normalized distances to borders
        dist_R = (n - 1 - hy) / (n - 1)
        dist_L = hy / (n - 1)
        dist_D = (n - 1 - hx) / (n - 1)
        dist_U = hx / (n - 1)

        ahead = head + DIRS[self.dir_idx]
        left_dir = DIRS[(self.dir_idx - 1) % 4]
        right_dir = DIRS[(self.dir_idx + 1) % 4]
        left_pos = head + left_dir
        right_pos = head + right_dir

        snake_set = {tuple(s) for s in self.snake}

        def danger(p: np.ndarray) -> float:
            x, y = int(p[0]), int(p[1])
            if x < 0 or x >= n or y < 0 or y >= n:
                return 1.0
            return 1.0 if (x, y) in snake_set else 0.0

        danger_ahead = danger(ahead)
        danger_left = danger(left_pos)
        danger_right = danger(right_pos)

        feat = np.array([
            hx / (n - 1) * 2 - 1,            # head_x_norm
            hy / (n - 1) * 2 - 1,            # head_y_norm
            self.dir_idx / 3 * 2 - 1,        # dir_idx_norm
            np.clip(dx / (n - 1), -1, 1),    # apple_dx_norm
            np.clip(dy / (n - 1), -1, 1),    # apple_dy_norm
            dist_R * 2 - 1,                  # dist_R_norm
            dist_D * 2 - 1,                  # dist_D_norm
            dist_L * 2 - 1,                  # dist_L_norm
            dist_U * 2 - 1,                  # dist_U_norm
            danger_ahead, danger_left, danger_right
        ], dtype=np.float32)
        return feat

    def render(self, mode: str = "rgb_array"):
        import matplotlib.pyplot as plt
        assert mode == "rgb_array"
        n = self.cfg.grid_n
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.set_xlim(0, n); ax.set_ylim(0, n)
        ax.set_xticks([]); ax.set_yticks([]); ax.set_aspect("equal")
        ax.set_facecolor((0.1, 0.1, 0.1))
        for i, p in enumerate(self.snake):
            c = (0.2, 0.8, 0.3) if i > 0 else (0.9, 0.9, 0.2)
            ax.add_patch(plt.Rectangle((p[1], n - 1 - p[0]), 1, 1, color=c))
        ax.add_patch(plt.Rectangle((self.apple[1], n - 1 - self.apple[0]), 1, 1, color=(0.9, 0.2, 0.2)))
        fig.canvas.draw()
        frame = np.array(fig.canvas.buffer_rgba())[..., :3]
        plt.close(fig)
        return frame
