import gymnasium as gym
from gymnasium import spaces
import numpy as np
import random
import math
from typing import Optional

class Game2048Env(gym.Env):
    """
    2048 Gymnasium Environment.
    Observation: 4x4 board normalized to [0,1] using log2 scaling.
    Actions: 0=UP, 1=DOWN, 2=LEFT, 3=RIGHT
    Reward personas:
      - maximizer: reward = score_delta
      - efficiency: reward = score_delta / (moves+1)
    """
    metadata = {"render_modes": ["ansi"]}

    def __init__(self, persona: str = "maximizer", seed: Optional[int] = None):
        super().__init__()
        self.size = 4
        self.persona = persona
        self.rng = random.Random(seed)
        self.np_rng = np.random.default_rng(seed)
        self.action_space = spaces.Discrete(4)
        # Use float32 observations for SB3 compatibility
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(16,), dtype=np.float32)
        self.board = None
        self.score = 0
        self.moves = 0

    def _normalized(self):
        # map tile -> log2(tile)/log2(2048); empty=0
        flat = self.board.flatten().astype(np.float32)
        out = np.zeros_like(flat, dtype=np.float32)
        for i, v in enumerate(flat):
            if v > 0:
                out[i] = math.log2(v) / math.log2(2048)
            else:
                out[i] = 0.0
        return out

    def seed(self, seed=None):
        self.rng = random.Random(seed)
        self.np_rng = np.random.default_rng(seed)

    def reset(self, *, seed=None, options=None):
        if seed is not None:
            self.seed(seed)
        self.board = np.zeros((self.size, self.size), dtype=np.int32)
        self.score = 0
        self.moves = 0
        self._add_tile()
        self._add_tile()
        obs = self._normalized()
        return obs, {"score": self.score, "moves": self.moves}

    def step(self, action: int):
        assert self.board is not None
        if action not in [0,1,2,3]:
            raise ValueError("Invalid action")

        old_score = self._score()
        changed = self._move(action)
        if changed:
            self._add_tile()
            self.moves += 1

        new_score = self._score()
        score_delta = new_score - old_score

        if self.persona == "efficiency":
            reward = score_delta / float(self.moves + 1)
        else:  # maximizer
            reward = float(score_delta)

        terminated = self._is_done()
        truncated = False
        self.score = new_score
        info = {"score": int(self.score), "moves": int(self.moves), "changed": bool(changed)}
        return self._normalized(), reward, terminated, truncated, info

    def render(self):
        return str(self.board)

    # --- 2048 mechanics ---
    def _add_tile(self):
        empties = list(zip(*np.where(self.board == 0)))
        if not empties:
            return
        r, c = empties[self.rng.randrange(len(empties))]
        self.board[r, c] = 2 if self.rng.random() < 0.9 else 4

    def _compress_merge_left(self, row):
        # Remove zeros
        tiles = row[row != 0].tolist()
        merged = []
        i = 0
        while i < len(tiles):
            if i + 1 < len(tiles) and tiles[i] == tiles[i+1]:
                merged.append(tiles[i] * 2)
                i += 2
            else:
                merged.append(tiles[i])
                i += 1
        # Pad with zeros
        merged += [0] * (self.size - len(merged))
        return np.array(merged, dtype=np.int32)

    def _move(self, action):
        before = self.board.copy()
        # Rotate so that action becomes LEFT
        # 0=UP -> rotate 1; 1=DOWN -> rotate 3; 2=LEFT -> rotate 0; 3=RIGHT -> rotate 2
        rotations = {0:1, 1:3, 2:0, 3:2}[action]
        rotated = np.rot90(self.board, k=rotations)
        for r in range(self.size):
            rotated[r] = self._compress_merge_left(rotated[r])
        # Rotate back
        self.board = np.rot90(rotated, k=(4 - rotations) % 4)
        changed = not np.array_equal(before, self.board)
        return changed

    def _score(self):
        # Standard proxy: sum of tiles
        return int(self.board.sum())

    def _is_done(self):
        if (self.board == 0).any():
            return False
        # Any possible merge?
        for i in range(self.size):
            for j in range(self.size - 1):
                if self.board[i, j] == self.board[i, j+1]:
                    return False
        for j in range(self.size):
            for i in range(self.size - 1):
                if self.board[i, j] == self.board[i+1, j]:
                    return False
        return True