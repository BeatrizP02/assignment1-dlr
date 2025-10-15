import gymnasium as gym
from gymnasium import spaces
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager
import numpy as np
import os, time

class WebWorkflowEnv(gym.Env):
    """
    Minimal Selenium-based env for a 3-step workflow in web_app/.
    Actions:
      0 = fill field (if present)
      1 = click primary button (Next/Submit)
      2 = noop (wait)
    Observation: simple discrete page index encoded as one-hot (length 4).
    Rewards:
      +1 on valid transition, +5 on reaching success page, -1 on invalid submission.
    """
    metadata = {"render_modes": ["human"]}

    def __init__(self, base_dir: str = None):
        super().__init__()
        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(4,), dtype=np.float32)

        # Serve files from local filesystem
        self.base_dir = base_dir or os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "web_app"))
        self.pages = ["page1.html", "page2.html", "success.html", "error.html"]

        opts = Options()
        opts.add_argument("--headless=new")
        self.driver = webdriver.Chrome(ChromeDriverManager().install(), options=opts)
        self.page_idx = 0

    def _obs(self):
        v = np.zeros((4,), dtype=np.float32)
        v[self.page_idx] = 1.0
        return v

    def reset(self, *, seed=None, options=None):
        self.page_idx = 0
        self.driver.get("file://" + os.path.join(self.base_dir, self.pages[self.page_idx]))
        return self._obs(), {}

    def step(self, action: int):
        reward = 0.0
        done = False
        # Simple heuristic to interact with our demo pages
        try:
            if action == 0:
                # fill first input if present
                inputs = self.driver.find_elements(By.TAG_NAME, "input")
                if inputs:
                    inputs[0].clear()
                    inputs[0].send_keys("test@example.com")
                    reward += 0.5
            elif action == 1:
                buttons = self.driver.find_elements(By.TAG_NAME, "button")
                if buttons:
                    buttons[0].click()
                    time.sleep(0.1)
            elif action == 2:
                time.sleep(0.05)
        except Exception:
            pass

        # Update internal page_idx by checking h1 text
        h1 = self.driver.find_element(By.TAG_NAME, "h1").text.lower()
        prev = self.page_idx
        if "step 1" in h1:
            self.page_idx = 0
        elif "step 2" in h1:
            self.page_idx = 1
        elif "success" in h1:
            self.page_idx = 2
        else:
            self.page_idx = 3  # error

        if self.page_idx == 2:
            reward += 5.0
            done = True
        elif self.page_idx == 3:
            reward -= 1.0

        if self.page_idx != prev:
            reward += 1.0

        return self._obs(), reward, done, False, {}

    def close(self):
        try:
            self.driver.quit()
        except Exception:
            pass