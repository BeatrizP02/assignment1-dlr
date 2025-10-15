# DRL for Automated Testing — 2048 + Web Workflow

This repo implements a **Deep Reinforcement Learning (DRL) testing framework** on two targets:
1) A custom **2048** Gymnasium environment (game), and
2) A minimal **web workflow** (multi‑page form) for software testing via RL (Selenium).

It satisfies the assignment requirements:
- Two non-trivial apps (game + web app)
- Trained DRL agents (PPO, A2C) — no scripted bots
- Two reward **personas** (Maximizer, Efficiency)
- Automatic metrics (CSV) + plots
- Reproducibility: seeds, pinned deps, clear commands

## Project Structure
```
envs/
  game_2048_env.py        # Gymnasium env for 2048 with persona support
  web_workflow_env.py     # Selenium-based Gym env (stub; HTML in web_app/)
src/
  train.py                # Train PPO/A2C on 2048
  eval.py                 # Evaluate and log metrics
configs/
  algo_ppo.yaml
  algo_a2c.yaml
  persona_maximizer.yaml
  persona_efficiency.yaml
web_app/
  page1.html              # Step 1 → Step 2
  page2.html              # Step 2 → Success
  success.html
  error.html
notebooks/
  analysis.ipynb          # (optional) plots & analysis
models/                   # saved model artifacts
logs/                     # CSV logs
```

## Quick Start
```bash
python -m venv .venv
source .venv/bin/activate  # (Windows: .venv\Scripts\activate)

pip install -r requirements.txt

python src/train.py --algo ppo --env 2048 --timesteps 200000 --seed 7 --persona maximizer
python src/eval.py --model models/ppo_2048_seed7.zip --episodes 20 --persona maximizer
```
This writes `logs/2048_metrics.csv`.

## Personas (Reward Designs)
- **maximizer**: reward = score gain per step
- **efficiency**: reward = (score gain) / (moves + 1)

## Reproducibility
```bash
python src/train.py --algo ppo --env 2048 --timesteps 200000 --seed 7 --persona maximizer
python src/train.py --algo a2c --env 2048 --timesteps 200000 --seed 7 --persona efficiency
python src/eval.py  --model models/ppo_2048_seed7.zip --episodes 50 --persona maximizer
```

## Web Workflow (second app)
`web_app/` contains a **3-step static HTML flow**. `envs/web_workflow_env.py` is a Selenium wrapper scaffold.
Training on this is optional for the first pass; it demonstrates framework portability.

## Share on GitHub
```bash
git init
git add .
git commit -m "Initial commit: DRL 2048 + Web Workflow"
git branch -M main
git remote add origin <YOUR_GITHUB_REPO_URL>
git push -u origin main
```
Then add your partner as a collaborator in **Settings → Collaborators**.
