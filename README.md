# DashBot Reimplementation

DashBot reproduction theo paper **"DashBot: Insight-Driven Dashboard Generation Based on Deep Reinforcement Learning"**.

Project nay tach hai muc tieu:

- **Research reproduction**: MDP environment, reward engine, constrained sampling, A3C actor-critic, rollout training.
- **Product demo**: FastAPI backend va frontend HTML/CSS/JS tach file, realtime A3C dashboard recommendation tu CSV upload.

## Structure

```text
backend/dashbot/
  api/        FastAPI endpoints
  core/       data profiler, insight detector, chart generator, A3C/greedy recommenders
  rl_env/     DashboardEnv, rewards, constrained sampling
  agent/      PyTorch Bi-LSTM actor-critic va trainer shell
frontend/
  index.html
  css/styles.css
  js/app.js
  js/api.js
configs/default.yaml
scripts/prepare_data.py
scripts/evaluate.py
scripts/train.py
tests/
```

## Paper Assumptions

| Component | Value |
|---|---|
| `alpha` | `3.0` |
| `n_best` | `4` |
| `n_max` | `8` |
| correlation threshold | `0.5` |
| top/bottom k | `5` |
| optimizer | Adam |
| learning rate | `1e-4` |
| Bi-LSTM hidden size | `128` per direction |
| entropy coefficient | `0.01` |
| value loss coefficient | `0.5` |
| gamma | `1.0` |
| max columns | `10` |
| max episode steps | `50` |
| training steps | `500000` |

## Setup

Install dependencies:

```bash
pip install -r requirements.txt
```

Set `PYTHONPATH` before running backend scripts.

Windows PowerShell:

```powershell
$env:PYTHONPATH='backend'
```

macOS/Linux terminal:

```bash
export PYTHONPATH=backend
```

Windows CMD:

```bat
set PYTHONPATH=backend
```

## Run Tests

After setting `PYTHONPATH`:

```bash
python -m pytest -q
```

## Evaluate Greedy Baseline

```bash
python scripts/evaluate.py data/processed/cars.csv
```

## Run Backend API

After setting `PYTHONPATH`:

```bash
python -m uvicorn dashbot.api.main:app --host 127.0.0.1 --port 8010
```

Backend URL:

```text
http://127.0.0.1:8010
```

Health check:

```text
http://127.0.0.1:8010/api/health
```

Recommendation API defaults to A3C inference:

```text
POST http://127.0.0.1:8010/api/recommend?max_charts=5&mode=a3c&search_steps=1000
```

Use `mode=greedy` only as a baseline comparison for the report.

## Run Frontend

Open this file in a browser:

```text
frontend/index.html
```

Then upload a CSV file. Recommended demo files:

- `data/processed/cars.csv`
- `data/processed/movies.csv`
- `data/processed/seattle-weather.csv`
- `data/processed/penguins.csv`

The frontend calls:

```text
POST http://127.0.0.1:8010/api/recommend?max_charts=5&mode=a3c&search_steps=1000
```

## Current Implementation Status

Done:

- Data profiler: Q/N/T inference, cardinality, entropy, Gini, numeric stats.
- Insight detector: distribution, trend, correlation, top/bottom k, co-correlation, comparison.
- Reward engine: diversity, parsimony, insight reward, reward delta.
- Dashboard environment: `reset`, `step`, `change/add/remove/terminate`.
- Constrained sampling masks.
- Greedy baseline recommender for comparison.
- Realtime A3C recommender: feature tensor, actor-critic policy, constrained sampling, env rollout, best-dashboard selection.
- FastAPI profile/recommend endpoints.
- PyTorch Bi-LSTM actor-critic.
- State feature encoder: `DashboardState -> torch.Tensor[max_charts, feature_size]`.
- Policy sampler with constrained action/parameter sampling.
- Actor-critic training loop in `scripts/train.py`.
- Asynchronous A3C training loop in `scripts/train_a3c.py`.
- Paper ablation baselines: `DashBot-ind.`, `DashBot-pen.`, and DQN training logs.
- Demo checkpoint at `backend/dashbot/weights/dashbot_actor_critic.pth`.

Next research step:

- validate training quality on all 27 Vega datasets;
- run multiple seeds for paper-style mean/std ablation curves;
- tune constraints and reward constants based on generated dashboards.

## Train Agent

Prepare cleaned data first:

```bash
python scripts/prepare_data.py
```

Install PyTorch first, then run:

```bash
python scripts/train.py --steps 500000
```

By default, training uses the 27-file Vega manifest:

```text
data/processed/vega_27_manifest.txt
```

For a smoke test:

```bash
python scripts/train.py --steps 20 --rollout-length 5
```

For the current demo checkpoint:

```bash
python scripts/train.py --steps 200 --rollout-length 20 --save-path backend/dashbot/weights/dashbot_actor_critic.pth
```

## Paper Ablation Study

Use these commands to reproduce the Fig. 6-style learning curves. Each command writes a separate CSV log and checkpoint, so the models are not mixed.

Full DashBot A3C:

```bash
python scripts/train_a3c.py --variant dashbot --steps 500000 --workers 4 --rollout-length 50 --learning-rate 1e-4 --entropy-coef 0.01 --hidden-size 128 --log-interval 5000 --log-csv reports/ablation/training_curve_dashbot.csv --checkpoint-interval 50000 --checkpoint-dir backend/dashbot/weights/ablation/checkpoints_dashbot --save-path backend/dashbot/weights/ablation/dashbot_actor_critic.pth
```

DashBot-ind. without sequential prediction blocks:

```bash
python scripts/train_a3c.py --variant dashbot-ind --steps 500000 --workers 4 --rollout-length 50 --learning-rate 1e-4 --entropy-coef 0.01 --hidden-size 128 --log-interval 5000 --log-csv reports/ablation/training_curve_dashbot_ind.csv --checkpoint-interval 50000 --checkpoint-dir backend/dashbot/weights/ablation/checkpoints_dashbot_ind --save-path backend/dashbot/weights/ablation/dashbot_ind_actor_critic.pth
```

DashBot-pen. without constrained sampling, using penalties for invalid choices:

```bash
python scripts/train_a3c.py --variant dashbot-pen --steps 500000 --workers 4 --rollout-length 50 --learning-rate 1e-4 --entropy-coef 0.01 --hidden-size 128 --invalid-penalty -1.0 --log-interval 5000 --log-csv reports/ablation/training_curve_dashbot_pen.csv --checkpoint-interval 50000 --checkpoint-dir backend/dashbot/weights/ablation/checkpoints_dashbot_pen --save-path backend/dashbot/weights/ablation/dashbot_pen_actor_critic.pth
```

DQN baseline:

```bash
python scripts/train_dqn.py --steps 500000 --learning-rate 1e-4 --hidden-size 128 --batch-size 64 --target-update-interval 5000 --log-interval 5000 --log-csv reports/ablation/training_curve_dqn.csv --checkpoint-interval 50000 --checkpoint-dir backend/dashbot/weights/ablation/checkpoints_dqn --save-path backend/dashbot/weights/ablation/dashbot_dqn.pth
```

Plot all four curves:

```bash
python scripts/plot_paper_figures.py learning-curve --dashbot-log reports/ablation/training_curve_dashbot.csv --dashbot-ind-log reports/ablation/training_curve_dashbot_ind.csv --dashbot-pen-log reports/ablation/training_curve_dashbot_pen.csv --dqn-log reports/ablation/training_curve_dqn.csv --output reports/fig6_ablation_learning_curve.png
```

For paper-style repeated runs, run each model with different `--seed` values and different log filenames, then pass all logs with `--dashbot-logs`, `--dashbot-ind-logs`, `--dashbot-pen-logs`, and `--dqn-logs`. The plot script will draw the mean line and standard-deviation band across runs.

For a quick smoke test, replace `--steps 500000` with `--steps 100` and `--checkpoint-interval 0`.

## Demo Flow

1. Start the API:

```bash
python -m uvicorn dashbot.api.main:app --host 127.0.0.1 --port 8010
```

2. Open `frontend/index.html` in a browser.
3. Upload any clean tabular CSV. Recommended demo files:

- `data/processed/cars.csv`
- `data/processed/movies.csv`
- `data/processed/seattle-weather.csv`
- `data/processed/penguins.csv`

The frontend sends the CSV to `POST /api/recommend?mode=a3c&search_steps=1000` on `http://127.0.0.1:8010`, then renders the returned dashboard charts. The backend response includes `method`, `model_loaded`, and `search_steps`; for the normal demo these should show `a3c`, `true`, and the search quota used.

Quick Windows PowerShell demo commands:

```powershell
$env:PYTHONPATH='backend'
python scripts/prepare_data.py
python -m pytest -q
python -m uvicorn dashbot.api.main:app --host 127.0.0.1 --port 8010
```

Quick macOS/Linux terminal demo commands:

```bash
export PYTHONPATH=backend
python scripts/prepare_data.py
python -m pytest -q
python -m uvicorn dashbot.api.main:app --host 127.0.0.1 --port 8010
```
