# Decisions

Short log of choices, so neither AI re-opens a settled question. Newest first.

## 2026-11-14 — Python version: 3.10 or 3.11, not 3.13
Use **Python 3.10** (3.11 is also fine). Reason: the Track B stack (PyTorch cu118, ultralytics, and roboflow/sports' `umap-learn` → `numba`/`llvmlite`) has the most reliable Windows wheels on 3.10/3.11. 3.12 usually works. **Avoid 3.13 for now** — some transitive wheels still lag on it and you'd hit build errors. 3.13 is not worth the debugging time for this project.

## 2026-11-14 — CUDA build: cu118
Driver is 526.98, which is too old for CUDA 12.x minor-version compatibility. Install the **cu118** PyTorch build (`--index-url https://download.pytorch.org/whl/cu118`). Alternative: update the NVIDIA driver to ≥527.41, then cu124 also works. ChatGPT was right on this.

## 2026-11-14 — Models: roboflow/sports, not generic YOLO
Use the **football-trained** models from `roboflow/sports` (players + goalkeepers + referees + ball, plus pitch-keypoint detection for homography). Generic COCO YOLO has no "ball"/"referee" split and no pitch keypoints, so it can't map pixels to real pitch coordinates. This matters for hot areas and xG geometry.

## 2026-11-14 — Collaboration: shared GitHub repo
Claude and ChatGPT don't talk directly. This repo is the shared office. Rizki moves work and messages between them through it. `collab/status.md` is the always-current brief to paste into ChatGPT.

## 2026-11-14 — xG model
Logistic regression on 5,606 non-penalty StatsBomb open-data shots (WC 2018/2022, Euro 2020/2024). Features: distance, goal-mouth angle, header, direct free kick. Penalty = 0.76 fixed. Coefficients frozen in `src/xg_model.py`. Broadcast-level only — no keeper position or pressure, so treat single-shot values as rough.
