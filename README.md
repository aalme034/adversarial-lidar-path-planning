[![arXiv](https://img.shields.io/badge/arXiv-2501.XXXXX-B31B1B.svg)](https://arxiv.org/abs/2501.XXXXX)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.XXXXXXX.svg)](https://doi.org/10.5281/zenodo.XXXXXXX)
# adversarial-lidar-path-planning

Lightweight, fully reproducible framework showing tiny adversarial LiDAR perturbations (±3 cells) crash learned robotic path planners from 96% to 65% success. Defends with adversarial training (91.4%) + 5-frame LSTM (94.2%). Pure NumPy/PyTorch • Ideal educational benchmark for robotics security.

# Adversarial Perturbations on Sensor Data for Compromising Robotic Path Planning in Simulated Environments

[![Paper](https://img.shields.io/badge/pdf-arXiv-B31B1B?logo=arxiv&logoColor=white)](Robotics_Paper.pdf)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.9+](https://img.shields.io/badge/python-3.9%2B-blue)](https://www.python.org/downloads/)

**A lightweight, fully reproducible framework showing that tiny adversarial perturbations on 8-ray LiDAR data can crash learned robotic path planners — and how to defend them.**

Official code for the paper:  
**Adversarial Perturbations on Sensor Data for Compromising Robotic Path Planning in Simulated Environments**  
Alejandro Almeida¹², Daniel Aviles Rueda²  
¹Analytics for Cyber Defense Lab – ²Florida International University  
Accepted version (2025)

<img width="3010" height="1119" alt="Test Success Rate Across Runs with PGD (1)" src="https://github.com/user-attachments/assets/697d2ea8-d515-40b5-b4a5-ac3e607ba848" />


## Key Results

| Model                  | Clean Success | PGD-10 ε=0.30 Success | Δ      |
|-----------------------|---------------|-----------------------|--------|
| MLP (baseline)        | 96.2%         | 68.1%                 | -28.1% |
| MLP (adv-trained)     | 93.2%         | 91.4%                 | -1.8%  |
| LSTM (adv-trained)    | 97.1%         | **94.2%**             | -2.9%  |

**A simple 5-frame LSTM + adversarial training recovers almost all robustness** (p=0.012 vs MLP).

## Features

- Pure NumPy/PyTorch 2D grid-world (20×20 default, easily scalable)
- 8-ray LiDAR (45° steps, 10-cell range)
- Static + dynamic obstacles (5% move probability)
- Imitation learning from A* optimal trajectories
- White-box PGD attacks on-the-fly (ℓ∞, ε≤0.30 ≈ ±3 grid cells)
- TRADES-style adversarial training
- LSTM temporal defense (stacks last 5 observations)
- Full statistical evaluation over 500 random maps with fixed seeds
- < 1000 lines of clean code → ideal for teaching & research


<img width="323" height="300" alt="correct_pathing" src="https://github.com/user-attachments/assets/03364f4a-133e-48ae-9510-6a8b5fb51ba2" />


## Quick Start

```bash
git clone https://github.com/yourusername/adversarial-lidar-path-planning.git
cd adversarial-lidar-path-planning
pip install -r requirements.txt



