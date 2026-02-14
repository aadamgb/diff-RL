# Differentiable Learning for 2D Bicopter Control

A PyTorch-based differentiable framework for training neural network policies to control a 2D bicopter (dual-rotor drone) model. The project supports multiple control modes and includes trajectory tracking  with visualization.

## Demo

![Bicopter Control Visualization](/media/sim.mp4)


## Overview

This project implements end-to-end learning of bicopter control policies using:
- **Differentiable dynamics**: A 2D bicopter physics simulator ([`dynamics/bicopter_dynamics.py`](dynamics/bicopter_dynamics.py))
- **Random trajectory generation**: Harmonic-based target trajectories ([`utils/rand_traj_gen.py`](utils/rand_traj_gen.py))
- **Neural network policies**: Simple feedforward policy networks ([`utils/nn.py`](utils/nn.py))
- **Interactive visualization**: Real-time trajectory rendering with Pygame ([`utils/renderer.py`](utils/renderer.py))

## Features

### Control Modes
The framework supports three different control abstractions:
- **SRT** (Single Rotor Thrust): Direct control of individual rotor thrusts (T₁, T₂)
- **CTBR** (Collective Thrust & Body Rate): Control total thrust and angular rate
- **LV** (Linear Velocity): Velocity tracking with geometric attitude control

### Training
- Multi-environment parallel training (GPU-accelerated)
- Chunk-based loss computation for efficient memory usage
- Configurable randomization via Hydra ([`cfg/dynamics/bicopter.yaml`](cfg/dynamics/bicopter.yaml))

### Evaluation
- Single rollout policy testing
- Multi-policy comparison visualization
- Real-time drone dynamics and thrust visualization

## Project Structure

```
├── dynamics/
│   └── bicopter_dynamics.py          # 2D bicopter physics engine
├── utils/
│   ├── nn.py                         # Neural network policy
│   ├── rand_traj_gen.py              # Harmonic trajectory generator
│   ├── renderer.py                   # Pygame visualization
│   └── randomizer.py                 # Domain randomization utilities
├── cfg/
│   └── dynamics/bicopter.yaml        # Physics parameters & randomization config
├── train.py                          # Training script
├── test_model.py                     # Policy evaluation & visualization
└── outputs/                          # Saved model checkpoints
```

## Usage

### Training
```bash
python train.py
```
Trains a policy for the control mode specified in [`train.py`](train.py) (default: `lv`). Saves model to `outputs/{control_mode}.pt`.

### Evaluation & Visualization
```bash
python test_model.py
```
Loads trained policies and renders them in an interactive Pygame window. Supports comparison of multiple control modes simultaneously.

## Key Components

- **[`BicopterDynamics`](dynamics/bicopter_dynamics.py)**: State-space model with 6D state [x, y, vx, vy, θ, ω] and thrust-based actuation
- **[`BicopterPolicy`](utils/nn.py)**: Shallow policy network mapping 9D observations to actions
- **[`RandomTrajectoryGenerator`](utils/rand_traj_gen.py)**: Generates smooth multi-harmonic reference trajectories
- **[`MultiTrajectoryRenderer`](utils/renderer.py)**: Visualizes drone states, thrust vectors, and target tracking

## Requirements

- PyTorch
- Pygame
- Hydra
- NumPy
- Matplotlib
