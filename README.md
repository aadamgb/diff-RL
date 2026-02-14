# Differentiable Learning for 2D Bicopter Control

A simple PyTorch-based differentiable framework for training neural network policies to control a 2D bicopter. The project supports three control modes and real-time visualization in Pygame.



<div align="center">

![Bicopter Control Visualization](/media/sim.gif)

</div>



## Project Structure

```
├── cfg/
│   └── dynamics/bicopter.yaml        # Physics params & randomization config
├── dynamics/
│   └── bicopter_dynamics.py          # 2D bicopter physics model
├── utils/
│   ├── nn.py                         # Neural network policy
│   ├── rand_traj_gen.py              # Harmonic trajectory generator
│   ├── renderer.py                   # Pygame visualization
│   └── randomizer.py                 # Domain randomization 

├── train.py                          # Training script
├── test_model.py                     # Policy evaluation & visualization
└── outputs/                          # Saved models
```

## Overview
The aim of the bicopetrs is to match the state $\mathbf{x}_{ref} = $[p, v, acc ] of a moving target whose trajectory is sampled from the sum of random sinusoidal harmonics. In this way, the loss function is calculated as

$$\mathcal{L} = \mathcal{L}_{pos}
+
\mathcal{L}_{vel}
+
0.25 \, \mathcal{L}_{\omega}
$$



$ \mathcal{L}_{pos} =\frac{1}{T} \sum_{t=1}^{T} ||\mathbf{p}_{t}^{(i)} - \mathbf{p}_{ref,t}^{(i)}|| ^2
 \quad \mathcal{L}_{vel} = \frac{1}{T} $

$\sum_{t=1}^{T} \left\|
\mathbf{v}_{t}^{(i)} -
\mathbf{v}_{ref,t}^{(i)}
\right\|^2 \quad \mathcal{L}_{\omega} =\frac{1}{T} \sum_{t=1}^{T}
 \omega_{t}^{(i)2}$

where T denotes the number of time steps in which the loss is acumulated. To minimize the loss a truncated backpropagation through time T-BPTT is applied with ADAM optimizer and learning rate $1e^{-3}$

### 🧠Neural Network
The neural network (policy) is constructed in [`utils/nn.py`](utils/nn.py) as a simple feedforward multi-layer perecptron (mlp). The policy takes 9 observational inputs which correspond to:
- Position error:  
    - $e_{px} = x_{ref} - x$
    - $e_{py} = y_{ref} - y$

- Velocity error:  
    - $e_{vx} = v_{x,ref} - v_x$
    - $e_{vy} = v_{y,ref} - v_y$

- Orientiation and body rate:
    - $\sin{\theta}$
    - $\cos{\theta}$
    - $\omega$

Hence the observation input is :
$$\mathbf{o} =
\begin{bmatrix}
e_{px} &
e_{py} &
e_{vx} &
e_{vy} &
a_{x,ref} &
a_{y,ref} &
\sin(\theta) &
\cos(\theta) &
\omega
\end{bmatrix}^T \ \in \R^9
$$

And it can ouput three different control modes:

Single rotor thrust (SRT)
Colective thrust and body rate (CTBR)
Linear velocities + geo gains (LV)


### 🚁Bicopter Dynamics 
The physical bicopter system is modeled in [`dynamics/bicopter_dynamics.py`](dynamics/bicopter_dynamics.py)

$$\mathbf{x} =
\begin{bmatrix}
x & y & v_x & v_y & \theta & \omega
\end{bmatrix}^T
$$

### 🏋️Differentiable Training
[`dynamics/bicopter_dynamics.py`](dynamics/bicopter_dynamics.py)



This project implements end-to-end learning of bicopter control policies using:
- **Differentiable dynamics**: A 2D bicopter physics simulator ([`dynamics/bicopter_dynamics.py`](dynamics/bicopter_dynamics.py))
- **Random trajectory generation**: Harmonic-based target trajectories ([`utils/rand_traj_gen.py`](utils/rand_traj_gen.py))
- **Neural network policies**: Simple feedforward policy networks ([`utils/nn.py`](utils/nn.py))
- **Interactive visualization**: Real-time trajectory rendering with Pygame ([`utils/renderer.py`](utils/renderer.py))



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
