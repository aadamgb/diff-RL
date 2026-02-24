# Expanding Dynamics

## Branch Description

This branch expands the quadcopter model to include the motor dynamics and drag. However, I've noticed that backpropagating through this expanded dynamics model can lead to exploding gradients, particularly for the Linear Velocity (LV) control mode. For this reason, only a simplified point mass model (PMM) with drag is backpropagated. This is implemented in [`dynamics/gradient_shaping.py`](dynamics/gradient_shaping.py) and [`dynamics/bicopter_dynamics.py`](dynamics/bicopter_dynamics.py).


The complete model including motor dynamics and aerodynamic drag:
$$ x_{t+1} = f_{\text{full}}(x_t, u_t)$$


During backpropagation, gradients flow through a simplified model:

$$ \frac {\partial x_{t+1}}{\partial x_t} \approx \frac {\partial f_{\text{simple}}(x_t, u_t)}{\partial x_t}$$



### Forward Pass: Full Dynamics Model


The motor dynamics are modeled as follow

$$ T_{cmd} = \pi_{\theta}(Observation) $$

$$\Omega^{\text{cmd}} = \sqrt{\frac{T^{\text{cmd}}}{k_1}}$$

$$\dot{\Omega}_{\min} \leq \frac{\Omega^{\text{cmd}} - \Omega}{k_m} \leq \dot{\Omega}_{\max}$$

$$\Omega_{t+1} = \Omega_t + \dot{\Omega} \Delta t$$

$$T = k_1 \Omega^2, \quad \tau = l(T_2 - T_1)$$



The drag forces are calculated using a quadratic drag model in the body frame. World-frame velocities are rotated to the aircraft body frame using the pitch angle $\theta$. The Quadratic drag is computed as $F_{\text{drag}} = \frac{1}{2} \rho C_D A v_{\text{norm}} v$, where:
    - $\rho$ is air density
    - $C_D$ are drag coefficients 
    - $A$ are projected areas 
    - $v_{\text{norm}} = \sqrt{v_x^2 + v_y^2}$ is velocity magnitude

Then, the drag forces are rotated back to world coordinates

$$F_{\text{drag},x} = F_{\text{drag},x}^{\text{body}} \cos(\theta) - F_{\text{drag},y}^{\text{body}} \sin(\theta)$$
$$F_{\text{drag},y} = F_{\text{drag},x}^{\text{body}} \sin(\theta) + F_{\text{drag},y}^{\text{body}} \cos(\theta)$$

*Note: Rotational body drag and propeller drag are neglected.*

Finally the translational and rotational accelaerations are:
$$a_x = \frac{-\sin(\theta) T - F_{\text{drag},x}}{m}, \quad a_y = \frac{\cos(\theta) T - mg - F_{\text{drag},y}}{m}$$

$$\alpha = \frac{\tau}{J}$$


### Backward Pass: Simplified Point Mass Model
The simplified PMM is the same as in the [main branch](https://github.com/aadamgb/diff-RL/blob/2e099ba4b1b89cf6cd7fb29d36752619c420b8c6/dynamics/bicopter_dynamics.py#L34), which excludes motor dynamics and drag forces.
