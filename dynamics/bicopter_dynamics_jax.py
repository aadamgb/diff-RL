import jax
import jax.numpy as jnp


def create_bicopter_params(cfg):
    return {
        "l": jnp.array(cfg.arm_l.nominal),
        "m": jnp.array(cfg.mass.nominal),
        "J": jnp.array(cfg.J.nominal),
        "g": jnp.array(cfg.g),
        "C_Dx": jnp.array(cfg.C_D.x.nominal),
        "C_Dy": jnp.array(cfg.C_D.y.nominal),
        "rho": jnp.array(cfg.rho),
        "k1": jnp.array(cfg.thrust_map.k1.nominal),
        "km_up": jnp.array(cfg.km.up.nominal),
        "km_down": jnp.array(cfg.km.down.nominal),
        "Omega_min": jnp.array(cfg.motor_speed_min),
        "Omega_max": jnp.array(cfg.motor_speed_max),
        "Omega_dot_min": jnp.array(cfg.motor_acc_min),
        "Omega_dot_max": jnp.array(cfg.motor_acc_max),
        "Ti_max": jnp.array(cfg.Ti_max),
        "eps": 1e-3,
    }



def full_dynamics(state, action, params, dt):

    x, y, vx, vy, theta, omega, Omega1, Omega2 = state

    T1_cmd = action[0]
    T2_cmd = action[1]

    T1_cmd = jnp.clip(T1_cmd, 0.0, params["Ti_max"])
    T2_cmd = jnp.clip(T2_cmd, 0.0, params["Ti_max"])

    Omega1_cmd = jnp.sqrt(jnp.maximum(T1_cmd / params["k1"], params["eps"]))
    Omega2_cmd = jnp.sqrt(jnp.maximum(T2_cmd / params["k1"], params["eps"]))

    km1 = jnp.where(Omega1_cmd > Omega1, params["km_up"], params["km_down"])
    km2 = jnp.where(Omega2_cmd > Omega2, params["km_up"], params["km_down"])

    Omega1_dot = jnp.clip(
        (Omega1_cmd - Omega1) / km1,
        params["Omega_dot_min"],
        params["Omega_dot_max"]
    )

    Omega2_dot = jnp.clip(
        (Omega2_cmd - Omega2) / km2,
        params["Omega_dot_min"],
        params["Omega_dot_max"]
    )

    Omega1 = Omega1 + Omega1_dot * dt
    Omega2 = Omega2 + Omega2_dot * dt

    T1 = params["k1"] * Omega1**2
    T2 = params["k1"] * Omega2**2

    T = T1 + T2
    tau = params["l"] * (T2 - T1)

    drag_x, drag_y = calculate_drag(vx, vy, theta, params)

    ax = (-jnp.sin(theta) * T - drag_x) / params["m"]
    ay = ( jnp.cos(theta) * T - params["m"] * params["g"] - drag_y) / params["m"]

    alpha = tau / params["J"]

    vx = vx + ax * dt
    vy = vy + ay * dt
    omega = omega + alpha * dt

    x = x + vx * dt
    y = y + vy * dt
    theta = theta + omega * dt

    return jnp.array([x, y, vx, vy, theta, omega, Omega1, Omega2])


def simplified_dynamics(state, action, params, dt):

    x, y, vx, vy, theta, omega, _, _ = state

    T1 = action[0]
    T2 = action[1]

    T = T1 + T2
    tau = params["l"] * (T2 - T1)

    ax = -jnp.sin(theta) * T / params["m"]
    ay =  jnp.cos(theta) * T / params["m"] - params["g"]

    alpha = tau / params["J"]

    vx = vx + ax * dt
    vy = vy + ay * dt
    omega = omega + alpha * dt

    x = x + vx * dt
    y = y + vy * dt
    theta = theta + omega * dt

    return jnp.array([x, y, vx, vy, theta, omega, 0.0, 0.0])

def calculate_drag(vx, vy, theta, params):
    cos_theta = jnp.cos(theta)
    sin_theta = jnp.sin(theta)

    vx_body = vx * cos_theta + vy * sin_theta
    vy_body = -vx * sin_theta + vy * cos_theta

    v_norm = jnp.sqrt(vx_body**2 + vy_body**2 + 1e-6)

    area_x = params["l"] * 0.1
    area_y = params["l"]

    drag_x_body = 0.5 * params["rho"] * params["C_Dx"] * area_x * v_norm * vx_body
    drag_y_body = 0.5 * params["rho"] * params["C_Dy"] * area_y * v_norm * vy_body

    drag_x = drag_x_body * cos_theta - drag_y_body * sin_theta
    drag_y = drag_x_body * sin_theta + drag_y_body * cos_theta

    return drag_x, drag_y


@jax.custom_vjp
def step(state, action, params, dt):
    return full_dynamics(state, action, params, dt)

def step_fwd(state, action, params, dt):
    next_state = full_dynamics(state, action, params, dt)
    return next_state, (state, action, params, dt)

def step_bwd(residual, g):

    state, action, params, dt = residual

    def pmm_only(s, a):
        return simplified_dynamics(s, a, params, dt)

    # Jacobians from simplified model
    dstate = jax.jacobian(pmm_only, argnums=0)(state, action)
    daction = jax.jacobian(pmm_only, argnums=1)(state, action)

    grad_state = dstate.T @ g
    grad_action = daction.T @ g

    return grad_state, grad_action, None, None


step.defvjp(step_fwd, step_bwd)