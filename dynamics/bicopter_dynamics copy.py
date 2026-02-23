import jax
import jax.numpy as jnp

# Control mode codes
MODE_SRT = 0
MODE_CTBR = 1
MODE_LV = 2

#    T1       T2
#  _____    _____
#    |________|


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
        "eps": jnp.array(1e-4),
    }


def _get_control(state, action, params, dt, control_mode):
    def srt_mode(args):
        _, action_local = args
        return action_local[..., 0], action_local[..., 1]

    def ctbr_mode(args):
        state_local, action_local = args
        T_cmd = action_local[..., 0]
        omega_cmd = action_local[..., 1]
        tau_cmd = params["J"] * (omega_cmd - state_local[..., 5]) / dt
        T1 = 0.5 * (T_cmd - tau_cmd / params["l"])
        T2 = 0.5 * (T_cmd + tau_cmd / params["l"])
        return T1, T2

    def lv_mode(args):
        state_local, action_local = args
        vx, vy = state_local[..., 2], state_local[..., 3]
        theta, omega = state_local[..., 4], state_local[..., 5]

        kv = jnp.clip(action_local[..., 2], 0.0, 10.0)
        kR = jnp.clip(action_local[..., 3], 0.0, 20.0)
        kw = jnp.clip(action_local[..., 4], 0.0, 5.0)

        ax_des = kv * (action_local[..., 0] - vx)
        ay_des = kv * (action_local[..., 1] - vy)

        fx = params["m"] * ax_des
        fy = params["m"] * (ay_des + params["g"])
        f_des = jnp.stack([fx, fy], axis=-1)

        b = jnp.stack([-jnp.sin(theta), jnp.cos(theta)], axis=-1)
        T_cmd = jnp.sum(f_des * b, axis=-1)

        f_norm = jnp.linalg.norm(f_des, axis=-1) + 1e-6
        b_des = f_des / f_norm[..., None]
        theta_des = jnp.arctan2(-b_des[..., 0], b_des[..., 1])

        eR = jnp.sin(theta - theta_des)
        tau = params["J"] * (-kR * eR - kw * omega)

        T1 = 0.5 * (T_cmd - tau / params["l"])
        T2 = 0.5 * (T_cmd + tau / params["l"])
        return T1, T2

    return jax.lax.switch(
        control_mode,
        (srt_mode, ctbr_mode, lv_mode),
        (state, action),
    )


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


def full_dynamics(state, action, params, dt, control_mode):
    x, y, vx, vy, theta, omega, Omega1, Omega2 = jnp.split(state, 8, axis=-1)
    x = x[..., 0]
    y = y[..., 0]
    vx = vx[..., 0]
    vy = vy[..., 0]
    theta = theta[..., 0]
    omega = omega[..., 0]
    Omega1 = Omega1[..., 0]
    Omega2 = Omega2[..., 0]

    T1_cmd, T2_cmd = _get_control(state, action, params, dt, control_mode)
    T1_cmd = jnp.clip(T1_cmd, 0.0, params["Ti_max"])
    T2_cmd = jnp.clip(T2_cmd, 0.0, params["Ti_max"])

    Omega1_cmd = jnp.sqrt(jnp.maximum(T1_cmd / params["k1"], params["eps"]))
    Omega2_cmd = jnp.sqrt(jnp.maximum(T2_cmd / params["k1"], params["eps"]))

    km1 = jnp.where(Omega1_cmd > Omega1, params["km_up"], params["km_down"])
    km2 = jnp.where(Omega2_cmd > Omega2, params["km_up"], params["km_down"])

    Omega1_dot = jnp.clip(
        (Omega1_cmd - Omega1) / km1,
        params["Omega_dot_min"],
        params["Omega_dot_max"],
    )
    Omega2_dot = jnp.clip(
        (Omega2_cmd - Omega2) / km2,
        params["Omega_dot_min"],
        params["Omega_dot_max"],
    )

    Omega1 = Omega1 + Omega1_dot * dt
    Omega2 = Omega2 + Omega2_dot * dt

    T1 = params["k1"] * Omega1**2
    T2 = params["k1"] * Omega2**2

    T = T1 + T2
    tau = params["l"] * (T2 - T1)

    drag_x, drag_y = calculate_drag(vx, vy, theta, params)

    ax = (-jnp.sin(theta) * T - drag_x) / params["m"]
    ay = (jnp.cos(theta) * T - params["m"] * params["g"] - drag_y) / params["m"]

    alpha = tau / params["J"]

    vx = vx + ax * dt
    vy = vy + ay * dt
    omega = omega + alpha * dt

    x = x + vx * dt
    y = y + vy * dt
    theta = theta + omega * dt

    return jnp.stack([x, y, vx, vy, theta, omega, Omega1, Omega2], axis=-1)


def simplified_dynamics(state, action, params, dt, control_mode):
    x, y, vx, vy, theta, omega, _, _ = jnp.split(state, 8, axis=-1)
    x = x[..., 0]
    y = y[..., 0]
    vx = vx[..., 0]
    vy = vy[..., 0]
    theta = theta[..., 0]
    omega = omega[..., 0]

    T1_cmd, T2_cmd = _get_control(state, action, params, dt, control_mode)
    T1 = jnp.clip(T1_cmd, 0.0, params["Ti_max"])
    T2 = jnp.clip(T2_cmd, 0.0, params["Ti_max"])

    T = T1 + T2
    tau = params["l"] * (T2 - T1)

    ax = -jnp.sin(theta) * T / params["m"]
    ay = jnp.cos(theta) * T / params["m"] - params["g"]

    alpha = tau / params["J"]

    vx = vx + ax * dt
    vy = vy + ay * dt
    omega = omega + alpha * dt

    x = x + vx * dt
    y = y + vy * dt
    theta = theta + omega * dt

    zeros = jnp.zeros_like(x)
    return jnp.stack([x, y, vx, vy, theta, omega, zeros, zeros], axis=-1)


@jax.custom_vjp
def step(state, action, params, dt, control_mode):
    return full_dynamics(state, action, params, dt, control_mode)


def step_with_mode(state, action, params, dt, control_mode):
    mode_map = {
        "srt": MODE_SRT,
        "ctbr": MODE_CTBR,
        "lv": MODE_LV,
    }
    mode_code = mode_map.get(control_mode, control_mode)
    return step(state, action, params, dt, mode_code)


def step_fwd(state, action, params, dt, control_mode):
    next_state = full_dynamics(state, action, params, dt, control_mode)
    return next_state, (state, action, params, dt, control_mode)


def step_bwd(residual, g):
    state, action, params, dt, control_mode = residual

    def pmm_only(s_in, a_in):
        return simplified_dynamics(s_in, a_in, params, dt, control_mode)

    if state.ndim == 1:
        state_b = state[None, :]
        action_b = action[None, :]
        g_b = g[None, :]
        _, pullback = jax.vjp(pmm_only, state_b, action_b)
        grad_state_b, grad_action_b = pullback(g_b)
        grad_state = grad_state_b[0]
        grad_action = grad_action_b[0]
    else:
        _, pullback = jax.vjp(pmm_only, state, action)
        grad_state, grad_action = pullback(g)

    return grad_state, grad_action, None, None, None


step.defvjp(step_fwd, step_bwd)
        


    


