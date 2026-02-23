import os
import time
import math
import pickle
from pathlib import Path
import importlib.util

import jax
import jax.numpy as jnp
import optax
import matplotlib.pyplot as plt

import hydra
from omegaconf import DictConfig


class RandomTrajectoryGenerator:
    def __init__(self, num_envs, key, num_harmonics=5):
        self.num_envs = num_envs
        self.key = key
        self.num_harmonics = num_harmonics
        self.reset()

    def reset(self):
        self.key, k1, k2, k3 = jax.random.split(self.key, 4)
        self.amps = jax.random.uniform(
            k1, (self.num_envs, 2, self.num_harmonics), minval=0.5, maxval=1.5
        )
        self.freqs = jax.random.uniform(
            k2, (self.num_envs, 2, self.num_harmonics), minval=0.2, maxval=2.2
        )
        self.phases = jax.random.uniform(
            k3, (self.num_envs, 2, self.num_harmonics), minval=0.0, maxval=2.0 * jnp.pi
        )

    def get_target(self, t_float):
        t = jnp.full((self.num_envs, 1, 1), t_float)
        pos = jnp.sum(self.amps * jnp.sin(self.freqs * t + self.phases), axis=2)
        vel = jnp.sum(self.amps * self.freqs * jnp.cos(self.freqs * t + self.phases), axis=2)
        acc = jnp.sum(
            -self.amps * (self.freqs**2) * jnp.sin(self.freqs * t + self.phases),
            axis=2,
        )
        return pos, vel, acc

    def get_params(self):
        return self.amps, self.freqs, self.phases


def target_from_params(traj_params, t_float):
    amps, freqs, phases = traj_params
    t = jnp.full((amps.shape[0], 1, 1), t_float)
    pos = jnp.sum(amps * jnp.sin(freqs * t + phases), axis=2)
    vel = jnp.sum(amps * freqs * jnp.cos(freqs * t + phases), axis=2)
    acc = jnp.sum(-amps * (freqs**2) * jnp.sin(freqs * t + phases), axis=2)
    return pos, vel, acc


def env_randomization(cfg: DictConfig, num_envs, key):
    key, k = jax.random.split(key)
    c = jax.random.uniform(k, (num_envs,), minval=cfg.sf.min, maxval=cfg.sf.max)

    l = c * (cfg.arm_l.max - cfg.arm_l.min) + cfg.arm_l.min

    m = ((l**3 - cfg.arm_l.min**3) / (cfg.arm_l.max**3 - cfg.arm_l.min**3)) * (
        cfg.mass.max - cfg.mass.min
    ) + cfg.mass.min

    J = ((l**5 - cfg.arm_l.min**5) / (cfg.arm_l.max**5 - cfg.arm_l.min**5)) * (
        cfg.J.max - cfg.J.min
    ) + cfg.J.min

    C_Dx = ((l**2 - cfg.arm_l.min**2) / (cfg.arm_l.max**2 - cfg.arm_l.min**2)) * (
        cfg.C_D.x.max - cfg.C_D.x.min
    ) + cfg.C_D.x.min

    C_Dy = ((l**2 - cfg.arm_l.min**2) / (cfg.arm_l.max**2 - cfg.arm_l.min**2)) * (
        cfg.C_D.y.max - cfg.C_D.y.min
    ) + cfg.C_D.y.min

    k1 = cfg.thrust_map.k1.min * ((cfg.thrust_map.k1.max / cfg.thrust_map.k1.min) ** c)

    def add_noise(x, noise_key):
        noise = jax.random.uniform(noise_key, (num_envs,), minval=-cfg.nf, maxval=cfg.nf)
        return x * (1.0 + noise)

    key, k1_key, k2_key, k3_key, k4_key, k5_key, k6_key = jax.random.split(key, 7)

    return (
        {
            "l": add_noise(l, k1_key),
            "m": add_noise(m, k2_key),
            "J": add_noise(J, k3_key),
            "C_Dx": add_noise(C_Dx, k4_key),
            "C_Dy": add_noise(C_Dy, k5_key),
            "k1": add_noise(k1, k6_key),
        },
        key,
    )


def init_mlp_params(key, layer_sizes):
    params = []
    for in_size, out_size in zip(layer_sizes[:-1], layer_sizes[1:]):
        key, w_key = jax.random.split(key)
        w = jax.random.normal(w_key, (in_size, out_size)) * jnp.sqrt(2.0 / in_size)
        b = jnp.zeros((out_size,))
        params.append({"w": w, "b": b})
    return params


def mlp_apply(params, x):
    for i, layer in enumerate(params):
        x = x @ layer["w"] + layer["b"]
        if i < len(params) - 1:
            x = jax.nn.relu(x)
    return x


def compute_chunk_loss(
    policy_params,
    init_states,
    dyn_params,
    traj_params,
    chunk_start,
    chunk_len,
    dt,
    horizon,
    cm_code,
    step_fn,
):
    def body_fun(carry, t_idx):
        states_local, pos_error_sum, vel_error_sum, rate_penalty_sum = carry
        t = (chunk_start + t_idx) * dt
        pos_ref, vel_ref, acc_ref = target_from_params(traj_params, t)

        e_px = pos_ref[:, 0] - states_local[:, 0]
        e_py = pos_ref[:, 1] - states_local[:, 1]
        e_vx = vel_ref[:, 0] - states_local[:, 2]
        e_vy = vel_ref[:, 1] - states_local[:, 3]

        obs = jnp.stack(
            [
                e_px,
                e_py,
                e_vx,
                e_vy,
                acc_ref[:, 0],
                acc_ref[:, 1],
                jnp.sin(states_local[:, 4]),
                jnp.cos(states_local[:, 4]),
                states_local[:, 5],
            ],
            axis=1,
        )

        actions = mlp_apply(policy_params, obs)
        next_states = step_fn(states_local, actions, dyn_params, dt, cm_code)

        mask = t_idx < chunk_len
        mask_f = mask.astype(next_states.dtype)
        states_local = jnp.where(mask, next_states, states_local)

        pos_error_sum = pos_error_sum + mask_f * jnp.mean(
            jnp.sum((states_local[:, :2] - pos_ref) ** 2, axis=1)
        )
        vel_error_sum = vel_error_sum + mask_f * jnp.mean(
            jnp.sum((states_local[:, 2:4] - vel_ref) ** 2, axis=1)
        )
        rate_penalty_sum = rate_penalty_sum + mask_f * jnp.mean(states_local[:, 5] ** 2)
        return (states_local, pos_error_sum, vel_error_sum, rate_penalty_sum), None

    (states_final, pos_error_sum, vel_error_sum, rate_penalty_sum), _ = jax.lax.scan(
        body_fun,
        (init_states, 0.0, 0.0, 0.0),
        jnp.arange(horizon),
    )

    steps_in_chunk = jnp.asarray(chunk_len)
    pos_error = pos_error_sum / steps_in_chunk
    vel_error = vel_error_sum / steps_in_chunk
    rate_penalty = rate_penalty_sum / steps_in_chunk
    loss = pos_error + vel_error + 0.25 * rate_penalty
    return loss, states_final


@hydra.main(config_path="cfg/dynamics", config_name="bicopter", version_base=None)
def train(cfg: DictConfig):
    """
    ----------------------------------------------------------------------
    Main training function
    ----------------------------------------------------------------------
    """
    backend = jax.default_backend()
    num_envs = 1 if backend == "cpu" else 2024
    print(f"num_envs: {num_envs}")
    epochs = 1000
    steps  = 500
    horizon = 50 
    dt = 0.01

    ACT_DIMS = {
        "srt": 2,  # T1, T2
        "ctbr": 2, # T, omega
        "lv": 5    #  vx, vy, kv, kR, kw  
    }

    #================#
    # Control mode:
    cm = "ctbr" 
    #================#

    dyn_path = Path(__file__).parent / "dynamics" / "bicopter_dynamics copy.py"
    spec = importlib.util.spec_from_file_location("bicopter_dynamics_copy", dyn_path)
    bicopter_dynamics = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(bicopter_dynamics)

    traj_gen = RandomTrajectoryGenerator(num_envs=num_envs, key=jax.random.PRNGKey(0))
    base_params = bicopter_dynamics.create_bicopter_params(cfg)

    cm_map = {
        "srt": bicopter_dynamics.MODE_SRT,
        "ctbr": bicopter_dynamics.MODE_CTBR,
        "lv": bicopter_dynamics.MODE_LV,
    }
    cm_code = cm_map[cm]

    obs_dim = 9
    act_dim = ACT_DIMS[cm]
    key = jax.random.PRNGKey(42)
    key, init_key = jax.random.split(key)
    policy_params = init_mlp_params(init_key, [obs_dim, 64, 64, act_dim])

    optimizer = optax.adam(1e-3)
    opt_state = optimizer.init(policy_params)

    loss_history = []

    def update_chunk(
        policy_params,
        opt_state,
        states,
        dyn_params,
        traj_params,
        chunk_start,
        chunk_len,
        horizon,
        step_fn,
    ):
        (loss, next_states), grads = jax.value_and_grad(compute_chunk_loss, has_aux=True)(
            policy_params,
            states,
            dyn_params,
            traj_params,
            chunk_start,
            chunk_len,
            dt,
            horizon,
            cm_code,
            step_fn,
        )
        updates, opt_state = optimizer.update(grads, opt_state, policy_params)
        policy_params = optax.apply_updates(policy_params, updates)
        return policy_params, opt_state, next_states, loss

    update_chunk = jax.jit(update_chunk, static_argnames=("step_fn", "horizon"))

    for epoch in range(epochs):
        # Generate random target trajectory #
        traj_gen.reset()
        traj_params = traj_gen.get_params()

        # Randomize the environmental parameters per num_envs #
        env_params, key = env_randomization(cfg, num_envs, key)
        dyn_params = {**base_params, **env_params}

        # Initialize the bicopter state #
        states = jnp.zeros((num_envs, 8))
        key, pos_key = jax.random.split(key)
        states = states.at[:, :2].set(
            jax.random.uniform(pos_key, (num_envs, 2), minval=0.0, maxval=5.0)
        )
        hover_speed = jnp.sqrt(dyn_params["m"] * dyn_params["g"] / (2.0 * dyn_params["k1"]))
        states = states.at[:, 6].set(hover_speed)
        states = states.at[:, 7].set(hover_speed)

        # print(states[:, 7])
        # break                       

        epoch_loss = 0.0
        num_chunks = 0
        
        for chunk_start in range(0, steps, horizon):
            chunk_end = min(chunk_start + horizon, steps)
            chunk_len = chunk_end - chunk_start
            policy_params, opt_state, next_states, loss = update_chunk(
                policy_params,
                opt_state,
                states,
                dyn_params,
                traj_params,
                chunk_start,
                chunk_len,
                horizon,
                bicopter_dynamics.step,
            )

            loss_value = float(loss)
            epoch_loss += loss_value

            if math.isnan(loss_value):
                print("\n" + "=" * 50)
                print("ERROR: Loss became NaN!")
                print("Training stopped at iteration:", epoch)
                print("=" * 50)
                break

            states = jax.lax.stop_gradient(next_states)
            num_chunks += 1
        
        # Average loss over chunks for logging
        avg_loss = epoch_loss / num_chunks
        loss_history.append(avg_loss)

        if epoch % 100 == 0 or epoch == (epochs-1):
            print(f"Epoch {epoch} | Loss = {avg_loss:.3f}")


    # Plot the loss
    plt.figure()
    plt.plot(loss_history)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss")
    plt.grid(True)
    plt.show()

    # Export the model
    output_dir = os.path.join(os.path.dirname(__file__), "outputs")
    os.makedirs(output_dir, exist_ok=True)
    model_path = os.path.join(output_dir, f"{cm}4.pkl")
    with open(model_path, "wb") as f:
        pickle.dump(jax.device_get(policy_params), f)
    print("Policy saved!")


if __name__ == "__main__":
    start_time = time.time()
    train()
    end_time = time.time()

    training_duration = end_time - start_time
    print(f"\nTraining completed!")
    print(f"Total training time: {int(training_duration // 3600)}h {int((training_duration % 3600) // 60)}m {training_duration % 60:.2f}s")

