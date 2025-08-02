from dm_control import suite
import numpy as np
import jax
import jax.numpy as jnp
from interpax import interp1d

def time_function_exp(scale, shape):
    dt = np.random.exponential(scale, shape)
    ts = np.concatenate([np.zeros((shape[0], 1)), np.cumsum(dt, axis=-1)], axis=-1)
    return ts

def interpolate(tp, ts, ys):
    yp = interp1d(tp, ts, ys, method='cubic')
    return yp
from tqdm import tqdm

def generate_cartpole_data(episodes=100, steps=101, sigma=0):
    env = suite.load(domain_name="cartpole", task_name="balance")
    physics = env.physics
    dt = physics.timestep()
    ts = np.linspace(0, (steps-1)*dt, steps)
    states, actions = [], []
    for i in tqdm(range(episodes)):
        s, a = _generate_data(env, steps)
        if sigma > 0:
            s += np.random.normal(0, sigma, s.shape)
        states.append(s)
        actions.append(a)
    return np.repeat(ts[None], axis=0, repeats=episodes), np.array(states), np.array(actions)

def _generate_data(env, time_steps):
    states = []
    actions = []
    env.reset()
    physics = env.physics
    for _ in range(time_steps):
        # Action: scalar torque [-1, 1]
        action = np.random.uniform(-1, 1, size=(1,))

        env.step(action)

        # Get raw state
        x, theta = physics.data.qpos  # cart position, pole angle
        x_dot, theta_dot = physics.data.qvel  # cart velocity, pole angular velocity

        state = np.array([x, x_dot, theta, theta_dot])
        states.append(state)
        actions.append(action)

    return np.array(states), np.array(actions)

ts, states, actions = generate_cartpole_data(episodes=2200, steps=501, sigma=0.1)
np.save('cartpole_noisy_train.npy', np.concatenate([ts[:2000][..., None], states[:2000], actions[:2000]], axis=-1))
np.save('cartpole_noisy_test.npy', np.concatenate([ts[2000:][..., None], states[2000:], actions[2000:]], axis=-1))