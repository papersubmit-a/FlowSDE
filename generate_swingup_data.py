from dm_control import suite
import numpy as np
import jax
import jax.numpy as jnp
# from interpax import interp1d
from tqdm import tqdm

# def time_function_exp(scale, shape):
#     dt = np.random.exponential(scale, shape)
#     ts = np.concatenate([np.zeros((shape[0], 1)), np.cumsum(dt, axis=-1)], axis=-1)
#     return ts
#
# def interpolate(tp, ts, ys):
#     yp = interp1d(tp, ts, ys, method='cubic')
#     return yp

def generate_acrobot_data(episodes=100, steps=101, sigma=0):
    env = suite.load(domain_name="acrobot", task_name="swingup")
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
    action_spec = env.action_spec()
    for _ in range(time_steps):
        # Action: scalar torque [-1, 1]
        action = np.random.uniform(action_spec.minimum, action_spec.maximum)

        env.step(action)

        # Get raw state
        theta1, theta2 = physics.data.qpos  # joint angles
        theta_dot1, theta_dot2 = physics.data.qvel  # joint angular velocity

        state = np.array([theta1, theta_dot1, theta2, theta_dot2])
        states.append(state)
        actions.append(action)
    return np.array(states), np.array(actions)

ts, states, actions = generate_acrobot_data(episodes=2200, steps=501, sigma=0.1)
np.save('acrobot_noisy_train.npy', np.concatenate([ts[:2000][..., None], states[:2000], actions[:2000]], axis=-1))
np.save('acrobot_noisy_test.npy', np.concatenate([ts[2000:][..., None], states[2000:], actions[2000:]], axis=-1))