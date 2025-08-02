import os
from numba import cuda
if cuda.is_available():
    import jax
else:
    os.environ["JAX_PLATFORMS"] = 'cpu'
    import jax
    jax.default_device('cpu')
import equinox as eqx
import jax.random as jr
import jax.nn as jnn
import jax.numpy as jnp


class Linear(eqx.Module):
    weight: jnp.ndarray
    bias: jnp.ndarray

    def __init__(self, in_features, out_features, key=jr.key(0)):
        initializer = jax.nn.initializers.glorot_normal()
        self.weight = initializer(key, (out_features, in_features), jnp.float32)
        self.bias = jnp.zeros(out_features, dtype=jnp.float32)

    def __call__(self, x):
        out = x @ self.weight.T + self.bias
        return out


class MLP(eqx.Module):
    layers: list

    def __init__(self, in_features, out_features, hidden_features, depth, key=jr.key(0), act_fun=jnn.relu, final_act=None):
        keys = jr.split(key, depth + 1)
        self.layers = [Linear(in_features, hidden_features, key=keys[0]), act_fun]
        for j in range(1, depth):
            self.layers += [Linear(hidden_features, hidden_features, key=keys[j]), act_fun]
        self.layers += [Linear(hidden_features, out_features, key=keys[-1])]
        if final_act is not None:
            self.layers.append(final_act)

    def __call__(self, x):
        for l in self.layers:
            x = l(x)
        return x

class LSTM(eqx.Module):
    cell: eqx.nn.LSTMCell

    def __init__(self, **kwargs):
        self.cell = eqx.nn.LSTMCell(**kwargs)

    def __call__(self, xs):
        scan_fn = lambda state, input: (self.cell(input, state), None)
        init_state = (jnp.zeros(self.cell.hidden_size),
                      jnp.zeros(self.cell.hidden_size))
        (h, c), _ = jax.lax.scan(scan_fn, init_state, xs)
        return h, c


class GRU(eqx.Module):
    cell: eqx.nn.GRUCell

    def __init__(self, **kwargs):
        self.cell = eqx.nn.GRUCell(**kwargs)

    def __call__(self, xs):
        def scan_fn(state, input):
            state_new = self.cell(input, state)
            return state_new, state_new
        init_state = jnp.zeros((self.cell.hidden_size,))
        final_state, out_seq = jax.lax.scan(scan_fn, init_state, xs)
        return final_state, out_seq