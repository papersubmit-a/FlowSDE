import os
from typing import Callable, Literal
from numba import cuda
if cuda.is_available():
    import jax
else:
    os.environ["JAX_PLATFORMS"] = 'cpu'
    import jax
    jax.default_device('cpu')
import diffrax
import equinox as eqx
import jax.nn as jnn
import jax.numpy as jnp
import jax.random as jr
from interpax import interp1d

class NeuralODE(eqx.Module):
    net: eqx.nn.MLP
    model_name: str

    def __init__(self, input_dim, output_dim, cond_dim, hidden_dim, depth, *, key, model_name):
        if 'ft' in model_name:
            input_dim = input_dim + cond_dim + 1
        else:
            input_dim = input_dim + cond_dim
        if "ResNet" in model_name:
            self.net = ResFunc(
                in_size=input_dim, out_size=output_dim, width_size=hidden_dim, depth=depth, activation=jnn.softplus,
                key=key)
        else:
            self.net = eqx.nn.MLP(
                in_size=input_dim, out_size=output_dim, width_size=hidden_dim, depth=depth, activation=jnn.softplus,
                key=key)
        self.model_name = model_name

    @eqx.filter_jit
    def __call__(self, t, h, args):
        ts, us = args["ts"], args["us"]
        if isinstance(h, tuple):
            h = h[0]
        if us is not None:
            u_fun = lambda t: interp1d(t, ts, us, method="nearest", extrap=True)
            if 'ft' in self.model_name:
                inp = jnp.concatenate([h, u_fun(t), jnp.sin(t)[None]], axis=-1)
            else:
                inp = jnp.concatenate([h, u_fun(t)], axis=-1)
        else:
            if 'ft' in self.model_name:
                inp = jnp.concatenate([h, jnp.array([t])], axis=-1)
            else:
                inp = h
        hdot = self.net(inp)
        return hdot

    @eqx.filter_jit
    def generator(self, x0, ts, us, dt0=1e-2, adaptive=False, max_steps=4096):
        terms = diffrax.ODETerm(self.__call__)
        saveat = diffrax.SaveAt(ts=ts)
        if adaptive:
            sol = diffrax.diffeqsolve(terms, diffrax.Tsit5(), ts[0], ts[-1], dt0, x0,
                                      args={"ts": ts, "us": us}, saveat=saveat, max_steps=max_steps,
                                      stepsize_controller=diffrax.PIDController(rtol=1e-3, atol=1e-6)
                                      )
        else:
            sol = diffrax.diffeqsolve(terms, diffrax.Tsit5(), ts[0], ts[-1], dt0, x0,
                                      args={"ts": ts, "us": us}, saveat=saveat, max_steps=max_steps
                                      )
        return sol.ys

    @eqx.filter_jit
    def solve_ode(self, xs, ts, us, dt0=1e-2, adaptive=False, max_steps=4096):
        y_preds = jax.vmap(lambda x, t, u: self.generator(x, t, u, dt0=dt0, adaptive=adaptive, max_steps=max_steps),
                           in_axes=(0, 0, 0 if us is not None else None))(xs[:, 0], ts, us)
        return y_preds

    @eqx.filter_jit
    def train(self, xs, ts, us, ms=False, dt0=1e-2, adaptive=False, max_steps=4096):
        try:
            if ms:
                x_preds = self.solve_ode(xs, ts, us, dt0=dt0, adaptive=adaptive, max_steps=max_steps)
                cont_loss = jnp.nansum((x_preds[:-1, -1] - xs[1:, 0]) ** 2)
                loss = jnp.nanmean((x_preds - xs) ** 2) + cont_loss
            else:
                x_preds = self.generator(xs[0], ts, us, dt0=dt0, adaptive=adaptive, max_steps=max_steps)
                loss = jnp.nanmean((x_preds - xs) ** 2)
        except:
            if ms:
                x_preds = self.solve_ode(xs, ts, us, dt0=dt0, adaptive=adaptive, max_steps=max_steps*2                              )
                cont_loss = jnp.nansum((x_preds[:-1, -1] - xs[1:, 0]) ** 2)
                loss = jnp.nanmean((x_preds - xs) ** 2) + cont_loss
            else:
                x_preds = self.generator(xs[0], ts, us, dt0=dt0, adaptive=adaptive, max_steps=max_steps*2)
                loss = jnp.nanmean((x_preds - xs) ** 2)
        return loss

    def test(self, xs, ts, us):
        xs_shape = xs.shape
        xs = xs.reshape(-1, xs_shape[-1])
        ts = ts.reshape(-1, 1)
        if us is None:
            if 'ft' in self.model_name:
                inp = jnp.concatenate([xs, jnp.sin(ts)], axis=-1)
            else:
                inp = xs
        else:
            us = us.reshape(-1, us.shape[-1])
            if 'ft' in self.model_name:
                inp = jnp.concatenate([xs, us, jnp.sin(ts)], axis=-1)
            else:
                inp = jnp.concatenate([xs, us], axis=-1)
        xdot_drift = jax.vmap(self.net)(inp)
        return xdot_drift.reshape(xs_shape)

class ResFunc(eqx.Module):
    layers: list
    activation: Callable

    def __init__(self, in_size: int | Literal["scalar"], out_size: int | Literal["scalar"],  width_size: int,
                 depth: int, activation: Callable, use_bias: bool = True,
                 use_final_bias: bool = True, *, key):
        keys = jr.split(key, depth + 1)
        self.activation = activation
        self.layers = [eqx.nn.Linear(in_size, width_size, use_bias=use_bias, key=keys[0])]
        for i in range(1, depth):
            self.layers.append(eqx.nn.Linear(width_size, width_size, use_bias=use_bias, key=keys[i]))
        self.layers.append(eqx.nn.Linear(width_size, out_size, use_bias=use_final_bias, key=keys[-1]))

    def __call__(self, y):
        y = self.activation(self.layers[0](y))
        for i in range(1, len(self.layers)-1):
            y = self.activation(self.layers[i](y)) + y
        return self.layers[-1](y)