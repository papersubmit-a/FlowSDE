import os
from numba import cuda
if cuda.is_available():
    import jax
else:
    os.environ["JAX_PLATFORMS"] = 'cpu'
    import jax
    jax.default_device('cpu')
from jaxtyping import Array, PRNGKeyArray
from typing import Literal, Optional, Union, Callable
import equinox as eqx
import jax.random as jr
import jax.nn as jnn
import jax.numpy as jnp
import jax.tree_util as jtu
from equinox._misc import default_floating_dtype
from equinox._module import field
from equinox._vmap_pmap import filter_vmap
from equinox._filters import is_array


class Linear(eqx.Module, strict=True):
    """Performs a linear transformation.
    based on the equinox.nn.Linear, enables matrix transformation,
    including initializer argument"""

    weight: Array
    bias: Optional[Array]
    in_features: Union[int, Literal["scalar"]] = field(static=True)
    out_features: Union[int, Literal["scalar"]] = field(static=True)
    use_bias: bool = field(static=True)

    def __init__(
        self,
        in_features: Union[int, Literal["scalar"]],
        out_features: Union[int, Literal["scalar"]],
        use_bias: bool = True,
        dtype=None,
        initializer=None,
        *,
        key: PRNGKeyArray
    ):
        dtype = default_floating_dtype() if dtype is None else dtype
        wkey, bkey = jr.split(key, 2)
        in_features_ = 1 if in_features == "scalar" else in_features
        out_features_ = 1 if out_features == "scalar" else out_features
        wshape = (out_features_, in_features_)
        if initializer is None:
            initializer = jax.nn.initializers.he_normal()
        self.weight = initializer(wkey, wshape, dtype)
        self.bias = jnp.zeros(out_features, dtype=default_floating_dtype()) if use_bias else None

        self.in_features = in_features
        self.out_features = out_features
        self.use_bias = use_bias

    def __call__(self, x: Array, *, key: Optional[PRNGKeyArray] = None) -> Array:
        x = x @ self.weight.T
        if self.bias is not None:
            x = x + self.bias
        return x

class MLP(eqx.Module, strict=True):
    """Standard Multi-Layer Perceptron; also known as a feed-forward network.
    implemented with the adapted Linear layer"""

    layers: tuple[Linear, ...]
    activation: Callable
    final_activation: Callable
    use_bias: bool = field(static=True)
    use_final_bias: bool = field(static=True)
    in_size: Union[int, Literal["scalar"]] = field(static=True)
    out_size: Union[int, Literal["scalar"]] = field(static=True)
    width_size: int = field(static=True)
    depth: int = field(static=True)

    def __init__(
        self,
        in_size: Union[int, Literal["scalar"]],
        out_size: Union[int, Literal["scalar"]],
        width_size: int,
        depth: int,
        activation: Callable = jnn.relu,
        final_activation: Callable = lambda x: x,
        use_bias: bool = True,
        use_final_bias: bool = True,
        dtype=None,
        initializer=None,
        final_initializer=None,
        *,
        key: PRNGKeyArray,
    ):
        dtype = default_floating_dtype() if dtype is None else dtype
        keys = jr.split(key, depth + 1)
        layers = []
        if depth == 0:
            layers.append(
                Linear(in_size, out_size, use_final_bias, initializer=final_initializer, dtype=dtype, key=keys[0])
            )
        else:
            layers.append(
                Linear(in_size, width_size, use_bias, initializer=initializer, dtype=dtype, key=keys[0])
            )
            for i in range(depth - 1):
                layers.append(
                    Linear(
                        width_size, width_size, use_bias, initializer=initializer, dtype=dtype, key=keys[i + 1]
                    )
                )
            layers.append(
                Linear(width_size, out_size, use_final_bias, initializer=final_initializer, dtype=dtype, key=keys[-1])
            )
        self.layers = tuple(layers)
        self.in_size = in_size
        self.out_size = out_size
        self.width_size = width_size
        self.depth = depth
        # In case `activation` or `final_activation` are learnt, then make a separate
        # copy of their weights for every neuron.
        self.activation = filter_vmap(
            filter_vmap(lambda: activation, axis_size=width_size), axis_size=depth
        )()
        if out_size == "scalar":
            self.final_activation = final_activation
        else:
            self.final_activation = filter_vmap(
                lambda: final_activation, axis_size=out_size
            )()
        self.use_bias = use_bias
        self.use_final_bias = use_final_bias

    @jax.named_scope("eqx.nn.MLP")
    def __call__(self, x: Array, *, key: Optional[PRNGKeyArray] = None) -> Array:
        for i, layer in enumerate(self.layers[:-1]):
            x = layer(x)
            layer_activation = jtu.tree_map(
                lambda x: x[i] if is_array(x) else x, self.activation
            )
            x = filter_vmap(lambda a, b: a(b))(layer_activation, x)
        x = self.layers[-1](x)
        if self.out_size == "scalar":
            x = self.final_activation(x)
        else:
            x = filter_vmap(lambda a, b: a(b))(self.final_activation, x)
        return x

class LSTM(eqx.Module):
    cell: eqx.nn.LSTMCell

    def __init__(self, **kwargs):
        self.cell = eqx.nn.LSTMCell(**kwargs)

    def __call__(self, xs):
        def scan_fn(state, input):
            state_new = self.cell(input, state)
            return state_new, state_new
        init_state = (jnp.zeros(self.cell.hidden_size),
                      jnp.zeros(self.cell.hidden_size))
        (final_h, final_c), (out_h, out_c) = jax.lax.scan(scan_fn, init_state, xs)
        return (final_h, final_c), (out_h, out_c)


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