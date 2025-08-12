import equinox as eqx
import jax.random as jr
import copy
import jax.nn as jnn
from distrax import Normal
from typing import Optional, Tuple, Any, List, Callable
import jax, jax.numpy as jnp
from equinox._module import field
from equinox.nn import Dropout
from neural_network_jax import Linear, MLP


class Flow(eqx.Module):
    base_dist_mean: jnp.ndarray
    base_dist_var: jnp.ndarray
    net: Any

    def __init__(self, input_size: int):
        # Initialize base distribution for log-prob calculation
        self.base_dist_mean = jnp.zeros(input_size)
        self.base_dist_var = jnp.ones(input_size)
        self.net = None  # This will be assigned in child classes

    def base_dist(self):
        return Normal(self.base_dist_mean, self.base_dist_var)

    def __call__(self, x, cond=None, key=None, inference=False):
        # data space to latent sapce
        if not inference:
            net, u, log_det_jacobian = self.net(x, cond, key, inference)
            self = eqx.tree_at(lambda m: m.net, self, net)
            return self, u, log_det_jacobian
        else:
            u, log_det_jacobian = self.net(x, cond, key, inference)
            return u, log_det_jacobian

    def inverse(self, u, cond=None, key=None, inference=False):
        # latent space to data space
        if not inference:
            net, u, log_det_jacobian = self.net.inverse(u, cond, key, inference)
            self = eqx.tree_at(lambda m: m.net, self, net)
            return self, u, log_det_jacobian
        else:
            u, log_det_jacobian = self.net.inverse(u, cond, key, inference)
            return u, log_det_jacobian

    def log_prob(self, x, cond=None, key=None, inference=False):
        # the density of data
        out = self.__call__(x, cond=cond, key=key, inference=inference)
        u, sum_log_det_jacobians = out[-2], out[-1]
        log_prob = self.base_dist().log_prob(u) + sum_log_det_jacobians
        return jnp.sum(log_prob, axis=-1)

    def sample(self, sample_shape=(1,), cond=None, key=None, inference=True):
        # sample x multiple times from latent variable
        shape = sample_shape if cond is None else cond.shape[:-1]
        u = self.base_dist().sample(seed=jr.key(0), sample_shape=shape)
        sample, _ = self.inverse(u, cond=cond, key=key, inference=inference)
        return sample

class FlowSequential(eqx.Module):
    layers: List

    def __init__(self, *layers):
        self.layers = list(layers)

    def __update__(self, i, new_layer):
        layer_list = self.layers.copy()
        layer_list[i] = new_layer
        return eqx.tree_at(lambda m: m.layers, self, layer_list)

    def __call__(self, x, cond=None, key=None, inference=False):
        log_det_jacobian = 0
        update_list = []
        for i, layer in enumerate(self.layers):
            if isinstance(layer, ActNorm) or isinstance(layer, BatchNorm):
                if not inference:
                    layer, x, log_det = layer(x, inference=inference)
                    update_list.append((i, layer))
                else:
                    x, log_det = layer(x, inference=inference)
            elif isinstance(layer, PermutationLayer):
                x = layer(x)
                log_det = 0
            elif isinstance(layer, Dropout):
                x = layer(x, key=key, inference=inference)
                log_det = 0
            else:
                x, log_det = layer(x, cond)
            log_det_jacobian += log_det
        for i, l in update_list:
            self = self.__update__(i, l)
        if not inference:
            return self, x, log_det_jacobian
        else:
            return x, log_det_jacobian

    def inverse(self, u, cond=None, key=None, inference=False):
        x, log_det_jacobian = u, 0
        update_list = []
        for i, layer in enumerate(reversed(self.layers)):
            if isinstance(layer, ActNorm) or isinstance(layer, BatchNorm):
                if not inference:
                    layer, x, log_det = layer.inverse(x, inference=inference)
                    update_list.append((i, layer))
                else:
                    x, log_det = layer.inverse(x, inference=inference)
            elif isinstance(layer, PermutationLayer):
                x = layer.inverse(x)
                log_det = 0
            elif isinstance(layer, Dropout):
                x = layer(x, key=key, inference=inference)
                log_det = 0
            else:
                x, log_det = layer.inverse(x, cond)
            log_det_jacobian += log_det
        for i, l in update_list:
            self = self.__update__(i, l)
        if not inference:
            return self, x, log_det_jacobian
        else:
            return x, log_det_jacobian


class ActNorm(eqx.Module):
    """
    Activation Normalization layer for normalizing flows.

    ActNorm performs an affine transformation: y = scale * x + bias
    The layer is initialized such that the first batch has zero mean and unit variance.

    Args:
        num_features: Number of features/channels to normalize
        axis: Axis or axes along which to normalize (default: -1 for last axis)
        eps: Small value for numerical stability
        initialized: Whether the layer has been initialized with data
    """
    eps: float
    initialized: bool
    scale: jnp.ndarray
    bias: jnp.ndarray

    def __init__(
            self,
            num_features: int,
            eps: float = 1e-8
    ):
        self.eps = eps
        self.initialized = False
        if isinstance(num_features, int):
            shape = (num_features,)
        else:
            shape = num_features

        self.scale = jnp.ones(shape)
        self.bias = jnp.zeros(shape)

    def __call__(self, x, inference=False):
        """
        Forward pass through ActNorm layer.

        Args:
            x: Input array

        Returns:
            tuple: (transformed_output, log_det_jacobian)
        """
        if not self.initialized:
            self = self.initialize(x)

        # Apply affine transformation: y = scale * x + bias
        y = self.scale * x + self.bias

        # Compute log determinant of Jacobian
        # For affine transformation y = scale * x + bias, log|det(J)| = sum(log|scale|)
        # We need to account for the shape properly
        log_det = jnp.log(self.scale)
        if not inference:
            return self, y, log_det
        else:
            return y, log_det

    def inverse(self, y, inference=False):
        """
        Inverse transformation: x = (y - bias) / scale

        Args:
            y: Input to inverse transform

        Returns:
            tuple: (inverse_output, log_det_jacobian)
        """
        if not self.initialized:
            self = self.initialize(y)

        # Inverse transformation: x = (y - bias) / scale
        x = (y - self.bias) / self.scale

        # Log determinant for inverse is negative of forward
        log_det = - jnp.log(self.scale)

        if not inference:
            return self, x, log_det
        else:
            return x, log_det

    def initialize(self, x):
        """
        Initialize the ActNorm parameters based on input statistics.

        This should be called with the first batch of data to ensure
        the output has zero mean and unit variance.

        Args:
            x: First batch of data for initialization

        Returns:
            New ActNorm instance with initialized parameters
        """
        # Compute statistics along all axes except the feature axis
        axes_to_reduce = tuple(i for i in range(x.ndim-1))
        if axes_to_reduce:
            mean = jnp.mean(x, axis=axes_to_reduce, keepdims=False)
            std = jnp.std(x, axis=axes_to_reduce, keepdims=False) + self.eps
        else:
            mean = jnp.zeros_like(x)
            std = jnp.ones_like(x)

        new_scale = 1.0 / std
        new_bias = - mean / std

        update_an = eqx.tree_at(lambda m: (m.scale, m.bias, m.initialized), self, (new_scale, new_bias, True))
        return update_an

class PermutationLayer(eqx.Module):
    perm: jnp.ndarray = field(static=True)
    inv_perm: jnp.ndarray = field(static=True)
    def __init__(self, dim, key=None):
        if key is None:
            key = jr.PRNGKey(0)
        perm = jr.permutation(key, dim)
        inv_perm = jnp.argsort(perm)
        self.perm = perm
        self.inv_perm = inv_perm
    def __call__(self, x):
        return x[..., self.perm]

    def inverse(self, x):
        return x[..., self.inv_perm]

class MaskedCoupling(eqx.Module):
    """Masked coupling layer for RealNVP, using functional JAX layers and masks."""
    s_net: MLP
    t_net: MLP
    mask: field(static=True)

    def __init__(self, input_size: int, hidden_size: int, n_hidden: int, mask,
                 cond_label_size: Optional[int] = None, key=jr.key(42)):
        # mv = mu; umv = umu*exp(s(mu))+t(mu)
        in_features = input_size + (cond_label_size or 0)
        s_key, t_key = jr.split(key, 2)
        self.s_net = MLP(in_size=in_features, out_size=input_size, width_size=hidden_size, depth=n_hidden,
                         key=s_key, activation=jnn.tanh, final_activation=jnn.tanh,
                         initializer=jnn.initializers.glorot_uniform(),
                         final_initializer=jnn.initializers.glorot_uniform())
        self.t_net = MLP(in_size=in_features, out_size=input_size, width_size=hidden_size, depth=n_hidden,
                         key=s_key, activation=jnn.relu, final_activation=jnn.relu,
                         initializer=jnn.initializers.he_normal(),
                         final_initializer=jnn.initializers.he_normal())
        self.mask = mask

    def __call__(self, x: jnp.ndarray, y: Optional[jnp.ndarray] = None) -> Tuple[jnp.ndarray, jnp.ndarray]:
        # Apply mask
        mx = x * self.mask
        umx = x * (1 - self.mask)

        # Concatenate conditional information if provided
        if y is not None:
            mxy = jnp.concatenate([mx, y], axis=-1)
        else:
            mxy = mx

        # Compute scale and translation
        s = self.s_net(mxy) * (1 - self.mask)
        t = self.t_net(mxy) * (1 - self.mask)

        # Apply scaling and translation transformations
        z = mx + umx * jnp.exp(s) + t
        # Log determinant of the Jacobian
        log_det_jacobian = s
        return z, log_det_jacobian

    def inverse(self, z: jnp.ndarray, y: Optional[jnp.ndarray] = None) -> Tuple[jnp.ndarray, jnp.ndarray]:
        # Apply inverse transformation
        mz = z * self.mask
        umz = z * (1 - self.mask)
        if y is not None:
            mzy = jnp.concatenate([mz, y], axis=-1)
        else:
            mzy = mz

        s = self.s_net(mzy) * (1 - self.mask)
        t = self.t_net(mzy) * (1 - self.mask)

        x = mz + (umz - t) * jnp.exp(-s)

        # Calculate log determinant for inverse
        log_det_jacobian = - s
        return x, log_det_jacobian


class CrossMaskedCoupling(eqx.Module):
    """Masked coupling layer for RealNVP, using functional JAX layers and masks."""
    s1_net: MLP
    t1_net: MLP
    s2_net: MLP
    t2_net: MLP
    mask: field(static=True)

    def __init__(self, input_size: int, hidden_size: int, n_hidden: int,
                 cond_label_size: Optional[int] = None, key=jr.key(42)):
        # mv = mu * exp(s1(umu))+ t1(umu)
        # umv = umu * exp(s2(mv)) + t2(mv)
        in_features = input_size + (cond_label_size or 0)
        s1_key, t1_key, s2_key, t2_key = jr.split(key, 4)
        self.s1_net = MLP(in_size=in_features, out_size=input_size, width_size=hidden_size, depth=n_hidden,
                         key=s1_key, activation=jnn.tanh, initializer=jnn.initializers.glorot_uniform(),
                         final_initializer=jnn.initializers.glorot_uniform())
        self.t1_net = copy.deepcopy(self.s1_net)
        self.s2_net = copy.deepcopy(self.s1_net)
        self.t2_net = copy.deepcopy(self.s1_net)
        self.mask = jnp.arange(input_size) % 2

    def __call__(self, x: jnp.ndarray, y: Optional[jnp.ndarray] = None) -> Tuple[jnp.ndarray, jnp.ndarray]:
        # Apply mask
        mx = x * self.mask
        umx = x * (1 - self.mask)

        # Concatenate conditional information if provided
        if y is not None:
            umxy = jnp.concatenate([umx, y], axis=-1)
        else:
            umxy = umx

        s1 = self.s1_net(umxy) * self.mask
        # Apply scaling and translation transformations
        mz = mx * jnp.exp(s1) + self.t1_net(umxy) * self.mask

        if y is not None:
            mzy = jnp.concatenate([mz, y], axis=-1)
        else:
            mzy = mz

        s2 = self.s2_net(mzy) * (1 - self.mask)
        umz = umx * jnp.exp(s2) + self.t2_net(mzy) * (1 - self.mask)
        z = mz + umz
        # Log determinant of the Jacobian
        log_det_jacobian = s1 * self.mask + s2 * (1 - self.mask)
        return z, log_det_jacobian

    def inverse(self, z: jnp.ndarray, y: Optional[jnp.ndarray] = None) -> Tuple[jnp.ndarray, jnp.ndarray]:
        # Apply inverse transformation
        mz = z * self.mask
        umz = z * (1 - self.mask)
        if y is not None:
            mzy = jnp.concatenate([mz, y], axis=-1)
        else:
            mzy = mz

        s2 = self.s2_net(mzy) * (1 - self.mask)
        umx = (umz - self.t2_net(mzy) * (1 - self.mask)) * jnp.exp(-s2)
        if y is not None:
            umxy = jnp.concatenate([umx, y], axis=-1)
        else:
            umxy = umx
        s1 = self.s1_net(umxy) * self.mask
        mx = (mz - self.t1_net(umxy) * self.mask) * jnp.exp(-s1)
        x = mx + umx
        # Calculate log determinant for inverse
        log_det_jacobian = - s1 * self.mask - s2 * (1 - self.mask)
        return x, log_det_jacobian


class RealNVP(Flow):
    def __init__(
        self,
        n_blocks: int,
        input_size: int,
        hidden_size: int,
        n_hidden: int,
        cond_label_size: Optional[int] = None,
        act_norm: bool = True,
        dropout:bool = True,
        key: jr.key = jr.key(0),
        model='coupling',
        dropout_rate=0.1
    ):
        super().__init__(input_size)
        perm_k, layer_k = jr.split(key, 2)
        # Build model layers
        mask = jnp.arange(input_size) % 2
        modules = []
        for i in range(n_blocks):
            if input_size > 2:
                modules.append(PermutationLayer(input_size, key=perm_k))
            if model == 'coupling':
                modules.append(
                    MaskedCoupling(
                        input_size, hidden_size, n_hidden, mask, cond_label_size, layer_k
                    )
                )
                modules.append(
                    MaskedCoupling(
                        input_size, hidden_size, n_hidden, 1-mask, cond_label_size, layer_k
                    )
                ) # Alternate mask between layers
            else:
                modules.append(CrossMaskedCoupling(input_size, hidden_size, n_hidden, cond_label_size, layer_k))
            if dropout:
                modules.append(Dropout(p=dropout_rate))
            if act_norm:
                modules.append(ActNorm(input_size))
        self.net = FlowSequential(*modules)


class BatchNorm(eqx.Module):
    gamma: jnp.ndarray
    beta: jnp.ndarray
    eps: float
    batch_mean: jnp.ndarray
    batch_var: jnp.ndarray


    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.gamma = jnp.zeros((dim,), dtype=jnp.float32)
        self.beta = jnp.zeros((dim,), dtype=jnp.float32)
        self.batch_mean = jnp.zeros((dim,), dtype=jnp.float32)
        self.batch_var = jnp.ones((dim,), dtype=jnp.float32)

    def __call__(self, x, inference=False):
        if not inference:
            self = eqx.tree_at(lambda m: (m.batch_mean, m.batch_var), self,
                                 (x.mean(axis=0), x.var(axis=0) + self.eps))
        m = self.batch_mean.copy()
        v = self.batch_var.copy()

        x_hat = (x - m) / jnp.sqrt(v)
        x_hat = x_hat * jnp.exp(self.gamma) + self.beta
        log_det = self.gamma - 0.5 * jnp.log(v)
        if not inference:
            return self, x_hat, log_det
        else:
            return x_hat, log_det

    def inverse(self, x, inference=False):
        if not inference:
            self = eqx.tree_at(lambda m: (m.batch_mean, m.batch_var), self,
                                 (x.mean(axis=0), x.var(axis=0) + self.eps))
        m = self.batch_mean
        v = self.batch_var

        x_hat = (x - self.beta) * jnp.exp(-self.gamma) * jnp.sqrt(v) + m
        log_det = -self.gamma + 0.5 * jnp.log(v)
        if not inference:
            return self, x_hat, log_det
        else:
            return x_hat, log_det


class MaskedLinear(Linear):
    mask: jnp.ndarray = field(static=True)
    cond_weight: jnp.ndarray = None

    def __init__(self, in_features, out_features, mask, cond_label_size=None, key=jr.key(0)):
        super().__init__(in_features, out_features, key=key)
        self.mask = mask
        if cond_label_size:
            self.cond_weight = jr.normal(key, (out_features, cond_label_size)) / jnp.sqrt(cond_label_size)

    def __call__(self, x, y=None):
        out = x @ (self.weight * self.mask).T
        if self.use_bias:
            out += self.bias
        if y is not None:
            out += y @ self.cond_weight.T
        return out



class MADE(eqx.Module):
    net: List[MaskedLinear]
    activation: Callable
    masks: List[jnp.ndarray] = field(static=True)
    input_order: jnp.ndarray = field(static=True)

    def __init__(
            self,
            input_size,
            hidden_size,
            n_hidden,
            cond_label_size=None,
            activation=jnn.relu,
            input_order='natural',  # Use natural for debugging
            key=jr.PRNGKey(0),
    ):
        mask_k, net_key = jr.split(key, 2)

        self._create_masks(input_size, hidden_size, n_hidden, mask_k, input_order)
        self.activation = activation

        keys = jr.split(net_key, n_hidden + 1)
        self.net = []

        # Input layer
        self.net.append(MaskedLinear(input_size, hidden_size, self.masks[0],
                                     cond_label_size=cond_label_size, key=keys[0]))

        # Hidden layers
        for i in range(n_hidden - 1):
            self.net.append(MaskedLinear(hidden_size, hidden_size, self.masks[i + 1],
                                         cond_label_size=cond_label_size, key=keys[i + 1]))

        # Output layer
        self.net.append(MaskedLinear(hidden_size, 2 * input_size, self.masks[-1],
                                     cond_label_size=cond_label_size, key=keys[-1]))

    def _create_masks(self, input_size, hidden_size, n_hidden, mask_k, input_order='natural'):
        keys = jr.split(mask_k, n_hidden + 2)

        if input_order == 'natural':
            self.input_order = jnp.arange(input_size)
        else:
            self.input_order = jr.permutation(keys[0], input_size)

        self.masks = []
        degrees = {}

        # Input degrees
        degrees[0] = self.input_order

        # Hidden layer degrees
        for l in range(n_hidden):
            min_prev = degrees[l].min()
            max_prev = input_size - 1
            degrees[l + 1] = jr.randint(keys[l + 1], (hidden_size,),
                                        minval=min_prev, maxval=max_prev + 1)

        # Output degrees - CRITICAL: each output predicts its own variable
        degrees[n_hidden + 1] = jnp.concatenate([self.input_order, self.input_order])

        # Create connectivity masks
        for l in range(n_hidden + 1):
            in_degrees = degrees[l]
            out_degrees = degrees[l + 1]

            # out[j] connects to in[i] if out_degree[j] > in_degree[i]
            # This ensures autoregressive property
            mask = (out_degrees[:, None] > in_degrees[None, :]).astype(jnp.float32)
            self.masks.append(mask)

    def forward(self, x, y=None):
        h = x
        for i, layer in enumerate(self.net):
            h = layer(h, y=y)
            if i < len(self.net) - 1:
                h = self.activation(h)
        return h

    def __call__(self, x, y=None):
        out = self.forward(x, y=y)
        mu, log_s = jnp.split(out, 2, axis=-1)

        # Apply transformation: z = (x - mu) / exp(log_s)
        # This ensures det(J) = exp(-sum(log_s))
        s = jnp.exp(log_s)
        z = (x - mu) / s
        log_det_J = -log_s

        return z, log_det_J

    def inverse(self, z, y=None):
        """
        CORRECT inverse transformation
        """
        x = jnp.zeros_like(z)

        # Process in autoregressive order
        for i in range(len(self.input_order)):
            # Get current estimates
            out = self.forward(x, y=y)
            mu, log_s = jnp.split(out, 2, axis=-1)

            # Find which input dimension to update
            # (the one with order value i)
            dim_to_update = jnp.where(self.input_order == i)[0][0]

            # Inverse transformation: x = mu + z * exp(log_s)
            s = jnp.exp(log_s[..., dim_to_update])
            x_new = mu[..., dim_to_update] + z[..., dim_to_update] * s
            x = x.at[..., dim_to_update].set(x_new)

        return x, log_s


class MAF(Flow):
    def __init__(
        self,
        n_blocks: int,
        input_size: int,
        hidden_size: int,
        n_hidden: int,
        cond_label_size: Optional[int] = None,
        activation: Callable = jnn.relu,
        batch_norm: bool = True,
        batch_norm_last: bool = False,
        key: jr.key = jr.key(0)
    ):
        super().__init__(input_size)

        # Build model layers
        modules = []
        for i in range(n_blocks):
            maf_block = MADE(
                input_size,
                hidden_size,
                n_hidden,
                cond_label_size,
                activation,
                key
            )
            modules.append(maf_block)
            if batch_norm and i < n_blocks - 1:
                modules.append(BatchNorm(input_size))
            if batch_norm_last and i == n_blocks - 1:
                modules.append(BatchNorm(input_size))

        self.net = FlowSequential(*modules)