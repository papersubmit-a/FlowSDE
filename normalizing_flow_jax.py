import equinox as eqx
import jax.random as jr
import copy
import jax.nn as jnn
from distrax import Normal
from typing import Optional, Tuple, Any, List, Callable
import jax, jax.numpy as jnp
from equinox.nn import Dropout
import diffrax
from neural_network_jax import Linear


def create_masks(input_size, hidden_size, n_hidden, input_order="sequential", input_degrees=None, key=jr.key(0)):
    degrees = []

    if input_order == "sequential":
        degrees.append(jnp.arange(input_size) if input_degrees is None else input_degrees)
        for _ in range(n_hidden + 1):
            degrees.append(jnp.arange(hidden_size) % (input_size - 1))
        degrees.append(jnp.arange(input_size) % input_size - 1 if input_degrees is None else input_degrees % input_size - 1)
    elif input_order == "random":
        keys = jr.split(key, n_hidden + 3)
        degrees.append(jr.permutation(keys[0], input_size) if input_degrees is None else input_degrees)
        for i in range(n_hidden + 1):
            min_prev_degree = min(degrees[-1].min(), input_size - 1)
            degrees.append(jr.randint(keys[i+1], (hidden_size,), min_prev_degree, input_size))
        min_prev_degree = min(degrees[-1].min(), input_size - 1)
        degrees.append(jr.randint(keys[-1], (input_size,), min_prev_degree, input_size) - 1 if input_degrees is None else input_degrees - 1)
    masks = [(d1[:, None] >= d0[None, :]).astype(jnp.float32) for d0, d1 in zip(degrees[:-1], degrees[1:])]
    return masks, degrees[0]

class MaskedLinear(eqx.Module):
    weight: jnp.ndarray
    bias: jnp.ndarray
    mask: jnp.ndarray
    cond_weight: jnp.ndarray = None

    def __init__(self, in_features, out_features, mask, cond_label_size=None, key=jr.key(0)):
        self.weight = jr.normal(key, (out_features, in_features))
        self.bias = jnp.zeros(out_features)
        self.mask = mask
        if cond_label_size:
            self.cond_weight = jr.normal(key, (out_features, cond_label_size)) / jnp.sqrt(cond_label_size)

    def __call__(self, x, y=None):
        out = x @ (self.weight * self.mask).T + self.bias
        if y is not None:
            out += y @ self.cond_weight.T
        return out


class MADE(eqx.Module):
    net_input: MaskedLinear
    net: list
    base_dist_mean: jnp.ndarray
    base_dist_var: jnp.ndarray
    input_degrees: jnp.ndarray

    def __init__(
            self,
            input_size,
            hidden_size,
            n_hidden,
            cond_label_size=None,
            activation="ReLU",
            input_order="sequential",
            input_degrees=None,
            key=jr.key(0),
    ):
        # Initialize base distribution
        self.base_dist_mean = jnp.zeros(input_size)
        self.base_dist_var = jnp.ones(input_size)

        # Create masks
        masks, self.input_degrees = create_masks(input_size, hidden_size, n_hidden, input_order, input_degrees, key)

        # Set up activation function
        if activation == "ReLU":
            activation_fn = jax.nn.relu
        elif activation == "Tanh":
            activation_fn = jax.nn.tanh
        else:
            raise ValueError("Unsupported activation function. Choose 'ReLU' or 'Tanh'.")

        # Build network layers
        keys = jr.split(key, len(masks))
        self.net_input = MaskedLinear(input_size, hidden_size, masks[0], cond_label_size, key=keys[0])
        self.net = []
        for m, k in zip(masks[1:-1], keys[1:-1]):
            self.net += [activation_fn, MaskedLinear(hidden_size, hidden_size, m, key=k)]
        self.net += [
            activation_fn,
            MaskedLinear(hidden_size, 2 * input_size, masks[-1].repeat(2, 1), key=keys[-1]),
        ]

    def base_dist(self):
        return Normal(self.base_dist_mean, self.base_dist_var)

    def subnet(self, net, x):
        if isinstance(net, list):
            for l in net:
                x = l(x)
        else:
            x = net(x)
        return x

    def __call__(self, x, y=None):
        # MAF eq 4 -- return mean and log std
        m, loga = jnp.split(self.subnet(self.net, self.net_input(x, y)), 2, axis=-1)
        u = (x - m) * jnp.exp(-loga)
        # MAF eq 5
        log_abs_det_jacobian = -loga
        return u, log_abs_det_jacobian

    def inverse(self, u, y=None):
        # MAF eq 3 (inverse pass)
        x = jnp.zeros_like(u)
        for i in self.input_degrees:
            m, loga = jnp.split(self.subnet(self.net, self.net_input(x, y)), 2, axis=-1)
            x = x.at[..., i].set(u[..., i] * jnp.exp(loga[..., i]) + m[..., i])


class Flow(eqx.Module):
    base_dist_mean: jnp.ndarray
    base_dist_var: jnp.ndarray
    scale: Optional[float] = None
    net: Any

    def __init__(self, input_size: int):
        # Initialize base distribution for log-prob calculation
        self.base_dist_mean = jnp.zeros(input_size)
        self.base_dist_var = jnp.ones(input_size)
        self.scale = None  # Scale can be set later externally
        self.net = None  # This will be assigned in child classes

    def base_dist(self):
        return Normal(self.base_dist_mean, self.base_dist_var)

    def __call__(self, x, cond=None, key=None, inference=False):
        if self.scale is not None:
            x /= self.scale
        u, log_abs_det_jacobian = self.net(x, cond, key, inference)
        return u, log_abs_det_jacobian

    def inverse(self, u, cond=None, key=None, inference=False):
        x, log_abs_det_jacobian = self.net.inverse(u, cond, key, inference)
        if self.scale is not None:
            x *= self.scale
            log_abs_det_jacobian += jnp.log(jnp.abs(self.scale))
        return x, log_abs_det_jacobian

    def log_prob(self, x, cond=None, key=None, inference=False):
        u, sum_log_abs_det_jacobians = self.__call__(x, cond=cond, key=key, inference=inference)
        log_prob = self.base_dist().log_prob(u) + sum_log_abs_det_jacobians
        return jnp.sum(log_prob, axis=-1)

    def sample(self, sample_shape=(1,), cond=None, key=None, inference=False):
        shape = sample_shape if cond is None else cond.shape[:-1]
        u = self.base_dist().sample(seed=jr.key(0), sample_shape=shape)
        sample, _ = self.inverse(u, cond=cond, key=key, inference=inference)
        return sample


class FlowSequential(eqx.Module):
    layers: List

    def __init__(self, *layers):
        self.layers = list(layers)

    def __call__(self, x, cond=None, key=None, inference=False):
        log_abs_det_jacobian = 0
        for layer in self.layers:
            if isinstance(layer, BatchNorm):
                if inference:
                    x, log_det = layer(x, inference=inference)
                else:
                    layer, x, log_det = layer(x, inference=inference)
            elif isinstance(layer, Dropout):
                x = layer(x, key=key, inference=key is None)
                log_det = 0
            else:
                x, log_det = layer(x, cond)
            log_abs_det_jacobian += log_det
        return x, log_abs_det_jacobian

    def inverse(self, u, cond=None, key=None, inference=False):
        x, log_abs_det_jacobian = u, 0
        for layer in reversed(self.layers):
            if isinstance(layer, BatchNorm):
                if inference:
                    x, log_det = layer(x, inference=inference)
                else:
                    layer, x, log_det = layer(x, inference=inference)
            elif isinstance(layer, Dropout):
                x = layer(x, key=key, inference=key is None)
                log_det = 0
            else:
                x, log_det = layer.inverse(x, cond)
            log_abs_det_jacobian += log_det
        return x, log_abs_det_jacobian


class BatchNorm(eqx.Module):
    """RealNVP BatchNorm layer implemented in Equinox and JAX."""
    log_gamma: jnp.ndarray
    beta: jnp.ndarray
    momentum: float
    eps: float
    running_mean: jnp.ndarray
    running_var: jnp.ndarray

    def __init__(self, input_size: int, momentum: float = 0.9, eps: float = 1e-5):
        self.momentum = momentum
        self.eps = eps
        self.log_gamma = jnp.zeros(input_size)
        self.beta = jnp.zeros(input_size)
        self.running_mean = jnp.zeros(input_size)
        self.running_var = jnp.ones(input_size)

    def __call__(self, x: jnp.ndarray, inference: bool = True) -> Tuple[jnp.ndarray, jnp.ndarray]:
        if inference:
            batch_mean = self.running_mean
            batch_var = self.running_var
        else:
            batch_mean = jnp.mean(x, axis=0)
            batch_var = jnp.var(x, axis=0)

            # Update running statistics in a functional way
            updated_bn = eqx.tree_at(lambda m: (m.running_mean, m.running_var), self, (batch_mean, batch_var))
        # Normalize input
        x_hat = (x - batch_mean) / jnp.sqrt(batch_var + self.eps)
        y = jnp.exp(self.log_gamma) * x_hat + self.beta

        # Log determinant of the Jacobian
        log_abs_det_jacobian = self.log_gamma - 0.5 * jnp.log(batch_var + self.eps)
        if not inference:
            return updated_bn, y, log_abs_det_jacobian
        else:
            return y, log_abs_det_jacobian


class LinearMaskedCoupling(eqx.Module):
    """Masked coupling layer for RealNVP, using functional JAX layers and masks."""
    s_net: List
    t_net: List
    mask: jnp.ndarray

    def __init__(self, input_size: int, hidden_size: int, n_hidden: int, mask: jnp.ndarray,
                 cond_label_size: Optional[int] = None, key=jr.key(42)):
        # Initialize masked coupling layer networks for scale (s_net) and translation (t_net)
        self.mask = mask

        # Scale (s_net)
        self.s_net = []
        in_features = input_size + (cond_label_size or 0)
        keys = jr.split(key, n_hidden + 1)
        for i in range(n_hidden):
            self.s_net.append(Linear(in_features, hidden_size, key=keys[i]))
            self.s_net.append(jnn.tanh)
            in_features = hidden_size
        self.s_net.append(Linear(hidden_size, input_size, key=keys[-1]))
        self.t_net = copy.deepcopy(self.s_net)
        for i in range(len(self.t_net)):
            if not isinstance(self.t_net[i], Linear):
                self.t_net[i] = jnn.relu

    def subnet(self, net, x):
        if isinstance(net, list):
            for l in net:
                x = l(x)
        else:
            x = net(x)
        return x

    def __call__(self, x: jnp.ndarray, y: Optional[jnp.ndarray] = None) -> Tuple[jnp.ndarray, jnp.ndarray]:
        # Apply mask
        mx = x * self.mask

        # Concatenate conditional information if provided
        if y is not None:
            mx = jnp.concatenate([y, mx], axis=-1)

        # Compute scale and translation
        s = self.subnet(self.s_net, mx)
        t = self.subnet(self.t_net, mx) * (1 - self.mask)

        # Apply scaling and translation transformations
        log_s = jnn.tanh(s) * (1 - self.mask)
        u = x * jnp.exp(log_s) + t

        # Log determinant of the Jacobian
        log_abs_det_jacobian = log_s
        return u, log_abs_det_jacobian

    def inverse(self, y: jnp.ndarray, cond_y: Optional[jnp.ndarray] = None) -> Tuple[jnp.ndarray, jnp.ndarray]:
        # Apply inverse transformation
        if cond_y is not None:
            masked_y = y * self.mask
            input_with_cond = jnp.concatenate([cond_y, masked_y], axis=-1)
        else:
            input_with_cond = y * self.mask

        s = self.subnet(self.s_net, input_with_cond)
        t = self.subnet(self.t_net, input_with_cond) * (1 - self.mask)

        # Scale and translate
        log_s = jnn.tanh(s) * (1 - self.mask)
        x_hat = (y - t) * jnp.exp(-log_s)

        # Calculate log determinant for inverse
        log_abs_det_jacobian = -log_s
        return x_hat, log_abs_det_jacobian


class RealNVP(Flow):
    def __init__(
        self,
        n_blocks: int,
        input_size: int,
        hidden_size: int,
        n_hidden: int,
        cond_label_size: Optional[int] = None,
        batch_norm: bool = True,
        batch_norm_last: bool = False,
        dropout:bool = True,
        key: jr.key = jr.key(0)
    ):
        super().__init__(input_size)

        # Build model layers
        modules = []
        mask = jnp.arange(input_size) % 2
        for i in range(n_blocks):
            modules.append(
                LinearMaskedCoupling(
                    input_size, hidden_size, n_hidden, mask, cond_label_size, key
                )
            )
            mask = 1 - mask  # Alternate mask between layers
            if dropout:
                modules.append(Dropout(p=0.1))
            if batch_norm and i < n_blocks - 1:
                modules.append(BatchNorm(input_size))
            if batch_norm_last and i == n_blocks - 1:
                modules.append(BatchNorm(input_size))

        self.net = FlowSequential(*modules)


class MAF(Flow):
    def __init__(
        self,
        n_blocks: int,
        input_size: int,
        hidden_size: int,
        n_hidden: int,
        cond_label_size: Optional[int] = None,
        activation: str = "ReLU",
        input_order: str = "sequential",
        batch_norm: bool = True,
        batch_norm_last: bool = False,
        key: jr.key = jr.key(0)
    ):
        super().__init__(input_size)

        # Build model layers
        modules = []
        input_degrees = None
        for i in range(n_blocks):
            maf_block = MADE(
                input_size,
                hidden_size,
                n_hidden,
                cond_label_size,
                activation,
                input_order,
                input_degrees,
                key
            )
            modules.append(maf_block)
            input_degrees = maf_block.input_degrees[::-1]  # Flip order for next block

            if batch_norm and i < n_blocks - 1:
                modules.append(BatchNorm(input_size))
            if batch_norm_last and i == n_blocks - 1:
                modules.append(BatchNorm(input_size))

        self.net = FlowSequential(*modules)


def antideriv_tanh(x):
    return jnp.abs(x) + jnp.log(1 + jnp.exp(-2.0 * jnp.abs(x)))


def deriv_tanh(x):
    return 1 - jnp.tanh(x) ** 2


class ResNN(eqx.Module):
    d: int
    m: int
    nTh: 2
    layers: list
    h: float
    act_fun = callable

    def __init__(self, d, m, nTh=2, key=jr.key(0), act_fun=antideriv_tanh):
        keys = jr.split(key, nTh)
        self.layers = [Linear(d + 1, m, key=keys[0])] + \
                      [Linear(m, m, key=keys[i]) for i in range(1, nTh)]
        self.act_fun = act_fun
        self.h = 1.0 / (self.nTh - 1)

    def __call__(self, x):
        x = self.act_fun(self.layers[0](x))
        for l in self.layers[1:]:
            x = x + self.h * self.act_fun(l(x))
        return x


class Phi(eqx.Module):
    nTh: int
    m: int
    d: int
    r: int
    alph: list
    A: jnp.ndarray
    c: Linear
    w: Linear
    N: ResNN

    def __init__(self, nTh, m, d, r=10, alph=None, key=None):
        self.nTh = nTh
        self.m = m
        self.d = d
        self.r = r
        self.alph = alph
        keys = jr.split(key, 3)
        self.A = jr.uniform(keys[0], (r, d+1))
        self.c = Linear(d + 1, 1, key=keys[1])
        self.w = Linear(m, 1, key=keys[2])
        self.N = ResNN(d, m, nTh=nTh, key=key)

    def __call__(self, x):
        symA = self.A.T @ self.A
        return self.w(self.N(x)) + 0.5 * jnp.sum(x @ symA @ x, axis=1, keepdims=True) + self.c(x)

    def trHess(self, x, just_grad=False):
        """
        compute gradient of Phi wrt x and trace(Hessian of Phi); see Eq. (11) and Eq. (13), respectively
        recomputes the forward propogation portions of Phi

        :param x: input data, torch Tensor nex-by-d
        :param justGrad: boolean, if True only return gradient, if False return (grad, trHess)
        :return: gradient , trace(hessian)    OR    just gradient
        """

        # code in E = eye(d+1,d) as index slicing instead of matrix multiplication
        # assumes specific N.act as the antiderivative of tanh
        N = self.N
        m = self.m
        nex = x.shape[0]
        d = x.shape[1] - 1
        symA = self.A.T @ self.A

        u = []
        z = [None] * N.nTh

        opening = N.layers[0](x)
        u.append(N.act_fun(opening))
        feat = u[0]

        for i in range(1, N.nTh):
            feat = feat + N.h * N.act_fun(N.layers[i](feat))
            u.append(feat)

        tanh_open = jnp.tanh(opening) # sigma'( K_0 * S + b_0 )

        for i in range(N.nTh - 1, 0, -1):
            term = self.w.weight.T if i == N.nTh - 1 else z[i + 1]
            # z_i = z_{i+1} + h K_i' diag(...) z_{i+1}
            diag = jnp.tanh(N.layers[i](u[i - 1]))
            z[i] = term + N.h * (N.layers[i].weight.T @ diag @ term)

        z[0] = N.layers[0].weight.T @ tanh_open @ z[1]
        grad = z[0] + symA @ x.T + self.c.weight.T

        if just_grad:
            return grad

        # -----------------
        # trace of Hessian
        #-----------------

        # t_0, the trace of the opening layer
        Kopen = N.layers[0].weight[:, :d]
        temp = deriv_tanh(opening.T) * z[1]
        trH = jnp.sum(temp.reshape(m, -1, nex) * Kopen[:, :, None] ** 2, axis=(0, 1))

        temp = tanh_open.T
        Jac = Kopen[:, :, None] * temp[:, None, :]

        for i in range(1, N.nTh):
            KJ = (N.layers[i].weight @ Jac.reshape(m, -1)).reshape(m, -1, nex)
            term = self.w.weight.T if i == N.nTh - 1 else z[i + 1]
            temp = N.layers[i](u[i - 1]).T
            t_i = jnp.sum((deriv_tanh(temp) * term).reshape(m, -1, nex) * KJ**2, axis=(0, 1))
            trH += N.h * t_i
            Jac = Jac + N.h * jnp.tanh(temp).reshape(m, -1, nex) * KJ

        return grad, trH + jnp.trace(symA[:d, :d])


class OTFlow(eqx.Module):
    phi: Phi
    alph: list

    def __init(self, nTh, m, d, r=10, alph=None, key=None):
        self.alph = alph
        self.phi = Phi(nTh, m, d, r=r, alph=alph, key=key)

    def __call__(self, x, ts, nt):
        h = (ts[1] - ts[0]) / nt

        # initialize "hidden" vector to propogate with all the additional dimensions for all the ODEs
        z = self.pad(x, (0, 3, 0, 0), value=0)

        tk = ts[0]

        for k in range(nt):
            z = self.stepRK4(z, tk, tk + h)
            tk += h

        return z

    def odefun(self, x, t):
        nex, d_extra = x.shape
        d = d_extra - 3
        z = self.pad(x[:, :d], ((0, 3, 0, 0)), value=t)
        grad_phi, tr_h = self.net.trHess(z)
        dx = -(1.0 / self.alph[0]) * grad_phi[:, 0:d]
        dl = -(1.0 / self.alph[0]) * tr_h[:, None]
        dv = 0.5 * jnp.sum(dx ** 2, axis=1, keepdims=True)
        dr = jnp.abs(-grad_phi[:, [-1]] + self.alph[0] * dv)

        return jnp.concatenate((dx, dl, dv, dr), axis=1)

    def pad(self, x, pad_shape, value=0):
        h, w = x.shape
        l, r, t, b = pad_shape
        y = jnp.ones((h + t + b, w + l + r)) * value
        y[t:t+h, l:l+w] = x.copy()
        return y

    def stepRK4(self, z, t0, t1):
        """
            Runge-Kutta 4 integration scheme
        :param odefun: function to apply at every time step
        :param z:      tensor nex-by-d+4, inputs
        :param Phi:    Module, the Phi potential function
        :param alph:   list, the 3 alpha values for the OT-Flow Problem
        :param t0:     float, starting time
        :param t1:     float, end time
        :return: tensor nex-by-d+4, features at time t1
        """

        h = t1 - t0 # step size
        z0 = z

        K = h * self.odefun(z0, t0, self.phi, alph=self.alph)
        z = z0 + (1.0/6.0) * K

        K = h * self.odefun( z0 + 0.5*K, t0+(h/2), self.phi, alph=self.alph)
        z += (2.0/6.0) * K

        K = h * self.odefun( z0 + 0.5*K , t0+(h/2), self.phi, alph=self.alph)
        z += (2.0/6.0) * K

        K = h * self.odefun( z0 + K, t0+h, self.phi, alph=self.alph)
        z += (1.0/6.0) * K

        return z

    def C(self, z):
        """Expected negative log-likelihood; see Eq.(3) in the paper"""
        d = z.shape[1] - 3
        l = z[:, [d]]  # log-det
        return -jnp.sum(-0.5 * jnp.log(2 * jnp.pi) - z[:, 0:d]**2 / 2, axis=1, keepdims=True) - l

    def loss(self, z):
        # ASSUME all examples are equally weighted
        costL = jnp.mean(z[:, -2])
        costC = jnp.mean(self.C(z))
        costR = jnp.mean(z[:, -1])

        cs = [costL, costC, costR]

        # return dot(cs, alph)  , cs
        return sum(i[0] * i[1] for i in zip(cs, self.alph)), cs


class AugmentedODENet(eqx.Module):
    """Augmented Neural ODE network for continuous time flow process."""
    layers: List[Linear]
    input_shape: Tuple[int, ...]
    aug_dim: int
    nonlinearity: Callable

    def __init__(self, hidden_dims: Tuple[int, ...], input_shape: Tuple[int, ...],
                 aug_dim: int = 0, nonlinearity: str = "softplus",
                 *, key):
        self.input_shape = input_shape
        self.aug_dim = aug_dim

        # Set nonlinearity
        if nonlinearity == "softplus":
            self.nonlinearity = jnn.softplus
        elif nonlinearity == "tanh":
            self.nonlinearity = jnn.tanh
        elif nonlinearity == "relu":
            self.nonlinearity = jnn.relu
        elif nonlinearity == "elu":
            self.nonlinearity = jnn.elu
        elif nonlinearity == "swish":
            self.nonlinearity = jnn.swish
        else:
            self.nonlinearity = jnn.softplus

        # Build network layers
        input_dim = input_shape[0] + aug_dim
        dims = [input_dim] + list(hidden_dims) + [input_dim]

        keys = jr.split(key, len(dims) - 1)
        self.layers = []

        for i in range(len(dims) - 1):
            self.layers.append(Linear(dims[i], dims[i + 1], key=keys[i]))

    def __call__(self, t, y, args=None):
        """Forward pass through the augmented ODE network."""
        x = y

        # Apply layers with nonlinearity
        for i, layer in enumerate(self.layers[:-1]):
            x = layer(x)
            x = self.nonlinearity(x)

        # Final layer without activation
        x = self.layers[-1](x)

        return x


class CNF(eqx.Module):
    """Continuous Normalizing Flow with augmented dimensions."""
    odefunc: AugmentedODENet
    T: float or jax.Array
    train_T: bool
    solver: str
    rtol: float
    atol: float

    def __init__(self, odefunc: AugmentedODENet, T: float = 1.0, train_T: bool = True,
                 solver: str = "dopri5", rtol: float = 1e-3, atol: float = 1e-6):
        self.odefunc = odefunc

        self.train_T = train_T
        if self.train_T:
            self.T = jnp.array(T)
        else:
            self.T = T
        self.solver = solver
        self.rtol = rtol
        self.atol = atol

    def __call__(self, x, logdet_init, reverse=False):
        """Forward pass through the CNF."""
        if reverse:
            t_span = (self.T, 0.0)
        else:
            t_span = (0.0, self.T)

        # Augment state with log determinant
        def augmented_dynamics(t, state, args):
            y, logdet = state

            # Compute vector field
            dy_dt = self.odefunc(t, y)

            # Compute trace of Jacobian for log determinant
            def trace_fn(y):
                jac = jax.jacfwd(lambda x: self.odefunc(t, x))(y)
                return jnp.trace(jac)

            if reverse:
                dlogdet_dt = -trace_fn(y)
            else:
                dlogdet_dt = trace_fn(y)

            return dy_dt, dlogdet_dt

        # Initial state
        initial_state = (x, logdet_init)

        # Solve ODE
        term = diffrax.ODETerm(augmented_dynamics)
        if self.solver == "dopri5":
            solver = diffrax.Dopri5()
        elif self.solver == "tsit5":
            solver = diffrax.Tsit5()
        else:
            solver = diffrax.Euler()

        solution = diffrax.diffeqsolve(
            term,
            solver,
            t0=t_span[0],
            t1=t_span[1],
            dt0=-0.1 if reverse else 0.1,
            y0=initial_state,
            stepsize_controller=diffrax.PIDController(rtol=self.rtol, atol=self.atol),
        )

        final_y, final_logdet = solution.ys

        return final_y[-1], final_logdet[-1]


class SequentialCNF(eqx.Module):
    """Sequential composition of CNF blocks."""
    flows: List[CNF]

    def __init__(self, flows: List[CNF]):
        self.flows = flows

    def __call__(self, x, logdet_init, reverse=False):
        """Apply sequential flows."""
        if reverse:
            flows = reversed(self.flows)
        else:
            flows = self.flows

        current_x = x
        current_logdet = logdet_init

        for flow in flows:
            current_x, delta_logdet = flow.forward(current_x, current_logdet, reverse)
            current_logdet = delta_logdet

        return current_x, current_logdet


if __name__ == '__main__':
    model = RealNVP(4, 2, 64, 3)
    a = jr.normal(jr.key(2), (64, 2))
    z, logJ = jax.vmap(model)(a)