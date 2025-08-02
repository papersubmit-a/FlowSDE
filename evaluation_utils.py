import numpy as np
import math
import matplotlib
matplotlib.use("WebAgg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
import argparse

def plot_results_ode(ts, xs_dot, xs_dot_est, xs_dot_pred, xs, xs_pred, filename=None):
    x_size = xs_pred.shape[-1]
    n_rows = int(math.sqrt(x_size * 2))
    if (x_size * 2) % n_rows == 0:
        n_columns = (x_size * 2) // n_rows
        rest = 0
    else:
        n_columns = (x_size * 2) // n_rows + 1
        rest = n_rows * n_columns - x_size * 2
    last_row_id = [x_size * 2 - 1 - k for k in range(n_columns - rest)] + \
                  [x_size * 2 - j for j in range(n_columns - rest + 1, n_columns + 1)]
    fig, ax = plt.subplots(n_rows, n_columns, figsize=(3 * n_columns, 3 * n_rows))
    if n_rows > 1:
        ax = ax.flat
    else:
        if n_columns == 1:
            ax = [ax]
    if rest > 0:
        for j in range(rest):
            ax[-1 - j].set_visible(False)
    for i in range(x_size):
        ax[2 * i].set_title(r"$\dot{x}$" + rf"$_{i}$")
        ax[2 * i].plot(ts, xs_dot[:, i], c="dodgerblue")
        ax[2 * i].plot(ts, xs_dot_est[:, i], c="forestgreen")
        ax[2 * i].plot(ts, xs_dot_pred[:, i], c="crimson")
        ax[2 * i + 1].legend(['drift_ref', 'drift_est', 'drift_pred'])
        ax[2 * i + 1].set_title(r"$x$" + rf"$_{i}$")
        ax[2 * i + 1].plot(ts, xs[:, i], c="dodgerblue")
        ax[2 * i + 1].plot(ts, xs_pred[:, i], c="crimson")
        if 2 * i in last_row_id:
            ax[2 * i].set_xlabel('time')
        else:
            ax[2 * i].set_xticks([])
        if 2 * i + 1 in last_row_id:
            ax[2 * i + 1].set_xlabel('time')
        else:
            ax[2 * i + 1].set_xticks([])

    if filename is not None:
        plt.savefig(filename)
        plt.close()
    # plt.show()

def plot_results_sde(ts, dxs_ref, dxs_est, dxs_pred, drift_ref, drift_est, drift_pred,
                     diff_ref, diff_est, diff_pred, xs, xs_pred, xs_pred_sample, filename=None):
    x_size = xs_pred.shape[-1]
    n_rows = int(math.sqrt(x_size * 4))
    if (x_size * 4) % n_rows == 0:
        n_columns = (x_size * 4) // n_rows
        rest = 0
    else:
        n_columns = (x_size * 4) // n_rows + 1
        rest = n_rows * n_columns - x_size * 4
    last_row_id = [x_size * 4 - 1 - k for k in range(n_columns - rest)] + \
                  [x_size * 4 - j for j in range(n_columns - rest + 1, n_columns + 1)]
    fig, ax = plt.subplots(n_rows, n_columns, figsize=(3 * n_columns, 3 * n_rows))
    if n_rows > 1:
        ax = ax.flat
    else:
        if n_columns == 1:
            ax = [ax]
    if rest > 0:
        for j in range(rest):
            ax[-1 - j].set_visible(False)

    for i in range(x_size):
        ax[4 * i].set_title(r"$\dot{x}$" + rf"$_{i}$" + r"$=f$" + rf"$_{i}$" + r"$(x,t)+$" +
                            rf"$g_{i}$" + r"$(x,t)\circ \dot W$")
        ax[4 * i].plot(ts, dxs_ref[:, i], c="dodgerblue")
        ax[4 * i].plot(ts, dxs_est[:, i], c="forestgreen")
        ax[4 * i].plot(ts, dxs_pred[:, i], c="crimson")
        ax[4 * i].legend(['dx_ref', 'dx_est', 'dx_pred'])
        ax[4 * i + 1].set_title(r"$f$" + rf"$_{i}$" + r"$(x, t)$")
        ax[4 * i + 1].plot(ts, drift_ref[:, i], c="dodgerblue")
        ax[4 * i + 1].plot(ts, drift_est[:, i], c="forestgreen")
        ax[4 * i + 1].plot(ts, drift_pred[:, i], c="crimson")
        ax[4 * i + 1].legend(['drift_ref', 'drift_est', 'drift_pred'])
        ax[4 * i + 2].set_title(rf"$g_{i}$" + r"$(x,t)\circ \dot W$")
        ax[4 * i + 2].plot(ts, diff_ref[:, 0], c="dodgerblue")
        ax[4 * i + 2].plot(ts, diff_est[:, 0], c="forestgreen")
        ax[4 * i + 2].plot(ts, diff_pred[:, 0], c="crimson")
        ax[4 * i + 2].legend(['diff_ref', 'diff_est', 'diff_pred'])
        ax[4 * i + 3].set_title(r"$x$" + rf"$_{i}$")
        ax[4 * i + 3].plot(ts, xs[:, i], c="dodgerblue")
        ax[4 * i + 3].plot(ts, xs_pred[:, i], c="crimson")
        ax[4 * i + 3].legend(['ref', 'pred'])
        for j in range(xs_pred_sample.shape[0]):
            ax[4 * i + 3].plot(ts, xs_pred_sample[j, :, i], c="crimson", alpha=0.1, label="Model")
        for k in range(4):
            if 4 * i + k in last_row_id:
                ax[4 * i + k].set_xlabel('time')
            else:
                ax[4 * i + k].set_xticks([])

    if filename is not None:
        plt.savefig(filename)
        plt.close()

def plot_results_sde_single(ts, xs, xs_pred, xs_pred_sample, filename=None):
    x_size = xs_pred.shape[-1]
    n_rows = int(math.sqrt(x_size))
    if x_size % n_rows == 0:
        n_columns = x_size // n_rows
        rest = 0
    else:
        n_columns = x_size // n_rows + 1
        rest = n_rows * n_columns - x_size
    last_row_id = [x_size - 1 - k for k in range(n_columns - rest)] + \
                  [x_size - j for j in range(n_columns - rest + 1, n_columns + 1)]
    fig, ax = plt.subplots(n_rows, n_columns, figsize=(3 * n_columns, 3 * n_rows))
    if n_rows > 1:
        ax = ax.flat
    else:
        if n_columns == 1:
            ax = [ax]
    if rest > 0:
        for j in range(rest):
            ax[-1 - j].set_visible(False)

    for i in range(x_size):
        ax[i].set_title(r"$x$" + rf"$_{i}$")
        ax[i].plot(ts, xs[:, i], c="dodgerblue")
        ax[i].plot(ts, xs_pred[:, i], c="crimson")
        ax[i].legend(['ref', 'pred'])
        for j in range(xs_pred_sample.shape[0]):
            ax[i].plot(ts, xs_pred_sample[j, :, i], c="crimson", alpha=0.1, label="Model")
        ax[i].set_xlabel('time')

    if filename is not None:
        plt.savefig(filename)
        plt.close()


def plot_sampled_results_derivative(ts, xs, xs_preds, fs, fs_preds, gs, gs_preds, filename=None):
    x_size = xs.shape[-1]
    fig = plt.figure(figsize=(6 * x_size, 3 * 3))
    ax_gs = gridspec.GridSpec(3, x_size+1, width_ratios=[1]*x_size+[0.05], wspace=0.1)
    ax1 = [fig.add_subplot(ax_gs[0, i]) for i in range(x_size)]
    ax2 = [fig.add_subplot(ax_gs[1, i]) for i in range(x_size)]
    ax3 = [fig.add_subplot(ax_gs[2, i]) for i in range(x_size)]
    cax = fig.add_subplot(ax_gs[2, -1])
    fs_mean = fs_preds.mean(0)
    fs_std = fs_preds.std(0)
    fs_l = fs_mean - fs_std
    fs_u = fs_mean + fs_std
    gs_std = gs_preds.std(0)
    gs_mean = gs_preds.mean(0)
    norm = Normalize(vmin=gs_std.min(), vmax=gs_std.max())
    cmap = plt.cm.coolwarm
    for i in range(x_size):
        ax1[i].set_title(r"$x$" + rf"$_{i}$")
        ax1[i].plot(ts, xs[:, i], c="dodgerblue")
        for j in range(xs_preds.shape[0]):
            ax1[i].plot(ts, xs_preds[j, :, i], c="crimson", alpha=0.7, lw=0.1)

        ax2[i].set_title(r"$f(x$" + rf"$_{i}$" + r"$,t)$")
        if fs is not None:
            ax2[i].plot(ts, fs[:, i], c="dodgerblue", label='reference')
        ax2[i].plot(ts, fs_mean[:, i], c="crimson", label='model')
        ax2[i].fill_between(ts, fs_l[:, i], fs_u[:, i], color="crimson", alpha=0.3, label='±1 standard Deviation')
        ax2[i].legend(loc='best')
        if gs is not None:
            if gs.shape[-1] == 1:
                ax3[i].plot(ts, gs, c="dodgerblue", alpha=0.2)
            else:
                ax3[i].plot(ts, gs[..., i], c="dodgerblue", alpha=0.2)
        ax3[i].plot(ts, gs_mean[..., i], c="crimson", alpha=0.5)
        ax3[i].set_title(r"$g(x$" + rf"$_{i}$" + r"$,t)dB_t$")
        for j in np.arange(0, len(ts), 10):
            color = cmap(norm(gs_std[j, i]))
            ax3[i].errorbar(ts[j], gs_mean[j, i], yerr=gs_std[j, i], fmt='o', color=color, ecolor=color, capsize=0.1)
        # Attach colorbar to correct axes
        ax3[i].set_xlabel('time')
    # Create ScalarMappable for colorbar (even though not plotted)
    sm = ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])  # Needed for compatibility
    cbar = fig.colorbar(sm, cax=cax)
    cbar.set_label("Standard Deviation")

    if filename is None:
        plt.show()
    else:
        plt.savefig(filename)
        plt.close()

def gaussian_nll(observations, forecasts):
    """
    pred_samples: shape (num_samples, batch, time, dim)
    target: shape (batch, time, dim)
    """
    mu = np.mean(forecasts, axis=0)  # (batch, time, dim)
    var = np.var(forecasts, axis=0) + 1e-6  # to avoid log(0)

    # Gaussian NLL (diagonal covariance)
    diff = observations - mu
    nll = 0.5 * (np.log(2 * np.pi * var) + (diff ** 2) / var)
    return nll

def crps_ensemble_vectorized(observations, forecasts):
    """
    An alternative but simpler implementation of CRPS for testing purposes

    This implementation is based on the identity:

    .. math::
        CRPS(F, x) = E_F|X - x| - 1/2 * E_F|X - X'|

    where X and X' denote independent random variables drawn from the forecast
    distribution F, and E_F denotes the expectation value under F.

    Hence it has runtime O(n^2) instead of O(n log(n)) where n is the number of
    ensemble members.

    Reference
    ---------
    Tilmann Gneiting and Adrian E. Raftery. Strictly proper scoring rules,
        prediction, and estimation, 2005. University of Washington Department of
        Statistics Technical Report no. 463R.
        https://www.stat.washington.edu/research/reports/2004/tr463R.pdf
    """
    observations = np.asarray(observations)
    forecasts = np.asarray(forecasts)

    if observations.ndim == forecasts.ndim - 1:
        assert observations.shape == forecasts.shape[1:]
        score = np.nanmean(abs(forecasts - observations), 0)
        # insert new axes along last and second to last forecast dimensions so
        # forecasts_diff expands with the array broadcasting
        forecasts_diff = (np.expand_dims(forecasts, 0) -
                          np.expand_dims(forecasts, 1))
        score += -0.5 * np.nanmean(abs(forecasts_diff),axis=(0, 1))
        return score
    elif observations.ndim == forecasts.ndim:
        # there is no 'realization' axis to sum over (this is a deterministic
        # forecast)
        return abs(observations - forecasts)
    return None

def compute_mean_ci_width(trajectories, ci=0.95):
    """
    trajectories: array of shape (num_samples, num_timesteps)
    ci: confidence interval level, e.g., 0.95
    """
    alpha = 1 - ci
    lower_percentile = 100 * (alpha / 2)
    upper_percentile = 100 * (1 - alpha / 2)

    # Compute lower and upper bounds at each time step
    lower = np.percentile(trajectories, lower_percentile, axis=0)
    upper = np.percentile(trajectories, upper_percentile, axis=0)

    # CI width at each time point
    ci_widths = upper - lower

    # Mean over time
    mean_ci_width = np.mean(ci_widths)
    return mean_ci_width