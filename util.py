from scipy.stats import binned_statistic
from matplotlib import pyplot as plt
import numpy as np
import pystan
import pickle

def plot_answer_prob_curve(df, bins=None, ax=None):
    """
    Plot

    Parameters
    ----------
    ax: matplotlib.Axes
    bins: int | list

    Returns
    -------
    ax
    """
    if ax is None:
        fig = plt.figure(figsize=(12, 12))
        ax = fig.add_subplot(111)

    if bins is None:
        bins = 20

    rdiffs = df['difference_relative'].values
    answers = df['answer'].values

    bin_centers, bin_edges, _bin_numbers = binned_statistic(rdiffs, rdiffs, statistic="mean", bins=bins)
    bin_means, bin_edges, _bin_numbers= binned_statistic(rdiffs, answers, statistic="mean", bins=bins)
    bin_stds, bin_edges, _bin_numbers = binned_statistic(rdiffs, answers, statistic="std", bins=bins)
    bin_counts, bin_edges, _bin_numbers = binned_statistic(rdiffs, answers, statistic="count", bins=bins)
    bin_sems = bin_stds / np.sqrt(bin_counts)

    ax.errorbar(bin_centers, bin_means, yerr=bin_sems, color="C1",
                label="Mean answer, errorbars = standard error of the mean")
    ax.set_xlabel('Relative difference (A - B) / min(A,B)')
    ax.set_ylabel('Answer (0=A, 1=B)')
    ax.axhline(0.5, color='k', ls="--")
    ax.axvline(0.0, color='k', ls="--")
    ax.legend(loc="best")
    ax.grid()
    return ax


STAN_MODELS = dict()

STAN_LINEAR_MODEL_TEMPLATE = \
"""
data {
    int N; // number of samples
    vector[N] rdiffs; // relative differences
    int answers[N]; // estimated treatment effects
}
parameters {
    real<lower={y_intercept_min}, upper={y_intercept_max}> y_intercept;
    real<lower={slope_min}, upper={slope_max}> slope;
}
transformed parameters {
    real x_intercept = -(y_intercept-0.5)/slope;
    vector[N] raw_probs = y_intercept + slope * rdiffs;
    vector[N] mod_probs;
    for (i in 1:N) {
        mod_probs[i] = min([max([raw_probs[i],0.01]),0.99]);
    }
}
model {
    answers ~ bernoulli(mod_probs);
}
"""

def get_linear_stan_model(model_params, verbose=True, recreate=False):
    key = "lin_" + "_".join([str(el) for el in
                             [model_params['y_intercept_min'], model_params['y_intercept_max'],
                             model_params['slope_min'], model_params['slope_max'], verbose]])
    fname = key + "_stan.pickle"
    try:
        if recreate:
            raise RuntimeError("Forcing recreation")
        with open(fname, "rb") as f:
            stan_model = pickle.load(f)
            return stan_model

    except (RuntimeError, FileNotFoundError) as e:
        # apply parameters
        stan_linear_fit_code = STAN_LINEAR_MODEL_TEMPLATE
        for param in ['y_intercept_min', 'y_intercept_max', 'slope_min', 'slope_max']:
            stan_linear_fit_code = stan_linear_fit_code.replace("{" + param + "}", str(model_params[param]))
        if verbose:
            print(stan_linear_fit_code)
        stan_model = pystan.StanModel(model_code=stan_linear_fit_code, verbose=verbose)
        with open(fname, "wb") as f:
            pickle.dump(stan_model, f)
        return stan_model


def fit_linear_stan(relative_differences, answers, n_iter=1000, n_chains=4,
                    y_intercept_min=0.4, y_intercept_max=1,
                    slope_min=-20, slope_max=20, verbose=False):
    model_params = dict(y_intercept_max=y_intercept_max, y_intercept_min=y_intercept_min,
                        slope_min=slope_min, slope_max=slope_max)
    stan_model = get_linear_stan_model(model_params, verbose=verbose)
    sampling_data = {
        'N': len(relative_differences),
        'rdiffs': relative_differences,
        'answers': answers
    }
    fit = stan_model.sampling(data=sampling_data, iter=n_iter, chains=4, verbose=verbose)
    return fit



