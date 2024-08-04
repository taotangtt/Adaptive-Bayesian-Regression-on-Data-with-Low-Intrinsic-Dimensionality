import argparse
import os
import time
import scipy
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

import jax
from jax import vmap
import jax.numpy as jnp
import jax.random as random

import numpyro
import numpyro.distributions as dist
from numpyro.infer import (
    MCMC,
    NUTS,
    init_to_feasible,
    init_to_median,
    init_to_sample,
    init_to_uniform,
    init_to_value,
)

matplotlib.use("Agg")  # noqa: E402
dimension = 2

# squared exponential kernel with diagonal noise term
# def kernel(X, Z, var, length, noise, jitter=1.0e-6, include_noise=True):
#     deltaXsq = jnp.power((X[:, None] - Z), 2.0)/ length
#     k = var * jnp.exp(-0.5 * deltaXsq)
#     if include_noise:
#         k += (noise + jitter) * jnp.eye(X.shape[0])
#     return k


def kernel(X, Z, var, length, noise, jitter=1.0e-6, include_noise=True):
    # Compute pairwise distances between input points
    #start = time.time()
    dists = scipy.spatial.distance.cdist(X, Z)
    #print(f"distanceltime {time.time() - start}")
    # Compute squared exponential kernel
    k = var * jnp.exp(-0.5 * jnp.power(dists, 2.0)/ length)
    #print(f"computetime1 {time.time() - start}")
    # Add diagonal noise term
    if include_noise:
        k += (noise + jitter) * jnp.eye(X.shape[0])
    #print(f"computetime2 {time.time() - start}")
    return k

# def nth_root_transform(n):
#     def transform(x):
#         return jnp.power(x, 1.0/n)
#     return transform


def model(X, Y, n = dimension):
    # set uninformative log-normal priors on our three kernel hyperparameters
    #var = numpyro.sample("kernel_var", dist.LogNormal(0.0, 10.0))
    var = 1
    #noise = numpyro.sample("kernel_noise", dist.LogNormal(0.0, 10.0))
    noise = 0.03
    #length = numpyro.sample("kernel_length", dist.InverseGamma(3, rate=1.0, validate_args=None))
    length_n = numpyro.sample("kernel_length", dist.InverseGamma(3, rate=1.0, validate_args=None))
    length =  jnp.power(length_n, 2/n)
    #     n = 5
    #     length = numpyro.sample("nth_root_x", dist.TransformedDistribution(dist.LogNormal(0.0, 10.0), nth_root_transform(n)))
    # compute kernel
    k = kernel(X, X, var, length, noise)

    # sample Y according to the standard gaussian process formula
    numpyro.sample(
        "Y",
        dist.MultivariateNormal(loc=jnp.zeros(X.shape[0]), covariance_matrix=k),
        obs=Y,
    )
    return length


# helper function for doing hmc inference
def run_inference(model, args, rng_key, X, Y):
    start = time.time()
    # demonstrate how to use different HMC initialization strategies
    if args.init_strategy == "value":
        init_strategy = init_to_value(
            #values={"kernel_var": 1.0, "kernel_noise": 0.05, "kernel_length": 0.5}
            values={"kernel_length": 0.5}
        )
    elif args.init_strategy == "median":
        init_strategy = init_to_median(num_samples=10)
    elif args.init_strategy == "feasible":
        init_strategy = init_to_feasible()
    elif args.init_strategy == "sample":
        init_strategy = init_to_sample()
    elif args.init_strategy == "uniform":
        init_strategy = init_to_uniform(radius=1)
    kernel = NUTS(model, init_strategy=init_strategy)
    mcmc = MCMC(
        kernel,
        num_warmup=args.num_warmup,
        num_samples=args.num_samples,
        num_chains=args.num_chains,
        thinning=args.thinning,
        progress_bar=False if "NUMPYRO_SPHINXBUILD" in os.environ else True,
    )
    mcmc.run(rng_key, X, Y)
    mcmc.print_summary()
    print("\nMCMC elapsed time:", time.time() - start)
    return mcmc.get_samples()


# do GP prediction for a given set of hyperparameters. this makes use of the well-known
# formula for gaussian process predictions
def predict(rng_key, X, Y, X_test, var, length, noise):
    # compute kernels between train and test data, etc.
    k_pp = kernel(X_test, X_test, var, length, noise, include_noise=True)
    k_pX = kernel(X_test, X, var, length, noise, include_noise=False)
    k_XX = kernel(X, X, var, length, noise, include_noise=True)
    K_xx_inv = jnp.linalg.inv(k_XX)
    K = k_pp - jnp.matmul(k_pX, jnp.matmul(K_xx_inv, jnp.transpose(k_pX)))
    sigma_noise = jnp.sqrt(jnp.clip(jnp.diag(K), a_min=0.0)) * jax.random.normal(
        rng_key, X_test.shape[:1]
    )
    mean = jnp.matmul(k_pX, jnp.matmul(K_xx_inv, Y))
    # we return both the mean function and a sample from the posterior predictive for the
    # given set of hyperparameters
    return mean, mean + sigma_noise

# do GP mean prediction for a given set of hyperparameters. this makes use of the well-known
# formula for gaussian process predictions
#@nb.jit
def predict_mean(rng_key, X, Y, X_test, var, length, noise):
    # compute kernels between train and test data, etc
    #start = time.time()
    k_pX = kernel(X_test, X, var, length, noise, include_noise=False)
    #print(f"kerneltime {time.time() - start}")
    k_XX = kernel(X, X, var, length, noise, include_noise=True)
    #print(f"kerneltime_2 {time.time() - start}")
    mean = jnp.matmul(k_pX, jnp.linalg.solve(k_XX, Y))
    #print(f"Sovlingtime {time.time() - start}")
    return mean

def fill_diagonal(a, val):
    assert a.ndim >= 2
    i, j = jnp.diag_indices(min(a.shape[-2:]))
    return a.at[..., i, j].set(val)

def kernel_fromdists(dists, var, length, noise, jitter=1.0e-6, include_noise=True):
    k = dists ** 2 #np.power(dists, 2)
    k = -0.5 *k/length
    k = jnp.exp(k)
    # Add diagonal noise term
    if include_noise:
        #k += (noise + jitter) * jnp.eye(dists.shape[0])
        k = fill_diagonal(k, (1+jitter+noise)*jnp.ones(k.shape[0]))
    #print(f"computetime2 {time.time() - start}")
    return k

def predict_mean_fromdists(rng_key, dists_XX, Y, dists_pX, var, length, noise):
    # compute kernels between train and test data, etc
    #start = time.time()
    k_pX = kernel_fromdists(dists_pX, var, length, noise, include_noise=False)
    #print(f"kerneltime {time.time() - start}")
    k_XX = kernel_fromdists(dists_XX, var, length, noise, include_noise=True)
    #print(f"kerneltime_2 {time.time() - start}")
    mean = jnp.matmul(k_pX, jnp.linalg.solve(k_XX, Y))
    #print(f"Sovlingtime {time.time() - start}")
    return mean




