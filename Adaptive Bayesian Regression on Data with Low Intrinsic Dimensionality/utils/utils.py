import argparse
import os
import time
from PIL import Image
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
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

# create artificial regression dataset
def get_data(N=30, sigma_obs=0.15, N_test=400):
    np.random.seed(0)
    X = jnp.linspace(-1, 1, N)
    Y = X + 0.2 * jnp.power(X, 3.0) + 0.5 * jnp.power(0.5 + X, 2.0) * jnp.sin(4.0 * X)
    Y += sigma_obs * np.random.randn(N)
    Y -= jnp.mean(Y)
    Y /= jnp.std(Y)

    assert X.shape == (N,)
    assert Y.shape == (N,)

    X_test = jnp.linspace(-1.3, 1.3, N_test)
    Y_test = X_test + 0.2 * jnp.power(X_test, 3.0) + 0.5 * jnp.power(0.5 + X_test, 2.0) * jnp.sin(4.0 * X_test)
    return X, Y, X_test, Y_test

def swill_roll_data_generate(n_data = 200, noise_add = True, sigma_noise = 0.1):
    swiss_roll = np.zeros((3, n_data))
    U = np.random.uniform(low=3*np.pi/2,high = 9*np.pi/2, size= n_data)
    V = np.random.uniform(low=0,high = 20, size= n_data)
    swiss_roll[0,:] = U * np.cos(U)
    swiss_roll[1,:] = V
    swiss_roll[2,:] =  U*np.sin(U)
    Y = 4 * (U/(3*np.pi)- (1+3*np.pi)/2)**2 + V*np.pi/20
    if noise_add:
        Y +=  np.random.randn(n_data) * sigma_noise
    Omega = np.random.rand(100, 3)
    swiss_roll_embed = Omega @ swiss_roll
    return swiss_roll_embed.T, Y

def swill_roll_non_embed(n_data = 200, noise_add = True, sigma_noise = 0.1):
    swiss_roll = np.zeros((3, n_data))
    U = np.random.uniform(low=3*np.pi/2,high = 9*np.pi/2, size= n_data)
    V = np.random.uniform(low=0,high = 20, size= n_data)
    swiss_roll[0,:] = U * np.cos(U)
    swiss_roll[1,:] = V
    swiss_roll[2,:] =  U*np.sin(U)
    Y = 4 * (U/(3*np.pi)- (1+3*np.pi)/2)**2 + V*np.pi/20
    Y_noise = Y
    if noise_add:
        Y_noise +=  np.random.randn(n_data) * sigma_noise
    #Omega = np.random.randn(100, 3)
    swiss_roll_embed =swiss_roll
    return swiss_roll_embed.T, Y, Y_noise

def generate_figure_data(num = 72, obj_name = 'obj4__',file_path = '/Users/tt/Desktop/Git Codes/Research/Gp:Kernel Regression on Unknown Manifold/Data/coil-20-proc/'):
    data = np.zeros((num, 128,128))
    for figure_num in range(num):
        data[figure_num,:,:] = np.array(Image.open(file_path+ obj_name + str(figure_num) + '.png'))
    return data

def swill_roll_non_embed_f(n_data = 200, noise_add = True, sigma_noise = 0.1, f = None):
    swiss_roll = np.zeros((3, n_data))
    U = np.random.uniform(low=2*np.pi/2,high = 9*np.pi/2, size= n_data)
    V = np.random.uniform(low=0,high = 15, size= n_data)
    swiss_roll[0,:] = U * np.cos(U)
    swiss_roll[1,:] = V
    swiss_roll[2,:] =  U*np.sin(U)
    Y = f(U,V)
    Y_noise = Y.copy()
    if noise_add:
        Y_noise +=  np.random.randn(n_data) * sigma_noise
    #Omega = np.random.randn(100, 3)
    swiss_roll_embed =swiss_roll
    return swiss_roll_embed.T, Y, Y_noise

def swill_roll_non_embed_nuniform_f(n_data = 200, noise_add = True, sigma_noise = 0.1, f = None):
    swiss_roll = np.zeros((3, n_data))
    U = np.random.uniform(low=(2*np.pi/2)**2,high = (9*np.pi/2)**2, size= n_data)
    U = np.sqrt(U)
    V = np.random.uniform(low=0,high = 15, size= n_data)
    swiss_roll[0,:] = U * np.cos(U)
    swiss_roll[1,:] = V
    swiss_roll[2,:] =  U*np.sin(U)
    Y = f(U,V)
    Y_noise = Y.copy()
    if noise_add:
        Y_noise +=  np.random.randn(n_data) * sigma_noise
    #Omega = np.random.randn(100, 3)
    swiss_roll_embed =swiss_roll
    return swiss_roll_embed.T, Y, Y_noise

def swill_roll_line_non_embed_f(n_data = 200, noise_add = True, sigma_noise = 0.1, p = 0.5, f = None):
    swiss_roll = np.zeros((3, n_data))
    n_swiss = np.random.binomial(n_data, p)
    U = np.random.uniform(low=2*np.pi/2,high = 9*np.pi/2, size= n_swiss)
    V = np.random.uniform(low=0,high = 15, size= n_swiss)

    swiss_roll[0,:n_swiss] = U * np.cos(U)
    swiss_roll[1,:n_swiss] = V
    swiss_roll[2,:n_swiss] =  U*np.sin(U)
    swiss_roll[0,n_swiss:] = np.random.uniform(low= -9 *np.pi/2,high = 9*np.pi/2, size=  n_data - n_swiss)
    swiss_roll[1,n_swiss:] = (swiss_roll[0,n_swiss:] + 9*np.pi/2) * 5/(3*np.pi)
    swiss_roll[2,n_swiss:] = swiss_roll[0,n_swiss:]
    U_1 = np.sqrt(swiss_roll[0,n_swiss:] ** 2 + swiss_roll[2,n_swiss:] ** 2)
    V_1 = swiss_roll[1,n_swiss:]
    Y = np.zeros(n_data)
    Y[:n_swiss] = f(U,V)
    Y[n_swiss:] = f(U_1,V_1)
    Y_noise = Y.copy()
    if noise_add:
        Y_noise +=  np.random.randn(n_data) * sigma_noise
    #Omega = np.random.randn(100, 3)
    swiss_roll_embed =swiss_roll
    return swiss_roll_embed.T, Y, Y_noise


def swill_roll_curve_non_embed_f(n_data = 200, noise_add = True, sigma_noise = 0.1, p = 0.5, f = None):
    swiss_roll = np.zeros((3, n_data))
    n_swiss = np.random.binomial(n_data, p)
    U = np.random.uniform(low=2*np.pi/2,high = 9*np.pi/2, size= n_swiss)
    V = np.random.uniform(low=0,high = 15, size= n_swiss)

    swiss_roll[0,:n_swiss] = U * np.cos(U)
    swiss_roll[1,:n_swiss] = V
    swiss_roll[2,:n_swiss] =  U*np.sin(U)
    t = np.random.uniform(low= -1,high = 1, size=  n_data - n_swiss)
    swiss_roll[0,n_swiss:] = 7*np.pi/2 *np.cos(np.pi * t) * np.cos(4 * np.pi * t)
    swiss_roll[1,n_swiss:] = 7*np.pi/2 + 7*np.pi/2 *np.cos(np.pi * t) * np.sin(4 * np.pi * t)
    swiss_roll[2,n_swiss:] = 7*np.pi/2 *np.sin(np.pi * t)
    U_1 = np.sqrt(swiss_roll[0,n_swiss:] ** 2 + swiss_roll[2,n_swiss:] ** 2)
    V_1 = swiss_roll[1,n_swiss:]
    Y = np.zeros(n_data)
    Y[:n_swiss] = f(U,V)
    Y[n_swiss:] = f(U_1,V_1)
    Y_noise = Y.copy()
    if noise_add:
        Y_noise +=  np.random.randn(n_data) * sigma_noise
    #Omega = np.random.randn(100, 3)
    swiss_roll_embed =swiss_roll
    return swiss_roll_embed.T, Y, Y_noise

# intersect at x =0, z = 4/\sqrt(3), U = 7*np.pi/3
def test_intersection_f(n_data = 200, noise_add = True, sigma_noise = 0.1, delta = 0.05,f = None):
    swiss_roll = np.zeros((3, n_data))
    U = np.ones(n_data) * 7 *np.pi/3 + np.random.uniform(low = -delta ,high = delta, size= n_data)
    V = np.random.uniform(low=0,high = 15, size= n_data)
    swiss_roll[0,:] = U * np.cos(U) - 7*np.pi/6
    swiss_roll[1,:] = V
    swiss_roll[2,:] =  U*np.sin(U)
    Y = f(U,V)
    Y_noise = Y.copy()
    if noise_add:
        Y_noise +=  np.random.randn(n_data) * sigma_noise
    #Omega = np.random.randn(100, 3)
    swiss_roll_embed =swiss_roll
    return swiss_roll_embed.T, Y, Y_noise

# intersect at x =0, z = 4/\sqrt(3), U = 7*np.pi/3
def test_intersection_f_2(n_data = 200, noise_add = True, sigma_noise = 0.1, delta = 0.05,f = None):
    swiss_roll = np.zeros((3, n_data))
    U = np.ones(n_data) * 7 *np.pi/3 + np.random.uniform(low = -delta ,high = delta, size= n_data)
    V = np.random.uniform(low=0,high = 15, size= n_data)
    swiss_roll[0,:] = -U * np.cos(U) + 7*np.pi/6
    swiss_roll[1,:] = V
    swiss_roll[2,:] =  U*np.sin(U)
    Y = f(U,V)
    Y_noise = Y.copy()
    if noise_add:
        Y_noise +=  np.random.randn(n_data) * sigma_noise
    #Omega = np.random.randn(100, 3)
    swiss_roll_embed =swiss_roll
    return swiss_roll_embed.T, Y, Y_noise

def boundary_f(n_data = 200, noise_add = True, sigma_noise = 0.1, p = 0.5, f = None):
    swiss_roll = np.zeros((3, n_data))
    n_swiss = np.random.binomial(n_data, p)

    #U = np.random.uniform(low=4*np.pi/2,high = 5*np.pi/2, size= n_swiss)
    U = np.random.choice([4*np.pi/2,5*np.pi/2], n_swiss)
    V = np.random.uniform(low=0,high = 15, size= n_swiss)
    swiss_roll[0,:n_swiss] = U * np.cos(U) - 7*np.pi/6
    swiss_roll[1,:n_swiss] = V
    swiss_roll[2,:n_swiss] =  U*np.sin(U)

    U_1 = np.random.uniform(low=4*np.pi/2,high = 5*np.pi/2, size= n_data - n_swiss)
    #V_1 = np.random.uniform(low=0,high = 15, size= n_data - n_swiss)
    V_1 = np.random.choice([0, 15], n_data - n_swiss)
    swiss_roll[0,n_swiss:] = U_1 * np.cos(U_1) - 7*np.pi/6
    swiss_roll[1,n_swiss:] = V_1
    swiss_roll[2,n_swiss:] = U_1*np.sin(U_1)
    Y = np.zeros(n_data)
    Y[:n_swiss] = f(U,V)
    Y[n_swiss:] = f(U_1,V_1)
    Y_noise = Y.copy()
    if noise_add:
        Y_noise +=  np.random.randn(n_data) * sigma_noise
    #Omega = np.random.randn(100, 3)
    swiss_roll_embed =swiss_roll
    return swiss_roll_embed.T, Y, Y_noise

def boundary_f_2(n_data = 200, noise_add = True, sigma_noise = 0.1, p = 0.5, f = None):
    swiss_roll = np.zeros((3, n_data))
    n_swiss = np.random.binomial(n_data, p)

    #U = np.random.uniform(low=4*np.pi/2,high = , size= n_swiss)
    U = np.random.choice([4*np.pi/2,5*np.pi/2], n_swiss)
    V = np.random.uniform(low=0,high = 15, size= n_swiss)
    swiss_roll[0,:n_swiss] = -U * np.cos(U) + 7*np.pi/6
    swiss_roll[1,:n_swiss] = V
    swiss_roll[2,:n_swiss] =  U*np.sin(U)

    U_1 = np.random.uniform(low=4*np.pi/2,high = 5*np.pi/2, size= n_data - n_swiss)
    #V_1 = np.random.uniform(low=0,high = 15, size= n_data - n_swiss)
    V_1 = np.random.choice([0, 15], n_data - n_swiss)
    swiss_roll[0,n_swiss:] = -U_1 * np.cos(U_1) + 7*np.pi/6
    swiss_roll[1,n_swiss:] = V_1
    swiss_roll[2,n_swiss:] = U_1*np.sin(U_1)
    Y = np.zeros(n_data)
    Y[:n_swiss] = f(U,V)
    Y[n_swiss:] = f(U_1,V_1)
    Y_noise = Y.copy()
    if noise_add:
        Y_noise +=  np.random.randn(n_data) * sigma_noise
    #Omega = np.random.randn(100, 3)
    swiss_roll_embed =swiss_roll
    return swiss_roll_embed.T, Y, Y_noise



def generate_figure_data(num = 72, obj_name = 'obj4__',file_path = '/Users/tt/Desktop/Git Codes/Research/Gp:Kernel Regression on Unknown Manifold/Data/coil-20-proc/'):
    data = np.zeros((num, 128,128))
    for figure_num in range(num):
        data[figure_num,:,:] = np.array(Image.open(file_path+ obj_name + str(figure_num) + '.png'))
    return data

lucky_cat_data = generate_figure_data()
f_0 = np.cos(2 * np.pi * np.arange(0,72)/72)

def cat_train_test(train_size = 18, sigma_noise = 0.1, f_0 = f_0):
    idx = np.arange(72)
    np.random.shuffle(idx)
    train_idx, test_idx = idx[:train_size], idx[train_size:]
    X, X_test = lucky_cat_data[train_idx,:].reshape(train_size, -1), lucky_cat_data[test_idx,:].reshape(72-train_size, -1)
    Y, Y_test = f_0[train_idx] + sigma_noise * np.random.randn(train_size) , f_0[test_idx]
    return X, X_test, Y, Y_test