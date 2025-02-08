import numpy as np
import torch
from torch import nn
from torch.functional import F
import matplotlib.pyplot as plt
import time
import pytorch_lightning as pl
from collections import OrderedDict
import os
import config

import hal.models as models
import hal.losses as losses
import hal.metrics as metrics
import matplotlib.colors as pltc
import scipy.io as sio
from scipy.linalg import eigh
from sklearn.manifold import TSNE
import scipy.stats as ss
import pdb

args = config.parse_args()

dataloader = getattr(datasets, args.dataset)(args)


class GDOPT(torch.nn.Module):

    def __init__(self, H, L_Y, L_S, epsilon, p, q, n, beta=5000, theta=None):
        super(GDOPT, self).__init__()
        # self.K_Z = K_Z
        self.H = H
        self.L_Y = L_Y
        self.L_S = L_S
        self.epsilon = epsilon
        self.p = p
        self.q = q
        if not theta:
            theta = torch.rand(p, q)
        theta.requires_grad = True
        self.theta = nn.Parameter(theta)
        self.n = n
        self.beta = beta

    def DEP(theta, K, H, L, n):
        return (torch.norm(torch.matmul(theta, torch.matmul(K, torch.matmul(H, L)))))/(n*n)

    def const(w):
        if w >= 0:
            return 10e-17
        else:
            return 1

    def objective_func(epsilon, theta, K, H, L_Y, L_S, n, beta):
        w = torch.tensor(epsilon - DEP(theta, K, H, L_S, n))
        obj = -DEP(theta, K, H, L_Y, n) - const(w) * \
            torch.log(torch.sigmoid(beta*w))
        return obj

    def forward(self, x):
        x = objective_func(self.epsilon, self.theta, x, self.H,
                           self.L_Y, self.L_S, self.n, self.beta)
        return x


# def training_loop(Z, Y, S, K_Z, K_Y, K_S, H, L_Y, L_S, epsilon, p=0, epochs=100, sigma=1):
def training_loop(model, optimizer, K_Z, epochs=100):
    # n = Z.shape[0]
    # q = Z.shape[1]
    # m = L_Y.shape[0]
    losses = []
    for i in range(epochs):
        loss = model(K_Z)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        losses.append(loss)

    return losses
    # Initialize variables


def GaussianKernel(x, s, sigma):
    n_x = x.shape[0]
    n_s = s.shape[0]

    x_norm = torch.pow(torch.norm(x, dim=1).reshape([1, n_x]), 2)
    s_norm = torch.pow(torch.norm(s, dim=1).reshape([1, n_s]), 2)

    ones_x = torch.ones([1, n_x]).to(device=x.device)
    ones_s = torch.ones([1, n_s]).to(device=x.device)

    kernel = torch.exp(
        (-torch.mm(torch.t(x_norm), ones_s) -
         torch.mm(torch.t(ones_x), s_norm) + 2 * torch.mm(x, torch.t(s)))
        / (2 * sigma ** 2))

    return kernel


Z = torch.tensor(dataloader.x)
Y = torch.tensor(dataloader.y_onehot)
# y_int = dataloader.y_int
S = torch.tensor(dataloader.s)
sigma = 1.0
n = Z.size[0]
q = Z.size[1]
p = q-1
K_Z = GaussianKernel(Z, Z, sigma)
K_Y = GaussianKernel(Y, Y, sigma)
K_S = GaussianKernel(S, S, sigma)

L_Y = torch.linalg.cholesky(K_Y)
L_S = torch.linalg.cholesky(K_S)
H = torch.eye(n) - torch.ones(n)/n
beta = 5000
epsilon = 0.1
theta = torch.rand(p, q)

model = GDOPT(H, L_Y, L_S, epsilon, p, q, n, beta, theta)
opt = torch.optim.Adam(model.parameters(), lr=0.001)

losses = training_loop(model, opt, K_Z, epochs=100)
print(theta)
