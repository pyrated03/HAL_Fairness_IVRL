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
from torch.utils.tensorboard import SummaryWriter

# args = config.parse_args()

# dataloader = getattr(datasets, args.dataset)(args)

writer = SummaryWriter()


def DEP(theta, K, H, L, n):
    return (torch.norm(torch.matmul(torch.transpose(torch.matmul(K, theta), 0, 1), torch.matmul(H, L))))/(n*n)


# def const(w):
#     if w >= 0:
#         return 10e-17
#     else:
#         return 0.0001


def barrier_func_log(beta, w):
    return torch.log(torch.sigmoid(beta*w))


def barrier_func_sq(alpha, w):
    if w < 0:
        return torch.square(alpha*w)
    else:
        return 0


def objective_func(epsilon, theta, K, H, L_Y, L_S, n, beta):
    theta = theta.double()
    K = K.double()
    H = H.double()
    L_S = L_S.double()
    L_Y = L_Y.double()
    # print("K: ", K.size())
    # print("L_Y: ", L_Y.size())
    # print("L_S: ", L_S.size())
    # print("H: ", H.size())
    # print("Theta: ", theta.size())
    w = torch.tensor(epsilon) - DEP(theta, K, H, L_S, n)
    T1 = -DEP(theta, K, H, L_Y, n)
    writer.add_scalar('T1', T1, 1)

    T2 = -barrier_func_log(beta, w)

    T3 = barrier_func_sq(250, w)

    # epsilon_tot = torch.linspace(0, 1, 100)
    # w_tot = torch.tensor(epsilon_tot - DEP(theta, K, H, L_S, n))
    # T2_tot = barrier_func_sq(1000, w_tot)
    # plt.plot(epsilon_tot, T2_tot)
    # plt.savefig("output.png")
    # exit()

    # obj = -T1-0.0001*T2
    obj = T1+T3
    print("####################")
    print("T1: ", T1)
    # print("T2: ", T2)
    print("T3: ", T3)
    print("OBJ: ", obj)
    print("####################")

    return obj


class GDOPT(torch.nn.Module):
    def __init__(self, H, L_Y, L_S, epsilon, r, n, beta=5000, theta=None):
        super(GDOPT, self).__init__()
        # self.K_Z = K_Z
        self.H = H
        self.L_Y = L_Y
        self.L_S = L_S
        self.epsilon = epsilon
        if theta == None:
            theta = torch.rand(r, n)
        theta.requires_grad = True
        self.theta = nn.Parameter(theta)
        self.n = n
        self.beta = beta

    def theta_ret(self):
        return self.theta

    def forward(self, x):
        x = objective_func(self.epsilon, self.theta, x, self.H,
                           self.L_Y, self.L_S, self.n, self.beta)
        # print(x)
        return x


def training_loop(model, optimizer, K_Z, epochs=1):
    # n = Z.shape[0]
    # q = Z.shape[1]
    # m = L_Y.shape[0]
    losses = []
    for i in range(epochs):
        loss = model(K_Z)
        writer.add_scalar('Loss/train', loss, i)

        # print(loss)
        # exit()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss)

    # new_theta =
    return losses


def theta_opt_main(K_Z, r, L_Y, L_S, theta):
    # writer.add_scalar(obj)
    n = K_Z.size()[0]
    H = torch.eye(n) - torch.ones(n)/n

    beta = 5000
    epsilon = 10e-7

    # theta = torch.rand(p, q)

    model = GDOPT(H, L_Y, L_S, epsilon, r, n, beta, theta)

    opt = torch.optim.Adam(model.parameters(), lr=0.001)

    losses = training_loop(model, opt, K_Z, epochs=100)

    # print(losses)
    plt.figure(figsize=(14, 7))
    plt.plot(losses)
    plt.savefig('output.png')

    # exit()
    new_theta = model.theta_ret()

    # print(torch.sum(new_theta == theta))
    # exit()
    writer.close()
    return new_theta
