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
        # theta.requires_grad = True
        self.theta = nn.Parameter(theta)
        self.n = n
        self.beta = beta

    # def DEP(self, K, L):
    #     return torch.pow((torch.norm(torch.matmul(torch.transpose(torch.matmul(K, self.theta), 0, 1), torch.matmul(self.H, L)))), 2)/(self.n*self.n)

    def barrier_func_log(self, beta, w):
        if w < 0:
            return torch.log(-beta*w)
        else:
            return 0

    def barrier_func_logsig(self, w):
        return torch.log(torch.sigmoid(self.beta*w))

    def barrier_func_sq(self, alpha, w):
        if w < 0:
            return torch.square(alpha*w)
        else:
            return 0

    def barrier_func_sigmoid(self, gamma, w):
        return gamma*torch.sigmoid(self.beta * w)

    def barrier_func_cos(self, kappa, w):
        if w < (-np.pi/kappa):
            print(w)
            print(np.pi/kappa)
            return 2
        elif w < 0:
            return 1 - torch.cos(kappa*w)
        else:
            return 0

    def barrier_func_sqrt(self, beta, w):
        if w < 0:
            return torch.sqrt(beta*-w)
        else:
            return 0

    def objective_func(self, K):
        # self.theta = self.theta.double()
        # self.theta = (self.theta).double()
        K = K.float()
        self.H = self.H.float()
        self.L_S = self.L_S.float()
        self.L_Y = self.L_Y.float()

        # dep_s = self.DEP(z, s, self.opts, self.rbfsigma)
        # dep_y = self.DEP(z, y, self.opts, self.rbfsigma, label=True)

        # print("K: ", K.size())
        # print("L_Y: ", L_Y.size())
        # print("L_S: ", L_S.size())
        # print("H: ", H.size())
        # print("Theta: ", theta.size())
        w = torch.tensor(self.epsilon) - self.DEP(K, self.L_S)
        self.T1 = -self.DEP(K, self.L_Y)
        # writer.add_scalar('T1', T1, 1)

        self.T2 = -self.barrier_func_logsig(w)

        self.T3 = self.barrier_func_sq(250, w)

        self.T4 = self.barrier_func_sigmoid(0.01, -w)

        self.w = w

        self.T5 = 10*self.barrier_func_cos(5*10e+3, w)

        self.T6 = self.barrier_func_log(10e+4, w)

        self.T7 = self.barrier_func_sqrt(1, w)

        # epsilon_tot = torch.linspace(0, 1, 100)
        # w_tot = torch.tensor(epsilon_tot - DEP(theta, K, H, L_S, n))
        # T2_tot = barrier_func_sq(1000, w_tot)
        # plt.plot(epsilon_tot, T2_tot)
        # plt.savefig("output.png")
        # exit()

        # obj = -T1-0.0001*T2
        obj = self.T1+self.T6
        print("####################")
        print("T1: ", self.T1)
        # print("T2: ", T2)
        print("T7: ", self.T7)
        print("OBJ: ", obj)
        print("####################")
        return obj

    def theta_ret(self):
        return self.theta

    def forward(self, x):
        x = self.objective_func(x)
        # print(x)
        return x

    def training_loop(self, model, optimizer, K_Z, epochs=100):
        # n = Z.shape[0]
        # q = Z.shape[1]
        # m = L_Y.shape[0]
        losses = []
        for i in range(epochs):
            loss = model.forward(K_Z)
            writer.add_scalar('Loss/train', loss, i)
            writer.add_scalar('DEP(Z,Y)', self.T1, i)
            writer.add_scalar('log(sigmoid(beta*w))', self.T2, i)
            writer.add_scalar('(alpha*w)^2', self.T3, i)
            writer.add_scalar('gamma*sigmoid(beta*w)', self.T4, i)
            writer.add_scalar('w', self.w, i)
            writer.add_scalar('1-cos(kappa*w)', self.T5, i)
            writer.add_scalar('log(beta*w)', self.T6, i)
            writer.add_scalar('(beta*w)^(1/2)', self.T7, i)

            # print(loss)
            # exit()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss)

        # new_theta =
        return losses


# def const(w):
#     if w >= 0:
#         return 10e-17
#     else:
#         return 0.0001


def theta_opt_main(opts, K_Z, r, L_Y, L_S, theta):
    # writer.add_scalar(obj)
    n = K_Z.size()[0]
    H = torch.eye(n) - torch.ones(n)/n

    beta = 0.1
    epsilon = 5*10e-7

    # theta = torch.rand(p, q)
    # theta_local = theta_local.double()
    model = GDOPT(opts, H, L_Y, L_S, epsilon, r, n, beta, theta)

    opt = torch.optim.Adam(model.parameters(), lr=0.005)

    losses = model.training_loop(model, opt, K_Z, epochs=100)

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
