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
# from torch.autograd import Variable

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
# from hal.metrics.fairness.HSIC import DepHSIC
from hal.metrics.fairness.dep_kcc_celeba import DEPKCC_CelebA

# args = config.parse_args()
import torch.optim.lr_scheduler as lr_scheduler

# dataloader = getattr(datasets, args.dataset)(args)

writer = SummaryWriter()


class GDOPT(torch.nn.Module):
    def __init__(self, opts,  y, s, epsilon, beta, theta):
        super(GDOPT, self).__init__()
        # self.K_Z = K_Z
        self.opts = opts
        self.rbfsigma = getattr(models, opts.gaussian_sigma)()

        # self.H = H
        self.y_tot = y
        self.s_tot = s
        self.epsilon = epsilon
        # if theta == None: 
        #     theta = torch.rand(r, n)
        # self.H = H
        # theta.requires_grad = True
        self.theta = nn.Parameter(theta, requires_grad=True)

        # print(self.theta.grad_fn)
        # exit()
        # self.n = n
        self.beta = beta
        # if opts.fairness_type is not None:
        self.DEP_y = getattr(metrics, opts.fairness_type)(
            **opts.fairness_options)
        self.DEP_s = getattr(metrics, opts.fairness_type)(
            **opts.fairness_options)

        # self.dpv = getattr(metrics, opts.evaluation_type)(**opts.evaluation_options)
        
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
        # if w < 0:
        #     return torch.square(alpha*w)
        # else:
        #     return 0
        #     return torch.square(alpha*w)
        return torch.square(alpha*w)


    def barrier_func_sigmoid(self, gamma, w):
        return gamma*torch.sigmoid(self.beta * w)

    def barrier_func_cos(self, kappa, w):
        if w < (-np.pi/kappa):
            # print(w)
            # print(np.pi/kappa)
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

    def barrier_func_sq_tol(self, delta, gamma, w):
        if (w > delta) or (w < -delta):
            # print("1")
            return (gamma * torch.square(w))/(2*delta) + (gamma * delta)/2
        else:
            # print("2")
            return gamma*torch.abs(w)
        

    def objective_func(self, x):
        # self.theta = self.theta.double()
        # self.theta = (self.theta).double()
        # print("here++")
        # pdb.set_trace()
        x = x.double()
        # self.H = self.H.float()
        self.s = self.s.double()
        self.y = self.y.double()
        # self.theta = Variable(self.theta, requires_grad=True)
        # self.theta = nn.Parameter(self.theta)
        self.theta.requires_grad = True
        x = torch.mm(x, self.theta)
        # dep_s = self.DEP_s(x, self.s, self.opts, self.rbfsigma)
        # dep_y = self.DEP_y(x, self.y, self.opts, self.rbfsigma, label=False)
        # kcc_s, dep_s = self.DEP_s.compute()
        # kcc_y, dep_y = self.DEP_y.compute()
        # n = x.shape[0]
        # H = torch.eye(n) - torch.ones(n) / n
        # H = torch.eye(n) - torch.ones(n) / n

        # dep_s_F = DepHSIC(sigma_y = self.opts.sigma_s)
        # dep_y_F = DepHSIC(sigma_y = self.opts.sigma_y)

        dep_s_F = DEPKCC_CelebA()
        dep_y_F = DEPKCC_CelebA()
        # pdb.set_trace()

        dep_s = dep_s_F.update(x, self.s, self.opts, self.rbfsigma)
        dep_y= dep_y_F.update(x, self.y, self.opts, self.rbfsigma)

        kcc_s, dep_s = dep_s_F.compute()
        kcc_y, self.T1 = dep_y_F.compute()


        # print("HERE: ",dep_y.grad_fn)
        # exit()
        # self.DEP_s.reset()

        # self.DEP_y.reset() 

        # print("K: ", K.size())
        # print("L_Y: ", L_Y.size())
        # print("L_S: ", L_S.size())
        # print("H: ", H.size())
        # print("Theta: ", theta.size())
        # print(dep_s)

        w = torch.tensor(self.epsilon) - dep_s
        # exit()


        # self.T1 = -dep_y
        
        
        # writer.add_scalar('T1', T1, 1)

        # self.T2 = -self.barrier_func_logsig(w)

        # self.T3 = self.barrier_func_sq(30, w)

        self.T8 = self.barrier_func_sq_tol(0.001,2.5, w)


        # self.T4 = self.barrier_func_sigmoid(4, -w)

        # self.w = w

        # self.T5 = self.barrier_func_cos(10, w)

        # self.T6 = self.barrier_func_log(10e+4, w)

        # self.T7 = self.barrier_func_sqrt(1, w)

        # epsilon_tot = torch.linspace(0, 1, 100)
        # w_tot = torch.tensor(epsilon_tot - DEP(theta, K, H, L_S, n))
        # T2_tot = barrier_func_sq(1000, w_tot)
        # plt.plot(epsilon_tot, T2_tot)
        # plt.savefig("output.png")
        # exit()

        # obj = -T1-0.0001*T2
        obj = -self.T1 + self.T8
        print("####################")
        print(self.epsilon, "T1: ", self.T1)
        # print("T2: ", T2)
        print(self.epsilon, "DEP(Z,S): ", dep_s)

        print(self.epsilon, "T8: ", self.T8)
        print(self.epsilon, "OBJ: ", obj)
        print("####################")
        return obj

    def theta_ret(self):
        return self.theta

    def forward(self, x):
        x = self.objective_func(x)
        # print(x)
        return x

    def training_loop(self, model, optimizer, scheduler, K_Z_tot, epochs=100):
        # n = Z.shape[0]
        # q = Z.shape[1]
        # m = L_Y.shape[0]
        print("here 2!!!")
        # pdb.set_trace()
        losses = []
        # pdb.set_trace()
        for i in range(epochs):
            # print(K_Z)
            # pdb.set_trace()
            randint = torch.randperm(len(K_Z_tot))[:15000]
            K_Z = K_Z_tot[randint]
            self.y = self.y_tot[randint]
            self.s = self.s_tot[randint]

            loss = model(K_Z)
            # print(i)
            writer.add_scalar('Loss/train', loss, i)
            writer.add_scalar('DEP(Z,Y)', self.T1, i)
            # writer.add_scalar('log(sigmoid(beta*w))', self.T2, i)
            # writer.add_scalar('(alpha*w)^2', self.T3, i)
            # writer.add_scalar('gamma*sigmoid(beta*w)', self.T4, i)
            # writer.add_scalar('w', self.w, i)
            # writer.add_scalar('1-cos(kappa*w)', self.T5, i)
            # writer.add_scalar('log(beta*w)', self.T6, i)
            # writer.add_scalar('(beta*w)^(1/2)', self.T7, i)
            writer.add_scalar('Sqauare with Tolerance', self.T8, i)

            # loss = Variable(loss, requires_grad=True)
            # print(loss)
            # exit()
            # print("HERE: ",loss.grad_fn)
 
            optimizer.zero_grad()
            if i!= epochs-1:
                loss.backward(retain_graph=True)
            else:
                loss.backward()

            optimizer.step()
            scheduler.step()
            losses.append(loss)

        # new_theta =
        return losses


# def const(w):
#     if w >= 0:
#         return 10e-17
#     else:
#         return 0.0001


def theta_opt_main(opts, L_x, y, s, theta, epochs = 10):
    # print("here 1!!!")

    # writer.add_scalar(obj)
    # n = L_x.size()[0]
    # print("here 2!!!")

    # H = torch.eye(n) - torch.ones(n)/n
    # print("here 3!!!")

    beta = 10
    # print("here 4!!!")

    epsilon = opts.epsilon
    # print("here 5!!!")

    # theta.requires_grad = True
    theta = nn.Parameter(theta)
    # print("here 6!!!")

    # theta = torch.rand(r, n)
    # theta_local = theta_local.double()
    n = L_x.shape[0]
    # H = torch.eye(n) - torch.ones(n) / n
    # H = H.double()


    # for i in range(epochs):
    #     # L_x_temp = L_x.detach().clone()
    #     # y_temp = y.detach().clone()
    #     # s_temp = s.detach().clone()
    #     # while L_x_temp != None and len(L_x_temp)>opts.batch_size_train_opt:
    #     #     # theta = theta_opt_main(opts, L_x[0:opts.batch_size_train_opt], y_temp[0:opts.batch_size_train_opt], s_temp[0:opts.batch_size_train_opt], theta)
    #     #     model = GDOPT(opts, y_temp[0:opts.batch_size_train_opt], s_temp[0:opts.batch_size_train_opt], epsilon, beta, theta)
    #     #     opt = torch.optim.Adam(model.parameters(), lr=0.005)
    #     #     losses = model.training_loop(model, opt, L_x[0:opts.batch_size_train_opt], epochs=1)
    #     #     L_x_temp = L_x_temp[opts.batch_size_train_opt:]
    #     #     y_temp = y_temp[opts.batch_size_train_opt:]
    #     #     s_temp = s_temp[opts.batch_size_train_opt:]
    #     model = GDOPT(opts, y, s, epsilon, beta, theta)
    #     opt = torch.optim.Adam(model.parameters(), lr=0.005)
    #     scheduler = lr_scheduler.ExponentialLR(opt, gamma=0.9)

    #     losses = model.training_loop(model, opt, L_x, epochs=1)

    #     print(i)
    #     theta = model.theta_ret()

    model = GDOPT(opts, y, s, epsilon, beta, theta)

    opt = torch.optim.Adam(model.parameters(), lr=0.005)    
    scheduler = lr_scheduler.ExponentialLR(opt, gamma=0.9)
    losses = model.training_loop(model, opt, scheduler, L_x, epochs=epochs)

        
    # print(model.parameters())
    # for name, param in model.named_parameters():
    #     # if param.requires_grad:
    #     print(name, param.data)
    # exit()

    # print(losses)
    plt.figure(figsize=(14, 7))
    plt.plot(losses)
    plt.savefig('output.png')

    # exit()
    new_theta = theta

    # print(torch.sum(new_theta == theta))
    # exit()
    writer.close()
    return new_theta
