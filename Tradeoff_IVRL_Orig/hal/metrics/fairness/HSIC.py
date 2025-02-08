# hsic.py

import torch
from hal.models import kernels
import torchmetrics.metric as tm
from typing import Any, Callable, Optional
import pdb

__all__ = ['DepHSIC']

class DepHSIC(tm.Metric):
    def __init__(self,
        alpha_x = kernels.Linear,
        alpha_y = kernels.GaussianKernel,
        compute_on_step: bool = False,
        dist_sync_on_step: bool = False,
        process_group: Optional[Any] = None,
        dist_sync_fn: Callable = None,
        sigma_y = 1):
        super().__init__(
            compute_on_step=compute_on_step,
            dist_sync_on_step=dist_sync_on_step,
            process_group=process_group,
            dist_sync_fn=dist_sync_fn,
        )
        self.alpha_x = alpha_x()
        self.alpha_y = alpha_y(sigma_y)
        self.add_state("xx", default=[], dist_reduce_fx=None)
        self.add_state("yy", default=[], dist_reduce_fx=None)

    def update(self, x, y):
        self.xx.append(x)
        self.yy.append(y)

    def normalize(self, x):
        x_mean = torch.mean(x, dim=0)
        x_std = torch.std(x, dim=0)
        xx = (x - x_mean) / (x_std + 1e-16)
        return xx

    def compute(self):
        # pdb.set_trace()
        x = torch.cat(self.xx, dim=0)
        y = torch.cat(self.yy, dim=0)
        x = self.normalize(x)
        y = self.normalize(y)
        kernel_x = self.alpha_x(x,x)
        kernel_y = self.alpha_y(y,y)
        # kernel_x = self.alpha_x.__call__(kernel_x,x)
        # # kernel_x = kernel_x.float()
        # kernel_y = self.alpha_y.__call__(kernel_y,y)
        # kernel_y = kernel_x.float()
        n = kernel_x.shape[0]
        # H = torch.eye(n, device = y.device) - torch.ones(n, device = y.device) / n

        # kernel_xm = torch.mm(H, torch.mm(kernel_x, H))
        # pdb.set_trace()
        # H_temp = H.double()
        # pdb.set_trace()

        # kernel_xm = torch.mm(H, (torch.mm(kernel_x.t(), H)).t())

        kernel_xm = kernel_x - torch.sum(kernel_x, 0)/n

        # kernel_ym = torch.mm(H, torch.mm(kernel_y, H))
        # kernel_ym = torch.mm(H, (torch.mm(kernel_y.t(), H)).t())
        kernel_ym = kernel_y - torch.sum(kernel_y, 0)/n

        if self.alpha_x == kernels.Linear:
            num = torch.norm(torch.mm(kernel_ym.t(), kernel_xm))
        else:
            num = torch.trace(torch.mm(kernel_ym.t(), kernel_xm))

        den = torch.sqrt(torch.trace(torch.mm(kernel_ym.t(), kernel_ym)) * torch.trace(torch.mm(kernel_xm.t(), kernel_xm)))
        hsic = num / den

        return hsic