from torch import nn
import torch
import torch.nn.functional as F

class PULoss(nn.Module):
    """wrapper of loss function for PU learning"""

    def __init__(self, prior, loss=lambda x: torch.nn.functional.softplus(-x), gamma=1, beta=0, nnPU=False):
        super(PULoss,self).__init__()

        if not 0 < prior < 1:
            raise NotImplementedError("The class prior should be in (0, 1)")

        self.prior = prior
        self.gamma = gamma
        self.beta = beta
        self.loss_func = loss # lambda x: (torch.tensor(1., device=x.device) - torch.sign(x)) / torch.tensor(2, device=x.device)
        self.nnPU = nnPU
        self.positive = 1
        self.unlabeled = -1
        self.min_count = 1
    
    def forward(self, inp, target, test=False):
  
        assert(inp.shape == target.shape)        

        if inp.is_cuda:
            self.prior = torch.tensor(self.prior, device=inp.device)
            #self.prior = self.prior.cuda()

        positive, unlabeled = target == self.positive, target == self.unlabeled
        positive, unlabeled = positive.type(torch.float), unlabeled.type(torch.float)

        n_positive, n_unlabeled = torch.clamp(torch.sum(positive), min=self.min_count), torch.clamp(torch.sum(unlabeled), min=self.min_count)

        y_positive = self.loss_func(inp) * positive
        y_positive_inv = self.loss_func(-inp) * positive
        y_unlabeled = self.loss_func(-inp) * unlabeled

        positive_risk = self.prior * torch.sum(y_positive) / n_positive
        negative_risk = - self.prior * torch.sum(y_positive_inv) / n_positive + torch.sum(y_unlabeled) / n_unlabeled

        if negative_risk < -self.beta and self.nnPU:
            return -self.gamma * negative_risk

        return positive_risk + negative_risk
    
    
class nnPULoss(nn.Module):
    def __init__(self, prior, beta=0, gamma=1, nnPU=True):
       
        super().__init__()
        self.prior = prior
        self.beta = beta
        self.gamma = gamma
        self.nnPU = nnPU

    def forward(self, outputs, labels):
        
        positive = labels == 1
        unlabeled = labels == 0

        n_p = positive.sum().item()
        n_u = unlabeled.sum().item()

        if n_p == 0 or n_u == 0:
            return torch.tensor(0.0, device=outputs.device, requires_grad=True)

        # positive risk
        y_p = outputs[positive]
        p_loss = F.binary_cross_entropy_with_logits(
            y_p, torch.ones_like(y_p), reduction='mean'
        )
        risk_p = self.prior * p_loss

        # negative risk estimate
        y_u = outputs[unlabeled]
        u_loss = F.binary_cross_entropy_with_logits(
            y_u, torch.zeros_like(y_u), reduction='mean'
        )

        pn_loss = F.binary_cross_entropy_with_logits(
            y_p, torch.zeros_like(y_p), reduction='mean'
        )

        risk_n = u_loss - self.prior * pn_loss

        if self.nnPU:
            # clip risk_n to at least -beta
            if risk_n < -self.beta:
                total_risk = risk_p - self.gamma * risk_n
            else:
                total_risk = risk_p + risk_n
        else:
            total_risk = risk_p + risk_n

        return total_risk
