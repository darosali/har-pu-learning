import torch
import torch.nn as nn
import torch.nn.functional as F

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
