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
    
class PULossWrapped(nn.Module):
    def __init__(self, prior, loss=lambda x: torch.nn.functional.softplus(-x), gamma=1, beta=0, nnPU=False):
        super().__init__()
        self.puloss = PULoss(prior=prior, loss=loss, gamma=gamma, beta=beta, nnPU=nnPU)
    
    def forward(self, inp, targets):
        targets_pu = targets.clone()
        targets_pu[targets_pu == 0] = -1 
        inp = inp[:, 1]
        
        return self.puloss(inp, targets_pu)    
        

class PURankingLoss(nn.Module):
    def __init__(self, margin=1.0):
        super().__init__()
        self.margin = margin
        
    def forward(self, scores, target):
        """
        scores: (batch_size, num_classes) or (batch_size,)\\
        target: tensor of shape (batch_size,) with 1 for positive, -1 for unlabeled
        """
        
        # Take positive class logits
        if scores.ndim == 2 and scores.size(1) > 1:
            scores = scores[:, 1]
        
        pos_mask = target == 1
        unl_mask = target == 0
        
        pos_scores = scores[pos_mask]
        unl_scores = scores[unl_mask]
        
        if pos_scores.numel() == 0 or unl_mask.numel() == 1:
            return torch.tensor(0.0, device=scores.device, requires_grad=True)
        
        # diff[i, j] = Pi - Uj
        diff = pos_scores.unsqueeze(1) - unl_scores.unsqueeze(0)
        
        loss = torch.relu(self.margin - diff).mean()
        
        return loss

