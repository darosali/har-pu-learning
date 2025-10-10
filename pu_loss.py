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
        if (inp.shape[1] > 1):
            inp = inp[:, 1]
        else:
            inp = inp[:, 0]
        
        return self.puloss(inp, targets_pu)    
        

class PUAsymLoss_Direct(nn.Module):
    """ L = -E_pos[log f1] - E_unl[ log(f0 + c) ],  c = gamma/(1-gamma) """
    def __init__(self, gamma=0.5):
        super().__init__()
        self.gamma = gamma

    def forward(self, logits, y_true):
    
        logp = F.log_softmax(logits, dim=1)
        log_f0, log_f1 = logp[:,0], logp[:,1]

        pos_mask = (y_true == 1)
        unl_mask = (y_true == 0)

        c = self.gamma / (1.0 - self.gamma)
        log_c = torch.log(torch.as_tensor(c, dtype=logits.dtype, device=logits.device))

        loss_pos = -log_f1[pos_mask].mean() if pos_mask.any() else logits.new_tensor(0.)
        # log(f0 + c) = logaddexp(log f0, log c)
        loss_unl = -torch.logaddexp(log_f0[unl_mask], log_c).mean() if unl_mask.any() else logits.new_tensor(0.)

        return loss_pos + loss_unl

class PUAsymLoss_CE(nn.Module):
    """
    L = -E_pos[log f1] - E_unl[log f0] - E_unl[ log(1 + c/f0) ]
      = CE part  + penalty,
    log(1 + c/f0) = softplus(log c - log f0)
    """
    def __init__(self, gamma=0.5):
        super().__init__()
        self.gamma = gamma
        self.ce = nn.CrossEntropyLoss(reduction='none')

    def forward(self, logits, y_true):
        logp = F.log_softmax(logits, dim=1)
        log_f0, log_f1 = logp[:,0], logp[:,1]

        pos_mask = (y_true == 1)
        unl_mask = (y_true == 0)

        # CE on logits
        ce_vec = self.ce(logits, y_true) 
        loss_pos = ce_vec[pos_mask].mean() if pos_mask.any() else logits.new_tensor(0.)
        loss_unl = ce_vec[unl_mask].mean() if unl_mask.any() else logits.new_tensor(0.)

        c = self.gamma / (1.0 - self.gamma)
        log_c = torch.log(torch.as_tensor(c, dtype=logits.dtype, device=logits.device))
        # penalty = -E_unl log(1 + c/f0) = -E_unl softplus(log c - log f0)
        penalty = -F.softplus(log_c - log_f0[unl_mask]).mean() if unl_mask.any() else logits.new_tensor(0.)

        return loss_pos + loss_unl + penalty

    


