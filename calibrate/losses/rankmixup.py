import torch
import torch.nn as nn
import torch.nn.functional as F

class RankMixup_MRL(nn.Module):
    def __init__(self, num_classes: int = 10,
                       margin: float = 0.1,
                       alpha: float = 0.1,
                       ignore_index: int =-100):
        super().__init__()
        self.margin = margin
        self.alpha = alpha
        self.ignore_index = ignore_index
        self.cross_entropy = nn.CrossEntropyLoss()

    @property
    def names(self):
        return "loss", "loss_ce", "loss_mixup"

    def get_logit_diff(self, inputs, mixup):
        max_values, indices = inputs.max(dim=1)
        max_values = max_values.unsqueeze(dim=1)
       
        max_values_mixup, indices_mixup = mixup.max(dim=1)
        max_values_mixup = max_values_mixup.unsqueeze(dim=1)
        # diff = max_values - max_values_mixup
        diff = max_values_mixup -  max_values 

        return diff
    
    def get_conf_diff(self, inputs, mixup):
        inputs = F.softmax(inputs, dim=1)
        max_values, indices = inputs.max(dim=1)
        max_values = max_values.unsqueeze(dim=1)

        mixup = F.softmax(mixup, dim=1)
        max_values_mixup, indices_mixup = mixup.max(dim=1)
        max_values_mixup = max_values_mixup.unsqueeze(dim=1)
        
        # diff = max_values - max_values_mixup
        diff = max_values_mixup -  max_values 

        return diff

    def forward(self, inputs, targets, mixup, target_re, lam):
        if inputs.dim() > 2:
            inputs = inputs.view(inputs.size(0), inputs.size(1), -1)  # N,C,H,W => N,C,H*W
            inputs = inputs.transpose(1, 2)    # N,C,H*W => N,H*W,C
            inputs = inputs.contiguous().view(-1, inputs.size(2))   # N,H*W,C => N*H*W,C
            targets = targets.view(-1)

        if self.ignore_index >= 0:
            index = torch.nonzero(targets != self.ignore_index).squeeze()
            inputs = inputs[index, :]
            targets = targets[index]

        loss_ce = self.cross_entropy(inputs, targets)
        
        self_mixup_mask = (target_re == 1.0).sum(dim=1).reshape(1, -1) 
        self_mixup_mask = (self_mixup_mask.sum(dim=0) == 0.0) 
     
        # diff = self.get_conf_diff(inputs, mixup) # using probability
        diff = self.get_logit_diff(inputs, mixup)
        loss_mixup = (self_mixup_mask * F.relu(diff+self.margin)).mean()

        loss = loss_ce + self.alpha * loss_mixup

        return loss, loss_ce, loss_mixup



class RankMixup_MNDCG(nn.Module):
    def __init__(self, num_classes: int = 10,
                       alpha: float = 0.1,
                       ignore_index: int =-100):
        super().__init__()
        self.alpha = alpha
        self.ignore_index = ignore_index
        self.num_classes = num_classes
        self.cross_entropy = nn.CrossEntropyLoss()

    @property
    def names(self):
        return "loss", "loss_ce", "loss_mixup"

    def get_indcg(self, inputs, mixup, lam, target_re):
        mixup = mixup.reshape(len(lam), -1, self.num_classes) # mixup num x batch x num class
        target_re = target_re.reshape(len(lam), -1, self.num_classes) # mixup num x batch x num class
       
        mixup = F.softmax(mixup, dim=2)
        inputs = F.softmax(inputs, dim=1)

        inputs_lam = torch.ones(inputs.size(0), 1, device=inputs.device)
        max_values = inputs.max(dim=1, keepdim=True)[0]
        max_mixup = mixup.max(dim=2)[0].t() #  batch  x mixup num 
        max_lam = target_re.max(dim=2)[0].t() #  batch  x mixup num 
        # compute dcg         
        sort_index = torch.argsort(max_lam, descending=True)
        max_mixup_sorted = torch.gather(max_mixup, 1, sort_index)
        order = torch.arange(1, 2+len(lam), device = max_mixup.device)
        dcg_order = torch.log2(order + 1)
        max_mixup_sorted = torch.cat((max_values, max_mixup_sorted), dim=1)
        dcg = (max_mixup_sorted / dcg_order).sum(dim=1)
      
        max_lam_sorted = torch.gather(max_lam, 1, sort_index)
        max_lam_sorted = torch.cat((inputs_lam, max_lam_sorted), dim=1)
        idcg = (max_lam_sorted / dcg_order).sum(dim=1)

        #compute ndcg
        ndcg = dcg / idcg
        inv_ndcg = idcg / dcg
        ndcg_mask = (idcg > dcg)
        ndcg = ndcg_mask * ndcg + (~ndcg_mask) * inv_ndcg   

        return ndcg

    def forward(self, inputs, targets, mixup, target_re, lam):
        if inputs.dim() > 2:
            inputs = inputs.view(inputs.size(0), inputs.size(1), -1)  # N,C,H,W => N,C,H*W
            inputs = inputs.transpose(1, 2)    # N,C,H*W => N,H*W,C
            inputs = inputs.contiguous().view(-1, inputs.size(2))   # N,H*W,C => N*H*W,C
            targets = targets.view(-1)

        if self.ignore_index >= 0:
            index = torch.nonzero(targets != self.ignore_index).squeeze()
            inputs = inputs[index, :]
            targets = targets[index]

        loss_ce = self.cross_entropy(inputs, targets)
        #NDCG loss
        loss_mixup = (1.0 - self.get_indcg(inputs, mixup, lam, target_re)).mean()
        loss = loss_ce + self.alpha * loss_mixup 

        return loss, loss_ce, loss_mixup

