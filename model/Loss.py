import torch
import torch.nn.functional as F
import torch.nn as nn
import pdb


class FocalLoss(nn.Module):
    def __init__(self, alpha=1.0, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.reduction = reduction

        if isinstance(alpha, list):
            self.alpha = torch.tensor(alpha)
        else:
            self.alpha = alpha  # scalar or tensor
    def forward(self, inputs, targets):
        """
        inputs: logits, shape [B, C]
        targets: int64 class index, shape [B]
        """
        log_probs = F.log_softmax(inputs, dim=-1)  # [B, C]
        probs = torch.exp(log_probs)               # softmax probs

        targets = targets.view(-1, 1)              # [B, 1]
        log_pt = log_probs.gather(1, targets).squeeze(1)  # [B]
        pt = probs.gather(1, targets).squeeze(1)          # [B]

        if isinstance(self.alpha, torch.Tensor):
            if self.alpha.device != inputs.device:
                self.alpha = self.alpha.to(inputs.device)
            at = self.alpha[targets.squeeze(1)]  # [B]
        else:
            at = self.alpha

        # focal loss
        focal_term = (1 - pt) ** self.gamma
        loss = -at * focal_term * log_pt  # [B]

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss  # [B]

class FinalLoss(nn.Module):
    def __init__(self, temperature=0.2, lambda_proto=0.5, neg_lambda=0.5):
        super(FinalLoss, self).__init__()
        self.temperature = temperature
        self.lambda_proto = lambda_proto
        self.FocalLoss=FocalLoss()
        self.neg_lambda=neg_lambda

    def supcon_loss(self, features, labels):
        """ 
        Supervisory comparison loss
        """
        bs, feature_dim = features.shape  
        features = F.normalize(features, dim=-1) 
        sim_matrix = torch.matmul(features, features.T)  
        mask = torch.eye(bs, device=features.device).bool()
        sim_matrix = sim_matrix.masked_fill(mask, 1e-9)
        labels = labels.view(bs, 1)  
        labels_expanded = labels.expand(bs, bs)  
        positive_mask = (labels_expanded == labels_expanded.T).float()  
        positive_mask = positive_mask * (~mask)
        neg_mask = (labels != labels.T).float()
        exp_sim_matrix = torch.exp(sim_matrix / self.temperature) 
        sum_exp_sim = exp_sim_matrix.sum(dim=1, keepdim=True)  
        pos_sim = (exp_sim_matrix * positive_mask).sum(dim=1)  
        supcon_loss = -torch.log(pos_sim / sum_exp_sim + 1e-9).mean()  
        # Negative pushing loss
        neg_sim = (sim_matrix * neg_mask).sum(dim=1) / (neg_mask.sum(dim=1) + 1e-9)
        neg_loss = neg_sim.mean()
        neg_loss = torch.clamp(neg_loss, min=0.0)

        total_loss = supcon_loss + self.neg_lambda * neg_loss

        return total_loss

    def prototype_loss(self, features, labels, prototypes,weight=2.95):
        bs = features.size(0)
        features = F.normalize(features, dim=-1) 
        proto_0=prototypes["0"]
        proto_1=prototypes["1"]
        proto_0 = proto_0.unsqueeze(1)
        proto_1 = proto_1.unsqueeze(1)
        sim_with_proto_0 = torch.mm(features, proto_0) 
        sim_with_proto_1 = torch.mm(features, proto_1) 
        sim_with_proto_0 = (sim_with_proto_0 + 1) / 2  
        sim_with_proto_1 = (sim_with_proto_1 + 1) / 2  
        loss = (
            (1 - sim_with_proto_0).T * (labels == 0).float() + 
            (1 - sim_with_proto_1).T * (labels == 1).float()*weight +
            sim_with_proto_1.T * (labels == 0).float()+ 
            sim_with_proto_0.T * (labels == 1).float()*weight 
        )
        return loss.mean()
    
    def proto_infonce_loss(self,features,labels,prototypes):
        bs = features.size(0)  
        features = F.normalize(features, dim=-1)  
        proto_0=prototypes["0"]
        proto_1=prototypes["1"]
        proto_0 = proto_0.unsqueeze(1)
        proto_1 = proto_1.unsqueeze(1)
        sim_with_proto_0 = torch.mm(features, proto_0)  
        sim_with_proto_1 = torch.mm(features, proto_1)  
        exp_metric_0=torch.exp(sim_with_proto_0/self.temperature)
        exp_metric_1=torch.exp(sim_with_proto_1/self.temperature)
        posmask_0=(labels == 0).float()
        posmask_1=(labels == 1).float()
        fenshi=((exp_metric_0.T * posmask_0)+(exp_metric_1.T * posmask_1))/(exp_metric_0.T+exp_metric_1.T)
        l=-torch.log(fenshi).mean()
        return l
    
    def forward(self, features, labels, prototypes,batch_q, logits, logit_labels):
        supcon_loss = self.supcon_loss(features, labels)
        prototype_loss = self.prototype_loss(batch_q, logit_labels, prototypes)
        focal_loss = self.FocalLoss(logits, logit_labels)
        
        return supcon_loss + self.lambda_proto * prototype_loss+focal_loss,supcon_loss,prototype_loss,focal_loss
