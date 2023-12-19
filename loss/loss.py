import torch
import torch.nn as nn
import torch.nn.functional as F

class RelaHashLoss(nn.Module):
    def __init__(self,
                 beta=8,
                 m=0.5,
                 multiclass=True,
                 onehot=True,
                 **kwargs):
        super(RelaHashLoss, self).__init__()
        self.beta = beta
        self.m = m
        self.multiclass = multiclass
        self.onehot = onehot

    def compute_margin_logits(self, logits, labels):
        if self.multiclass:
            y_onehot = labels * self.m
            margin_logits = self.beta * (logits - y_onehot)
        else:
            y_onehot = torch.zeros_like(logits)
            y_onehot.scatter_(1, torch.unsqueeze(labels, dim=-1), self.m)
            margin_logits = self.beta * (logits - y_onehot)
        return margin_logits

    def forward(self, logits, labels):
        if self.multiclass:
            if not self.onehot:
                labels = F.one_hot(labels, logits.size(1))
            labels = labels.float()

            margin_logits = self.compute_margin_logits(logits, labels)
            # label smoothing
            log_logits = F.log_softmax(margin_logits, dim=1)
            # if there are some Zero Vector in Matrix
            A = ((labels == 0).sum(dim=1) == labels.shape[1])
            labels[A == True] = 1
            labels_scaled = labels / labels.sum(dim=1, keepdim=True)
            loss = - (labels_scaled * log_logits).sum(dim=1)
            loss = loss.mean()
        # else:
        #     if self.onehot:
        #         labels = labels.argmax(1)
        #
        #     margin_logits = self.compute_margin_logits(logits, labels)
        #     loss = F.cross_entropy(margin_logits, labels)
        return loss
    
def smooth_CE(logits, labels, peak):
    # logits - [batch, num_cls]
    # label - [batch]
    A = ((labels == 0).sum(dim=1) == labels.shape[1])
    labels[A == True] = 1
    batch, num_cls = logits.size()
    label_logits = labels
    # label_logits = F.one_hot(label, num_cls)
    smooth_label = torch.ones(logits.size()) * (1 - peak) / (num_cls - 1)
    smooth_label[label_logits == 1] = peak
    
    logits = F.log_softmax(logits, -1)
    ce = torch.mul(logits, smooth_label.to(logits.device))
    loss = torch.mean(-torch.sum(ce, -1))  # batch average

    return loss