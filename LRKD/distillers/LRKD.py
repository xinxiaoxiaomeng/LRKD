import torch.nn as nn
import torch
import torch.nn.functional as F

import torchvision

class LRKDLoss(nn.Module):
    def __init__(self, T):
        super(LRKDLoss, self).__init__()
        self.T = T


    def js_div(self, q, p):
        q = F.softmax(q, dim=1)
        p = F.softmax(p, dim=1)

        log_mean = ((q + p) / 2).log()
        loss = (F.kl_div(log_mean, q, reduction='sum') + F.kl_div(log_mean, p, reduction='sum')) / 2
        return loss

    def kl(self, y_s, y_t):
        p_s = F.log_softmax(y_s / self.T, dim=1)
        p_t = F.softmax(y_t / self.T, dim=1)
        # 计算KL散度，p_t指导p_s，后者指导前者的训练，p_t真实分布，p_s是预测分布取log值
        loss = F.kl_div(p_s, p_t, reduction='sum') * (self.T ** 2) / y_s.shape[0]
        return loss

    def forward(self,logits_s, logits_t, targets):
        batch = len(logits_s)
        num_c = logits_s.size(1)
        targets = targets.unsqueeze(1)
        mask = torch.ones(batch, num_c).cuda()
        mask = torch.scatter(input=mask, dim=1, index=targets, value=0)
        mask = (mask != 0)
        logits_s_t = logits_s.gather(1, targets)
        relation_logits_s = (logits_s_t - logits_s)
        relation_logits_s = torch.masked_select(relation_logits_s, mask=mask).view(batch, num_c-1)
        logits_t_t = logits_t.gather(1, targets)
        relation_logits_t = (logits_t_t - logits_t)
        relation_logits_t = torch.masked_select(relation_logits_t, mask=mask).view(batch, num_c-1)
        loss_sim = self.kl(relation_logits_s, relation_logits_t)


        logits_s = F.normalize(logits_s, dim=1)
        logits_t = F.normalize(logits_t, dim=1)
        center_s = logits_s.mean(0)
        center_t = logits_t.mean(0)
        r_s = logits_s - center_s.detach()
        r_t = logits_t - center_t.detach()
        loss_c = (r_s - r_t).pow(2).sum() / batch

        loss = 10 * loss_sim + 10 * loss_c
        return loss