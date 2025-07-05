import torch
import torch.nn as nn
import math
import torch.nn.functional as F


    
class YA_loss(nn.Module):
    
    def __init__(self, num_classes = 10, feat_dim=256, warmup=15, temperature=0.07):
        super(YA_loss, self).__init__()
        self.num_classes = num_classes
        self.feat_dim = feat_dim
        self.warmup = warmup
        
        self.temperature = temperature
        self.alpha=0.5
        self.a=0.2
        
        # center层面都可以有很好的策略
        # 正交初始化
        self.centers = nn.Parameter(torch.randn(self.num_classes, self.feat_dim).cuda())

    def forward(self, features, labels, w=None, weight=None):
        
        # print(x.shape, labels.shape)
        # batch_size = features.size(0)
        
        features = F.normalize(features, p=2, dim=1)

        centers = self.centers
        centers = F.normalize(centers, p=2, dim=1)
        
        
        # mask : one_hot

    
        mask = torch.nn.functional.one_hot(labels,num_classes = self.num_classes).float().cuda()
        
        # 表示余弦相似度
        # batch * num_classes
        anchor_dot_contrast = torch.div(
            torch.matmul(features, centers.T),
            self.temperature)
     
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()
        
   

        # compute log_prob
        exp_logits = torch.exp(logits)
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))
  
 

        mask_sum = mask.sum(1)



        mean_log_prob_pos = (mask * log_prob).sum(1) / mask_sum
  
        p = torch.exp(mean_log_prob_pos)

        # focal
        ya_= (1.0 - p)


        loss = - mean_log_prob_pos * ya_
        if weight is not None:
            loss = loss*weight
        loss = loss.mean()

        return loss, self.centers
    
