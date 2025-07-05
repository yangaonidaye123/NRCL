import torch
import torch.nn as nn
import torch.nn.functional as F

class YangaoLoss(nn.Module):
    def __init__(self, gamma=1.0, weight=None, operation='mean'):
        super(YangaoLoss, self).__init__()
        self.gamma = gamma
        self.weight = weight
        self.operation = operation

    def forward(self, input, target):
  
        log_probs = F.log_softmax(input, dim=1)  # 计算 log-softmax 概率
        probs = torch.exp(log_probs)  # 计算 softmax 概率
        focal_factor = (1 - probs) ** self.gamma  # (1 - p)^gamma

        # 获取真实类别对应的 log 概率值
        log_probs = log_probs.gather(1, target.unsqueeze(1))  # 获取真实类别的 log 概率
        focal_factor = focal_factor.gather(1, target.unsqueeze(1))  # 获取对应真实类别的焦点因子

        loss = -focal_factor * log_probs  # 计算 Focal Loss

        if self.weight is not None:
            loss = loss * self.weight[target]  # 应用类别权重

        if self.operation == 'mean':
            return loss.mean()
        elif self.operation == 'sum':
            return loss.sum()
        else:
            return loss