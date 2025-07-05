import torch.nn as nn
import torch.nn.functional as F
import torch

class Pt_encoder(nn.Module):
    def __init__(self, args, output_channels=10):
        super(Pt_encoder, self).__init__()
        self.args = args
        
        self.linear1 = nn.Linear(512, 512, bias=False)
        self.bn6 = nn.BatchNorm1d(512)
        self.dp1 = nn.Dropout(p=args.dropout)
        self.linear2 = nn.Linear(512, 256)
        self.bn7 = nn.BatchNorm1d(256)
        # self.linear2 = nn.Linear(512, 256)
        

    def forward(self, x):
        x = F.relu(self.bn6(self.linear1(x)))
        x = self.dp1(x)
        x = F.relu(self.bn7(self.linear2(x)))
        return x
    

class TextNet(nn.Module):
    def __init__(self, input_dim, output_dim):
        """
        :param input_dim: dimension of tags
        :param output_dim: dimensionality of the final representation
        """
        super(TextNet, self).__init__()
        self.module_name = "text_model"

        # full-conv layers
        mid_num = 4096
        self.fc1 = nn.Linear(input_dim, mid_num)
        self.fc2 = nn.Linear(mid_num, mid_num)
        self.fc3 = nn.Linear(mid_num, output_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        # TODO
        # 与基本baseline的函数调用的细微差别
        # norm = torch.norm(x, dim=1, keepdim=True)
        # x = x / norm
        return x