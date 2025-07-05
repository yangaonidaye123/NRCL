# from models.MVCNN import MVCNN
from __future__ import division, absolute_import
from models.fused_layer import Fusedformer
# from models.resnet import resnet18
from tools.utils import calculate_accuracy
import numpy as np
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision.models as models
import argparse
import torch.optim as optim
import time
import torch.utils.checkpoint as cp
# from .Model import Model


class Img_encoder(nn.Module):

    def __init__(self, pre_trained = None):
        super(Img_encoder, self).__init__()

        if pre_trained:
            self.img_net = torch.load(pre_trained)
        else:
            print('---------Loading ImageNet pretrained weights --------- ')
            resnet18 = models.resnet18(pretrained=True)
            resnet18 = list(resnet18.children())[:-1]
            self.img_net = nn.Sequential(*resnet18)
            self.linear1 = nn.Linear(512, 256, bias=False)
            self.bn6 = nn.BatchNorm1d(256)
            

    def forward(self, img, img_v):
        
        # img_feat = cp.checkpoint(self.img_net, img)
        # img_feat_v = cp.checkpoint(self.img_net, img_v)
        img_feat = self.img_net(img)
        img_feat_v = self.img_net(img_v)
        img_feat = img_feat.squeeze(3)
        img_feat = img_feat.squeeze(2)
        img_feat_v = img_feat_v.squeeze(3)
        img_feat_v = img_feat_v.squeeze(2)

        img_feat = F.relu(self.bn6(self.linear1(img_feat)))
        img_feat_v = F.relu(self.bn6(self.linear1(img_feat_v)))
        
        final_feat = 0.5*(img_feat + img_feat_v)

        return final_feat
    

class HeadNet_dual_fused(nn.Module):

    def __init__(self, num_classes, num_modals=2):
        super(HeadNet_dual_fused, self).__init__()
        self.num_classes=num_classes
        self.num_modals = num_modals
        self.u_head = nn.Sequential(*[nn.Linear(256, 128), nn.ReLU(), nn.Linear(128, self.num_classes)])
        self.m_head = nn.Sequential(*[nn.Linear(256*num_modals, 128*num_modals), nn.ReLU(), nn.Linear(128*num_modals, self.num_classes)])
        
        

    def forward(self, img_feat, pt_feat):

        img_pred = self.u_head(img_feat)
        pt_pred = self.u_head(pt_feat)
        joint_pred = self.m_head(torch.cat((img_feat, pt_feat), dim=-1))
        return img_pred,pt_pred,joint_pred    


class ImageNet(nn.Module):
    def __init__(self, input_dim, output_dim):
        """
        :param input_dim: dimension of tags
        :param output_dim: dimensionality of the final representation
        """
        super(ImageNet, self).__init__()
        self.module_name = "image_model"

        # full-conv layers
        mid_num = 4096
        self.fc1 = nn.Linear(input_dim, mid_num)
        self.fc2 = nn.Linear(mid_num, mid_num)
        # self.fc2_2 = nn.Linear(mid_num, mid_num)
        self.fc3 = nn.Linear(mid_num, output_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        # x = F.relu(self.fc2_2(x))
        x = self.fc3(x)
        # TODO
        # norm = torch.norm(x, dim=1, keepdim=True)
        # x = x / norm
        return x
    
class HeadNet_imgtext_fused(nn.Module):

    def __init__(self, args,num_classes, num_modals=2):
        super(HeadNet_imgtext_fused, self).__init__()
        self.num_classes=num_classes
        self.num_modals = num_modals
        mid_num = 512
        self.u_head = nn.Sequential(*[nn.Linear(args.output_dim, mid_num), nn.ReLU(), nn.Linear(mid_num, mid_num), nn.ReLU(), nn.Linear(mid_num,args.output_dim),nn.ReLU(),nn.Linear(args.output_dim,self.num_classes)])
        self.m_head = nn.Sequential(*[nn.Linear(args.output_dim*num_modals, mid_num*num_modals), nn.ReLU(), nn.Linear(mid_num*num_modals, mid_num*num_modals),nn.ReLU(),nn.Linear(mid_num*num_modals, args.output_dim*num_modals),nn.ReLU(),nn.Linear(args.output_dim*num_modals, self.num_classes)])
        
        

    def forward(self, img_feat, pt_feat):

        img_pred = self.u_head(img_feat)
        pt_pred = self.u_head(pt_feat)
        joint_pred = self.m_head(torch.cat((img_feat, pt_feat), dim=-1))
        return img_pred,pt_pred,joint_pred