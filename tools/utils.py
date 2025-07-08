import os
import torch
import numpy as np
import torch.nn.functional as F
from torch.autograd import Variable
from sklearn.mixture import GaussianMixture
from losses.YA_loss import YangaoLoss
import pickle
# from torch
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns
def calculate_accuracy(outputs, targets):
    batch_size = outputs.size(0)

    _, pred = outputs.topk(1, 1, True)
    pred = pred.t()

    correct = pred.eq(targets.view(1, -1))
    n_correct_elems = correct.float().sum().item()

    return 1.0 * n_correct_elems / batch_size

class EMA:
    def __init__(self, num_data, num_classes, beta=0.99, historical_length=10, warm_up=8):
        self.beta = beta
        self.num_data = num_data
        self.num_classes = num_classes
        self.historical_length = 10
        self.label_bank = torch.zeros((num_data, num_classes))
        self.epoch_img_label =  -torch.ones((num_data, historical_length), dtype=torch.long)
        self.epoch_pt_label = -torch.ones((num_data, historical_length), dtype=torch.long)
        self.ptr = 0
        self.warm_up = warm_up
        # print(self.warm_up)
        self.updated = 0
        self.set_epoch(0)

    def initial_bank(self, data_loader):
        self.label_bank = torch.ones((self.num_data, self.num_classes))/self.num_classes
    
    @torch.no_grad()
    def update(self, value, indices, epoch, weight=None):
        batch_size = value.size(0)
        row_index = torch.arange(batch_size, dtype=torch.long).unsqueeze(1)
        indices = indices.to(torch.long)
        y_pre = self.label_bank[indices, :]
        if weight is not None:
            value = (1-weight)* self.label_bank[indices, :] + weight*value
        
        # print(self.label_bank[indices, :].shape, value.shape)
        self.label_bank[indices, :] = self.beta * self.label_bank[indices, :] + (1 - self.beta) * value
        
        # print(epoch)
        if epoch==self.warm_up:
            self.updated=1
            return self.label_bank[indices, :]
        
        return y_pre
    
    
    def update_epoch_label(self, img_pred, pt_pred, indices):
        img_pred = torch.argmax(img_pred,dim=-1)
        pt_pred = torch.argmax(pt_pred,dim=-1)
        
        self.epoch_img_label[indices, self.ptr] = img_pred
        self.epoch_pt_label[indices, self.ptr] = pt_pred
    
    def update_ptr(self):
        self.ptr = (self.ptr + 1)%self.historical_length
        
    def save_label(self, path):
        np.save(path, self.label_bank.numpy())
    
    
    def get_labels(self):
        return self.label_bank
    
    def set_epoch(self, epoch):
        self.epoch = epoch


def cross_model_divide(args, img_net, pt_net, model, epoch, eval_train_loader,  model_name):
    
    pt_net.eval()
    model.eval()
    img_net.eval()
    # mesh_net.eval()
    
    num_samples = len(eval_train_loader.dataset)
    costs = torch.zeros(num_samples)
    mask = torch.zeros(num_samples, dtype=torch.bool)
    with torch.no_grad():
        for batch_idx, (img_list, pt_feat, targets, targets_ori,  indices) in enumerate(eval_train_loader):
            img_feat, img_feat_v = img_list[0], img_list[1]  
            img_feat = Variable(img_feat).to(torch.float32).to('cuda')
            img_feat_v = Variable(img_feat_v).to(torch.float32).to('cuda')
            
            pt_feat = Variable(pt_feat).to(torch.float32).to('cuda')
            # pt_feat = pt_feat.permute(0,2,1)
            
            targets = Variable(targets).to(torch.long).to('cuda')
            targets_ori = Variable(targets_ori).to(torch.long).to('cuda')
            
            _img_feat, _pt_feat = img_net(img_feat, img_feat_v), pt_net(pt_feat)
            # _mesh_feat = mesh_net(centers, corners, normals, neighbor_index)

            img_pred, pt_pred, joint_pred = model(_img_feat, _pt_feat)
            batch_mask = (targets == targets_ori).bool().cpu()  
            mask[indices] = batch_mask
            cost = None
            YA = YangaoLoss(operation='none')
            cost_img = YA(img_pred,targets)
            cost_pt = YA(pt_pred,targets)
            cost_joint = YA(joint_pred,targets)
            
            cost = cost_joint + (cost_img + cost_pt)/2.0
            
            indices = indices.to(torch.long)
            costs[indices] = cost.cpu().squeeze(1)
    
           
    costs = (costs-costs.min())/(costs.max()-costs.min())    
    costs = costs.reshape(-1,1)
        
    gmm = GaussianMixture(n_components=2,max_iter=10,reg_covar=5e-4,tol=1e-3)
    gmm.fit(costs)
    scores = gmm.predict_proba(costs)
    scores = scores[:,gmm.means_.argmin()]
    return scores,gmm.means_.max()
def cross_model_imgtext_divide(args, img_net, pt_net, model, epoch, eval_train_loader,  input_data_attr,model_name):
    
    pt_net.eval()
    model.eval()
    img_net.eval()
    # mesh_net.eval()
    num_samples = input_data_attr['num_train']
    # num_samples = len(eval_train_loader.dataset)
    costs = torch.zeros(num_samples)
    mask = torch.zeros(num_samples, dtype=torch.bool)
    with torch.no_grad():
        for batch_idx, (img, txt, labels, ori_labels,  indices) in enumerate(eval_train_loader):
            img_feat = Variable(img).to(torch.float32).to('cuda')
            
            pt_feat = Variable(txt).to(torch.float32).to('cuda')
            # pt_feat = pt_feat.permute(0,2,1)
            targets = labels.argmax(dim=1)
            targets_ori = ori_labels.argmax(dim=1)
            targets = Variable(targets).to(torch.long).to('cuda')
            targets_ori = Variable(targets_ori).to(torch.long).to('cuda')
            
            _img_feat, _pt_feat = img_net(img_feat), pt_net(pt_feat)

            img_pred, pt_pred, joint_pred = model(_img_feat, _pt_feat)
            batch_mask = (targets == targets_ori).bool().cpu()  
            mask[indices] = batch_mask
            cost = None
            YA = YangaoLoss(operation='none')
            cost_img = YA(img_pred,targets)
            cost_pt = YA(pt_pred,targets)
            cost_joint = YA(joint_pred,targets)
            
            cost = cost_joint + (cost_img + cost_pt)/2.0
            
            indices = indices.to(torch.long)
            costs[indices] = cost.cpu().squeeze(1)
    
           
    costs = (costs-costs.min())/(costs.max()-costs.min())    
    costs = costs.reshape(-1,1)

    # plt.hist(costs, bins=100)
    # plt.savefig("costs/costs_histogram{}.png".format(epoch), dpi=300) 
    # plt.clf()  # 清空当前图像
    # plt.close()
    gmm = GaussianMixture(n_components=2,max_iter=50,reg_covar=5e-4,tol=1e-4)
    gmm.fit(costs)
    if not gmm.converged_:
        print("GMM未收敛，可能需要提高max_iter或调整reg_covar")
    scores = gmm.predict_proba(costs)
    scores = scores[:,gmm.means_.argmin()]
    x = np.linspace(min(costs), max(costs), 1000).reshape(-1, 1)
    logprob = gmm.score_samples(x)
    pdf = np.exp(logprob)
    print("min:", costs.min().item(), "max:", costs.max().item())
    print("mean:", costs.mean().item(), "std:", costs.std().item())
    print("GMMMeans:", gmm.means_.flatten())
    print("Variances:", gmm.covariances_.flatten())
    print("Weights:", gmm.weights_.flatten())
    return scores,gmm.means_.max()
