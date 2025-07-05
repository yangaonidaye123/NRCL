from __future__ import division, absolute_import
import numpy as np
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
# import wandb
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from torch.autograd import Variable
import torchvision.models as models
from method.train_refine import refinement_datasets
import argparse
import torch.optim as optim
import time
from method.warmup_modelnet import warmup
# from method.trainselect_modelnet import refinement_datasets
import random
from sklearn.preprocessing import normalize
from losses.YaR_loss import YA_loss
from models.image_encoder import Img_encoder, HeadNet_dual_fused
from models.pt_encoder import Pt_encoder
from tools.dataset import ModelNet_Dataset
from tools.utils import calculate_accuracy
from losses.cross_modal_loss import CrossModalLoss

from tools.utils import EMA, cross_model_divide
import warnings
import scipy
warnings.filterwarnings('ignore',category=FutureWarning)

def fx_calc_map_label(view_1, view_2, view_3,view_4,label_test):
    dist_a = scipy.spatial.distance.cdist(view_1, view_2, 'cosine') #rows view_1 , columns view 2 
    dist_b = scipy.spatial.distance.cdist(view_3, view_4, 'cosine')
    dist = (dist_a + dist_b) * 0.5
    ord = dist.argsort()
    # print("sort dist finished.....")
    numcases = dist.shape[0]
    res = []
    for i in range(numcases):
        order = ord[i]
        p = 0.0
        r = 0.0
        for j in range(numcases):
            if label_test[i] == label_test[order[j]]:
                r += 1
                p += (r / (j + 1))
        if r > 0:
            res += [p / r]
        else:
            res += [0]
    return np.mean(res)


def test2(args,
        epoch,
        img_net_a, 
        pt_net_a, 
        img_net_b,
        pt_net_b,
        eval_loader):
    
    pt_net_a.eval()
    # head_net.eval()
    img_net_a.eval()
    pt_net_b.eval()
    # head_net.eval()
    img_net_b.eval()
    img_feat_list_a = np.zeros((len(eval_loader.dataset), 256))
    img_feat_list_b = np.zeros((len(eval_loader.dataset), 256))
    img_feat_list_a4 = np.zeros((len(eval_loader.dataset), 256))
    img_feat_list_b4 = np.zeros((len(eval_loader.dataset), 256))
    pt_feat_list_a = np.zeros((len(eval_loader.dataset), 256))
    pt_feat_list_b = np.zeros((len(eval_loader.dataset), 256))
    label = np.zeros((len(eval_loader.dataset)))
    #################################
    iteration = 0
    for data in eval_loader:
        img_list, pt_feat, noisy_label, ori_label, indices = data
        batch_size = ori_label.shape[0]
        
        img_feat1, img_feat2,img_feat3,img_feat4= img_list    
        img_feat1 = Variable(img_feat1).to(torch.float32).to('cuda')
        img_feat2 = Variable(img_feat2).to(torch.float32).to('cuda')
        img_feat3 = Variable(img_feat3).to(torch.float32).to('cuda')
        img_feat4 = Variable(img_feat4).to(torch.float32).to('cuda')
        pt_feat = Variable(pt_feat).to(torch.float32).to('cuda')
        # pt_feat = pt_feat.permute(0,2,1)

        ori_label = Variable(ori_label).to(torch.long).to('cuda')
        ##########################################
        img_net_a = img_net_a.to('cuda')
        img_net_b = img_net_b.to('cuda')
        _img_feat1_a = img_net_a(img_feat1, img_feat2)
        _img_feat1_a2 = img_net_a(img_feat3, img_feat4)

        _img_feat1_b = img_net_b(img_feat1, img_feat2)
        _img_feat1_b2 = img_net_b(img_feat3, img_feat4)
        pt_net_a = pt_net_a.to('cuda')
        pt_net_b = pt_net_b.to('cuda')
        _pt_feat_a = pt_net_a(pt_feat)
        _pt_feat_b = pt_net_b(pt_feat)
        feat_views_a = _img_feat1_a
        feat_view_a_4 = 0.5 * (_img_feat1_a2 + _img_feat1_a)
        feat_views_b = _img_feat1_b
        feat_view_b_4 = 0.5 * (_img_feat1_b2 + _img_feat1_b)
        img_feat_list_a[iteration:iteration+batch_size, :] = feat_views_a.data.cpu().numpy()
        img_feat_list_b[iteration:iteration+batch_size, :] = feat_views_b.data.cpu().numpy()
        img_feat_list_a4[iteration:iteration+batch_size, :] = feat_view_a_4.data.cpu().numpy()
        img_feat_list_b4[iteration:iteration+batch_size, :] = feat_view_b_4.data.cpu().numpy()
        pt_feat_list_a[iteration:iteration+batch_size, :] = _pt_feat_a.data.cpu().numpy()
        pt_feat_list_b[iteration:iteration+batch_size, :] = _pt_feat_b.data.cpu().numpy()
        label[iteration:iteration+batch_size] = ori_label.data.cpu().numpy()
        iteration = iteration+batch_size
    pt_eval_a = normalize(pt_feat_list_a, norm='l1', axis=1)
    img_eval_a = normalize(img_feat_list_a, norm='l1', axis=1)
    img_eval_a4 = normalize(img_feat_list_a4, norm='l1', axis=1)
    pt_eval_b = normalize(pt_feat_list_b, norm='l1', axis=1)
    img_eval_b = normalize(img_feat_list_b, norm='l1', axis=1)
    img_eval_b4 = normalize(img_feat_list_b4, norm='l1', axis=1)
    i2p_acc = fx_calc_map_label(img_eval_a,pt_eval_a, img_eval_b,pt_eval_b,label)
    i2p_acc = round(i2p_acc*100,2)
    p2i_acc = fx_calc_map_label(pt_eval_a,img_eval_a, pt_eval_b,img_eval_b,label)
    p2i_acc = round(p2i_acc*100,2)
    print("Eval Epoch%s: I2P:%.2f, P2I:%.2f"%(epoch, i2p_acc, p2i_acc))

    i2p_acc = fx_calc_map_label(img_eval_a4,pt_eval_a, img_eval_b4,pt_eval_b,label)
    i2p_acc = round(i2p_acc*100,2)
    p2i_acc = fx_calc_map_label(pt_eval_a,img_eval_a4, pt_eval_b,img_eval_b4,label)
    p2i_acc = round(p2i_acc*100,2)
    return i2p_acc, p2i_acc
def test1(args,
        epoch,
        img_net_a, 
        pt_net_a, 
        img_net_b,
        pt_net_b,
        eval_loader):
    
    pt_net_a.eval()
    # head_net.eval()
    img_net_a.eval()
    pt_net_b.eval()
    # head_net.eval()
    img_net_b.eval()
    img_feat_list_a = np.zeros((len(eval_loader.dataset), 256))
    img_feat_list_b = np.zeros((len(eval_loader.dataset), 256))
    pt_feat_list_a = np.zeros((len(eval_loader.dataset), 256))
    pt_feat_list_b = np.zeros((len(eval_loader.dataset), 256))
    label = np.zeros((len(eval_loader.dataset)))
    iteration = 0
    for data in eval_loader:
        img_list, pt_feat, noisy_label, ori_label, indices = data
        batch_size = ori_label.shape[0]
        
        img_feat1, img_feat2= img_list    
        img_feat1 = Variable(img_feat1).to(torch.float32).to('cuda')
        img_feat2 = Variable(img_feat2).to(torch.float32).to('cuda')
        
        pt_feat = Variable(pt_feat).to(torch.float32).to('cuda')
        # pt_feat = pt_feat.permute(0,2,1)

        ori_label = Variable(ori_label).to(torch.long).to('cuda')
        ##########################################
        img_net_a = img_net_a.to('cuda')
        img_net_b = img_net_b.to('cuda')
        _img_feat1_a = img_net_a(img_feat1, img_feat2)

        _img_feat1_b = img_net_b(img_feat1, img_feat2)
        pt_net_a = pt_net_a.to('cuda')
        pt_net_b = pt_net_b.to('cuda')
        _pt_feat_a = pt_net_a(pt_feat)
        _pt_feat_b = pt_net_b(pt_feat)
        feat_views_a = _img_feat1_a
        feat_views_b = _img_feat1_b
        img_feat_list_a[iteration:iteration+batch_size, :] = feat_views_a.data.cpu().numpy()
        img_feat_list_b[iteration:iteration+batch_size, :] = feat_views_b.data.cpu().numpy()
        pt_feat_list_a[iteration:iteration+batch_size, :] = _pt_feat_a.data.cpu().numpy()
        pt_feat_list_b[iteration:iteration+batch_size, :] = _pt_feat_b.data.cpu().numpy()
        label[iteration:iteration+batch_size] = ori_label.data.cpu().numpy()
        iteration = iteration+batch_size
    
    pt_eval_a = normalize(pt_feat_list_a, norm='l1', axis=1)
    img_eval_a = normalize(img_feat_list_a, norm='l1', axis=1)
    pt_eval_b = normalize(pt_feat_list_b, norm='l1', axis=1)
    img_eval_b = normalize(img_feat_list_b, norm='l1', axis=1)

    # plot_tsne_modalities_separated(img_eval_a,pt_eval_a,label,save_path="./visualize",epoch=epoch)
    i2p_acc = fx_calc_map_label(img_eval_a,pt_eval_a, img_eval_b,pt_eval_b,label)
    i2p_acc = round(i2p_acc*100,2)
    p2i_acc = fx_calc_map_label(pt_eval_a,img_eval_a, pt_eval_b,img_eval_b,label)
    p2i_acc = round(p2i_acc*100,2)
    print("Eval Epoch%s: I2P:%.2f, P2I:%.2f"%(epoch, i2p_acc, p2i_acc))
    
    return i2p_acc, p2i_acc

def training(args):
    if not os.path.exists(args.save):
        os.makedirs(args.save)

    img_net_a = Img_encoder(pre_trained = None)
    pt_net_a = Pt_encoder(args)
    model_a = HeadNet_dual_fused(num_classes=args.num_classes)

    img_net_b = Img_encoder(pre_trained = None)
    pt_net_b = Pt_encoder(args)
    model_b = HeadNet_dual_fused(num_classes=args.num_classes)
    
    img_net_a.train(True)
    img_net_a = img_net_a.to('cuda')
    pt_net_a.train(True)
    pt_net_a = pt_net_a.to('cuda')
    model_a.train(True)
    model_a = model_a.to('cuda')

    img_net_b.train(True)
    img_net_b = img_net_b.to('cuda')
    pt_net_b.train(True)
    pt_net_b = pt_net_b.to('cuda')
    model_b.train(True)
    model_b = model_b.to('cuda')

    ce = nn.CrossEntropyLoss()
    sem_criterion_a = YA_loss(num_classes=args.num_classes, feat_dim=256, temperature=args.center_temp)
    sem_criterion_b = YA_loss(num_classes=args.num_classes, feat_dim=256, temperature=args.center_temp)

    inst_criterion = CrossModalLoss(modal_num=2)
    
    optimizer_img_a = optim.Adam(img_net_a.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    optimizer_pt_a = optim.Adam(pt_net_a.parameters(), lr=args.lr_pt, weight_decay=args.weight_decay)
    optimizer_model_a = optim.Adam(model_a.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    optimizer_centloss_a = optim.Adam(sem_criterion_a.parameters(), lr=args.lr_center)

    optimizer_img_b = optim.Adam(img_net_b.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    optimizer_pt_b = optim.Adam(pt_net_b.parameters(), lr=args.lr_pt, weight_decay=args.weight_decay)
    optimizer_model_b = optim.Adam(model_b.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    optimizer_centloss_b = optim.Adam(sem_criterion_b.parameters(), lr=args.lr_center)

    train_set = ModelNet_Dataset(dataset = args.dataset, num_points = args.num_points, num_classes=args.num_classes, 
                                 dataset_dir=args.dataset_dir,  partition='train', 
                                 noise_rate=args.noise_rate, noise_type=args.noise_type)
    
    eval_set = ModelNet_Dataset(dataset=args.dataset, num_points = args.num_points, num_classes=args.num_classes,
                                  dataset_dir=args.dataset_dir, partition="test", 
                                  noise_rate=args.noise_rate, noise_type=args.noise_type)

    data_loader = torch.utils.data.DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=4)
    
    eval_data_loader = torch.utils.data.DataLoader(eval_set, batch_size=args.eval_batch_size, shuffle=False, num_workers=1)
    
    print("train batch number:%s"%(len(data_loader)))
    print("eval batch number:%s"%(len(eval_data_loader)))

    ema_epoch_a = EMA(num_data=train_set.__len__(), num_classes=args.num_classes, beta=args.beta, warm_up=args.warm_up)
    ema_epoch_a.initial_bank(data_loader)

    ema_epoch_b = EMA(num_data=train_set.__len__(), num_classes=args.num_classes, beta=args.beta, warm_up=args.warm_up)
    ema_epoch_b.initial_bank(data_loader)
    
    
    gt_clean, gt_noisy = train_set.get_gt_divide()
    gt_clean, gt_noisy = torch.from_numpy(gt_clean), torch.from_numpy(gt_noisy)
    best_I2P = 0.0
    best_P2I = 0.0
    best_result = 0.0
    clean_indices_a, noisy_indices_a=[],[]
    clean_scores_a = []

    clean_indices_b, noisy_indices_b=[],[]
    clean_scores_b = []
    
    warmup_loader = torch.utils.data.DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=4)
    print("num batch: ",len(warmup_loader))
    print('Warmuping Network......')
    
    iteration_a = 0
    iteration_b = 0
    start_time = time.time()
    
    divide_acc_list_a = torch.zeros(args.epochs)
    divide_rec_list_a = torch.zeros(args.epochs)
    data_acc_list_a = torch.zeros(args.epochs)
    i2p_epochs_a = torch.zeros(args.epochs)
    p2i_epochs_a = torch.zeros(args.epochs)

    divide_acc_list_b = torch.zeros(args.epochs)
    divide_rec_list_b = torch.zeros(args.epochs)
    data_acc_list_b = torch.zeros(args.epochs)
    i2p_epochs_b = torch.zeros(args.epochs)
    p2i_epochs_b = torch.zeros(args.epochs)
    
    for epoch in range(args.epochs):
        
        if epoch == args.warm_up:
            lr = 0.0001
            print('New  Learning Rate:     ' + str(lr))
            for param_group in optimizer_img_a.param_groups:
                param_group['lr'] = lr
            for param_group in optimizer_pt_a.param_groups:
                param_group['lr'] = lr
            for param_group in optimizer_model_a.param_groups:
                param_group['lr'] = lr

            for param_group in optimizer_img_b.param_groups:
                param_group['lr'] = lr
            for param_group in optimizer_pt_b.param_groups:
                param_group['lr'] = lr
            for param_group in optimizer_model_b.param_groups:
                param_group['lr'] = lr

        if epoch == args.warm_up:
            lr_center = 0.0001
            print('New  Center LR:     ' + str(lr_center))
            for param_group in optimizer_centloss_a.param_groups:
                param_group['lr'] = lr_center
            for param_group in optimizer_centloss_b.param_groups:
                param_group['lr'] = lr_center
                
        if (epoch%args.lr_step) == 0:
            lr = args.lr * (0.1 ** (epoch // args.lr_step))
            print('New  Learning Rate:     ' + str(lr))
            for param_group in optimizer_img_a.param_groups:
                param_group['lr'] = lr
            for param_group in optimizer_pt_a.param_groups:
                param_group['lr'] = lr
            for param_group in optimizer_model_a.param_groups:
                param_group['lr'] = lr
            for param_group in optimizer_img_b.param_groups:
                param_group['lr'] = lr
            for param_group in optimizer_pt_b.param_groups:
                param_group['lr'] = lr
            for param_group in optimizer_model_b.param_groups:
                param_group['lr'] = lr

      
        if (epoch%args.lr_step) == 0:
            lr_center = args.lr_center * (0.1 ** (epoch // args.lr_step))
            print('New  Center LR:     ' + str(lr_center))
            for param_group in optimizer_centloss_a.param_groups:
                param_group['lr'] = lr_center
            for param_group in optimizer_centloss_b.param_groups:
                param_group['lr'] = lr_center
        if epoch <= 2:
            eval_loader = torch.utils.data.DataLoader(train_set, batch_size=args.batch_size, shuffle=False, num_workers=4, drop_last=False)
            scores_ce_a,mean_a = cross_model_divide(args, img_net_a, pt_net_a, model_a, epoch, eval_loader,'A')
            scores_ce_b,mean_b= cross_model_divide(args, img_net_b, pt_net_b, model_b, epoch, eval_loader,'B')
        if epoch < args.warm_up:     # warm up  
          
            iteration_a = warmup(args, epoch, img_net_a, pt_net_a, model_a, 
                   warmup_loader, optimizer_img_a, optimizer_pt_a, optimizer_model_a, optimizer_centloss_a, 
                   ce, inst_criterion, sem_criterion_a)
            iteration_b = warmup(args, epoch, img_net_b, pt_net_b, model_b, 
                   warmup_loader, optimizer_img_b, optimizer_pt_b, optimizer_model_b, optimizer_centloss_b, 
                   ce, inst_criterion, sem_criterion_b)
             
        
        else:
            if (epoch >= args.warm_up and epoch <= args.warm_up+16) or (epoch%20==0): 
                print('Training Model......')
                batch_size_clean_a = ((clean_indices_a.shape[0]/(clean_indices_a.shape[0]+noisy_indices_a.shape[0]))*args.batch_size//2)*2
                batch_size_clean_a = max(int(batch_size_clean_a), args.batch_size//2)
                batch_size_noisy_a = int(args.batch_size - batch_size_clean_a)

                batch_size_clean_b = ((clean_indices_b.shape[0]/(clean_indices_b.shape[0]+noisy_indices_b.shape[0]))*args.batch_size//2)*2
                batch_size_clean_b = max(int(batch_size_clean_b), args.batch_size//2)
                batch_size_noisy_b = int(args.batch_size - batch_size_clean_b)
       
                
                print("A model clean and noisy batch size:%.2f,%.2f"%(batch_size_clean_a, batch_size_noisy_a))
                print("B model clean and noisy batch size:%.2f,%.2f"%(batch_size_clean_b, batch_size_noisy_b))
                train_set_clean_a = ModelNet_Dataset(dataset=args.dataset, num_points=args.num_points, num_classes=args.num_classes,
                                    dataset_dir=args.dataset_dir, partition="train", 
                                    noise_rate=args.noise_rate, select_indices=clean_indices_a, 
                                    noise_type=args.noise_type)
                train_loader_clean_a = torch.utils.data.DataLoader(train_set_clean_a, batch_size=batch_size_clean_a, shuffle=True, num_workers=8, drop_last=True)
                
                train_set_noisy_a = ModelNet_Dataset(dataset=args.dataset, num_points=args.num_points, num_classes=args.num_classes,
                                    dataset_dir=args.dataset_dir, partition="train", 
                                    noise_rate=args.noise_rate, select_indices=noisy_indices_a, 
                                    noise_type=args.noise_type)
                train_loader_noisy_a = torch.utils.data.DataLoader(train_set_noisy_a, batch_size=batch_size_noisy_a, shuffle=True, num_workers=8, drop_last=True)

                train_set_clean_b = ModelNet_Dataset(dataset=args.dataset, num_points=args.num_points, num_classes=args.num_classes,
                                    dataset_dir=args.dataset_dir, partition="train", 
                                    noise_rate=args.noise_rate, select_indices=clean_indices_b, 
                                    noise_type=args.noise_type)
                train_loader_clean_b = torch.utils.data.DataLoader(train_set_clean_b, batch_size=batch_size_clean_b, shuffle=True, num_workers=8, drop_last=True)
                
                train_set_noisy_b = ModelNet_Dataset(dataset=args.dataset, num_points=args.num_points, num_classes=args.num_classes,
                                    dataset_dir=args.dataset_dir, partition="train", 
                                    noise_rate=args.noise_rate, select_indices=noisy_indices_b, 
                                    noise_type=args.noise_type)
                train_loader_noisy_b = torch.utils.data.DataLoader(train_set_noisy_b, batch_size=batch_size_noisy_b, shuffle=True, num_workers=8, drop_last=True)
                

            iteration_a = refinement_datasets(args, epoch, img_net_a, pt_net_a, model_a,img_net_b,pt_net_b,model_b, 
                   train_loader_clean_b, train_loader_noisy_b, optimizer_img_a, optimizer_pt_a, optimizer_model_a, optimizer_centloss_a, 
                   inst_criterion, sem_criterion_a, ema_epoch_a,iteration_a)
            iteration_b = refinement_datasets(args, epoch, img_net_b, pt_net_b, model_b, img_net_a, pt_net_a, model_a,
                   train_loader_clean_a, train_loader_noisy_a, optimizer_img_b, optimizer_pt_b, optimizer_model_b, optimizer_centloss_b, 
                   inst_criterion, sem_criterion_b, ema_epoch_b,iteration_b)
        
        if epoch%5==0:
            print('\n==== eval %s epoch ===='%(epoch))
            # if epoch%10==0:
            with torch.no_grad(): 
                # i2p_acc_a, p2i_acc_a = test(args, epoch, img_net_a, pt_net_a, eval_data_loader)
                # i2p_acc_b, p2i_acc_b = test(args, epoch, img_net_b, pt_net_b, eval_data_loader)
                # i2p_acc_a, p2i_acc_a = test1(args, epoch, img_net_a, pt_net_a, img_net_b,pt_net_b,eval_data_loader)
                i2p_acc_a, p2i_acc_a = test1(args, epoch, img_net_a, pt_net_a, img_net_b,pt_net_b,eval_data_loader)
                # i2p_acc_a_4, p2i_acc_a_4 = test2(args, epoch, img_net_a, pt_net_a, img_net_b,pt_net_b,eval_data_loader)
                i2p_epochs_a[epoch] = i2p_acc_a
                p2i_epochs_a[epoch] = p2i_acc_a

                # i2p_epochs_b[epoch] = i2p_acc_b
                # p2i_epochs_b[epoch] = p2i_acc_b
                if (i2p_acc_a + p2i_acc_a)>best_result:
                    best_result = i2p_acc_a +p2i_acc_a
                    best_I2P = i2p_acc_a
                    best_P2I = p2i_acc_a

                    print('----------------- Save The Network ------------------------')
                    with open(args.save + 'best-head_net_a.pkl', 'wb') as f:
                        torch.save(model_a, f)
                    with open(args.save + 'best-img_net_a.pkl', 'wb') as f:
                        torch.save(img_net_a, f)
                    with open(args.save + 'best-pt_net_a.pkl', 'wb') as f:
                        torch.save(pt_net_a, f)
                    print('----------------- Save The Network ------------------------')
                    with open(args.save + 'best-head_net_b.pkl', 'wb') as f:
                        torch.save(model_b, f)
                    with open(args.save + 'best-img_net_b.pkl', 'wb') as f:
                        torch.save(img_net_b, f)
                    with open(args.save + 'best-pt_net_b.pkl', 'wb') as f:
                        torch.save(pt_net_b, f)
                # wandb.log({
                #     "epoch":epoch,
                #     "best_i2p_acc":best_I2P,
                #     "best_p2i_acc":best_P2I,
                # })
            
                # wandb.log({
                #     "epoch":epoch,
                #     "epoch_i2p_acc":i2p_acc_a,
                #     "epoch_p2i_acc":p2i_acc_a,
                #     # "epoch_i2p_acc_4":i2p_acc_a_4,
                #     # "epoch_p2i_acc_4":p2i_acc_a_4
                # })
                # if (i2p_acc_a+p2i_acc_a)>(i2p_acc_b+p2i_acc_b) and (i2p_acc_a + p2i_acc_a)>best_result:
                #     best_result = i2p_acc_a +p2i_acc_a
                #     best_I2P = i2p_acc_a
                #     best_P2I = p2i_acc_a

                #     print('----------------- Save The Network ------------------------')
                #     with open(args.save + 'best-head_net_a.pkl', 'wb') as f:
                #         torch.save(model_a, f)
                #     with open(args.save + 'best-img_net_a.pkl', 'wb') as f:
                #         torch.save(img_net_a, f)
                #     with open(args.save + 'best-pt_net_a.pkl', 'wb') as f:
                #         torch.save(pt_net_a, f)
                # if (i2p_acc_a + p2i_acc_a) < (i2p_acc_b + p2i_acc_b) and (i2p_acc_b+p2i_acc_b) >best_result:
                #     best_result = p2i_acc_b + i2p_acc_b
                #     best_P2I = p2i_acc_b
                #     best_I2P = i2p_acc_b
                #     print('----------------- Save The Network ------------------------')
                #     with open(args.save + 'best-head_net_b.pkl', 'wb') as f:
                #         torch.save(model_b, f)
                #     with open(args.save + 'best-img_net_b.pkl', 'wb') as f:
                #         torch.save(img_net_b, f)
                #     with open(args.save + 'best-pt_net_b.pkl', 'wb') as f:
                #         torch.save(pt_net_b, f)
                # wandb.log({
                #     "epoch":epoch,
                #     "best_i2p_acc":best_I2P,
                #     "best_p2i_acc":best_P2I,
                # })
                # if (i2p_acc_a + p2i_acc_a) < (i2p_acc_b + p2i_acc_b):
                #     wandb.log({
                #     "epoch":epoch,
                #     "epoch_i2p_acc":i2p_acc_b,
                #     "epoch_p2i_acc":p2i_acc_b
                # })
                # else:
                #     wandb.log({
                #     "epoch":epoch,
                #     "epoch_i2p_acc":i2p_acc_a,
                #     "epoch_p2i_acc":p2i_acc_a
                # })
         
        if (epoch >= 3 and epoch <= args.warm_up+10) or (epoch >= args.warm_up-1 and (epoch+1)%20==0): 
            print('\n==== Divide %s epoch training data ===='%(epoch+1)) 
            ###get centers

            eval_loader = torch.utils.data.DataLoader(train_set, batch_size=args.batch_size, shuffle=False, num_workers=4, drop_last=False)
            scores_ce_a,mean_a = cross_model_divide(args, img_net_a, pt_net_a, model_a, epoch, eval_loader,'A')
            scores_ce_b,mean_b= cross_model_divide(args, img_net_b, pt_net_b, model_b, epoch, eval_loader,'B')
            
            clean_scores_a = torch.from_numpy(scores_ce_a)
            clean_scores_b = torch.from_numpy(scores_ce_b)
            
            clean_indices_a = torch.nonzero(clean_scores_a>args.p_threshold).squeeze()

            # 获取为False的元素的索引
            noisy_indices_a = torch.nonzero(clean_scores_a<=args.p_threshold).squeeze()
            correct_clean_a = gt_clean[clean_indices_a.cpu().numpy()]
            num_clean = np.sum(gt_clean.cpu().numpy())
            num_predicted_clean_a = correct_clean_a.sum().item()
            print("number of clean samples in dataset:",num_clean)
            print("number of clean samples in gmm:",clean_indices_a.shape[0])
            print("number of clean samples in prediction:",num_predicted_clean_a)

            clean_indices_b = torch.nonzero(clean_scores_b>args.p_threshold).squeeze()

            # 获取为False的元素的索引
            noisy_indices_b = torch.nonzero(clean_scores_b<=args.p_threshold).squeeze()
            
            weight_clean_a = clean_indices_a.shape[0]/(clean_indices_a.shape[0]+noisy_indices_a.shape[0])
            weight_noisy_a = 1-weight_clean_a
            print("A model clean and noisy learning weight:%.2f,%.2f"%(weight_clean_a, weight_noisy_a))
            
            weight_clean_b = clean_indices_b.shape[0]/(clean_indices_b.shape[0]+noisy_indices_b.shape[0])
            weight_noisy_b = 1-weight_clean_b
            print("B model clean and noisy learning weight:%.2f,%.2f"%(weight_clean_b, weight_noisy_b))
            # print(clean_indices.shape, noisy_indices.shape)
            pred_clean_a = clean_scores_a>args.p_threshold
            # num_data = gt_clean.shape[0]
            divide_acc_a = (((gt_clean == pred_clean_a)&pred_clean_a).sum())/pred_clean_a.sum()
            divide_rec_a = (((gt_clean == pred_clean_a)&gt_clean).sum())/gt_clean.sum()
            data_acc_a = ((gt_clean == pred_clean_a).sum())/pred_clean_a.shape[0]
            ##计算divide的ac
            ##计算divide的acc
            print("A model Divide acc: %.4f"%divide_acc_a.item())
            print("A model Divide rec: %.4f"%divide_rec_a.item())
            print("A model Data acc: %.4f"%data_acc_a.item())
            divide_acc_list_a[epoch]=divide_acc_a
            divide_rec_list_a[epoch]=divide_rec_a
            data_acc_list_a[epoch]=data_acc_a

            pred_clean_b = clean_scores_b>args.p_threshold
            # num_data = gt_clean.shape[0]
            divide_acc_b = (((gt_clean == pred_clean_b)&pred_clean_b).sum())/pred_clean_b.sum()
            divide_rec_b = (((gt_clean == pred_clean_b)&gt_clean).sum())/gt_clean.sum()
            data_acc_b = ((gt_clean == pred_clean_b).sum())/pred_clean_b.shape[0]
            ##计算divide的ac
            ##计算divide的acc
            print("B model Divide acc: %.4f"%divide_acc_b.item())
            print("B model Divide rec: %.4f"%divide_rec_b.item())
            print("B model Data acc: %.4f"%data_acc_b.item())
            divide_acc_list_b[epoch]=divide_acc_b
            divide_rec_list_b[epoch]=divide_rec_b
            data_acc_list_b[epoch]=data_acc_b
            
    
    print("Best Acc: %.2f, %.2f"%(best_I2P, best_P2I))


if __name__ == "__main__":
    # Training settings
    parser = argparse.ArgumentParser(description='Cross Modal Retrieval')

    parser.add_argument('--dataset', type=str, default='ModelNet10', metavar='dataset',
                        help='ModelNet10 or ModelNet40')

    parser.add_argument('--dataset_dir', type=str, default='/home/yangao/yangao_ModelNet/yangao_retrieval/datasets/', metavar='dataset_dir',
                        help='dataset_dir')

    parser.add_argument('--num_classes', type=int, default=10, metavar='num_classes',
                        help='10 or 40')

    parser.add_argument('--batch_size', type=int, default=128, metavar='batch_size',
                        help='Size of batch)')
    parser.add_argument('--eval_batch_size', type=int, default=20, metavar='batch_size',
                        help='Size of batch)')
    parser.add_argument('--epochs', type=int, default=400, metavar='N',
                        help='number of episode to train ')
    
    parser.add_argument('--warm_up', type=int, default=5, metavar='N',
                        help='number of episode to train ')
    #optimizer
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                        help='learning rate (default: 0.001, 0.1 if using sgd)')
    
    parser.add_argument('--lr_pt', type=float, default=0.001, metavar='LR',
                        help='learning rate (default: 0.001, 0.1 if using sgd)')

    parser.add_argument('--lr_step', type=int,  default=100,
                        help='how many iterations to decrease the learning rate')

    parser.add_argument('--lr_center', type=float, default=0.001, metavar='LR',
                        help='learning rate for center loss (default: 0.5)')
                                         
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    
    parser.add_argument('--p_threshold', default=0.5, type=float, help='clean probability threshold')

    #DGCNN
    parser.add_argument('--num_points', type=int, default=2048,
                        help='num of points to use')
    
    parser.add_argument('--k', type=int, default=20, metavar='N',
                        help='Num of nearest neighbors to use')
    
    parser.add_argument('--dropout', type=float, default=0.4,
                        help='dropout rate')
    parser.add_argument('--emb_dims', type=int, default=1024, metavar='N',
                        help='Dimension of embeddings') 
    #loss
    parser.add_argument('--w_sem_c', type=float, default=10, metavar='weight_center',
                        help='weight center (default: 1.0)')

    parser.add_argument('--w_ce_c', type=float, default=1, metavar='weight_ce',
                        help='weight ce' ) 

    parser.add_argument('--w_inst_c', type=float, default=1, metavar='weight_mse',
                        help='weight mse' )
    
    parser.add_argument('--w_sem_n', type=float, default=0.1, metavar='weight_center',
                        help='weight noisy center (default: 1.0)')

    parser.add_argument('--w_ce_n', type=float, default=0.1, metavar='weight_ce',
                        help='weight cls (noisy)' ) 
    
    parser.add_argument('--wn', type=float, default=1, metavar='weight_ce',
                        help='weight S_n' ) 

    parser.add_argument('--w_inst_n', type=float, default=1, metavar='weight_mse',
                        help='weight inst (noisy)' )

    parser.add_argument('--weight_decay', type=float, default=1e-3, metavar='weight_decay',
                        help='learning rate (default: 1e-3)')

    parser.add_argument('--per_save', type=int,  default=50,
                        help='how many iterations to save the model')

    parser.add_argument('--per_print', type=int,  default=20,
                        help='how many iterations to print the loss and accuracy')
                        
    parser.add_argument('--save', type=str,  default='./checkpoints_modelnet/ModelNet40/',
                        help='path to save the final model')

    parser.add_argument('--gpu_id', type=str,  default='1',
                        help='GPU used to train the network')
    
    parser.add_argument('--yrectified', action='store_true', help='GPU used to train the network')
    parser.add_argument('--yema', action='store_true', help='EMA')
    
    parser.add_argument('--noise_rate', type=float, default=0.60, metavar='noise_rate',
                        help='noise rate')
    # 0.4
    parser.add_argument('--center_temp', type=float, default=0.4, metavar='noise_rate',
                        help='noise rate')
    
    parser.add_argument('--beta', type=float, default=0.9, metavar='moving average beta',
                        help='weight mse' )
    # /home/yangao/yangao_ModelNet/yangao_retrieval/datasets/ModelNet10/train_asy_label_10.npy
    parser.add_argument('--noise_type', type=str, default='sym',
                        help='sym or asy')

    args = parser.parse_args()
    seed=2000
    random.seed(seed)
    torch.manual_seed(seed)
    print(args)
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
    training(args)