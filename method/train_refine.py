from __future__ import division, absolute_import
import torch
import torch.nn.functional as F
from torch.autograd import Variable
import time
import wandb
from tools.utils import calculate_accuracy
import warnings
warnings.filterwarnings('ignore',category=FutureWarning)
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import wandb

def refinement_datasets(args,
             epoch, 
             img_net_a, 
             pt_net_a,
             model_a,
             img_net_b, 
             pt_net_b,
             model_b,
             train_trainloader_clean,
             train_trainloader_noisy,
             optimizer_img, 
             optimizer_pt, 
             optimizer_model, 
             optimizer_cmc, 
             inst_criterion, 
             sem_criterion,
             MA_epoch,
             iteration):


    pt_net_a.train()
    model_a.train()
    img_net_a.train()
    pt_net_b.eval()
    model_b.eval()
    img_net_b.eval()
    noisy_train_iter = iter(train_trainloader_noisy)
    start_time = time.time()


    for data in train_trainloader_clean:
        img_list_c, pt_feat_c, target_c, target_ori_c, indices_c = data
        
        if target_c.shape[0]==1:
            continue
        
        try:
            img_list_n, pt_feat_n, target_n, target_ori_n, indices_n = next(noisy_train_iter)
            assert target_n.shape[0] != 1
        except:
            noisy_train_iter = iter(train_trainloader_noisy)
            img_list_n, pt_feat_n, target_n, target_ori_n, indices_n = next(noisy_train_iter)
        
        optimizer_img.zero_grad()
        optimizer_pt.zero_grad()
        optimizer_model.zero_grad()
        optimizer_cmc.zero_grad()
        
        
        
        img_feat_c, img_feat_v_c = img_list_c[0], img_list_c[1] 
        img_feat_c = Variable(img_feat_c).to(torch.float32).to('cuda')
        img_feat_v_c = Variable(img_feat_v_c).to(torch.float32).to('cuda')
        pt_feat_c = Variable(pt_feat_c).to(torch.float32).to('cuda')
        target_c = Variable(target_c).to(torch.long).to('cuda')
        target_ori_c = Variable(target_ori_c).to(torch.long).to('cuda')
        img_feat_n, img_feat_v_n = img_list_n[0], img_list_n[1] 
        img_feat_n = Variable(img_feat_n).to(torch.float32).to('cuda')
        img_feat_v_n = Variable(img_feat_v_n).to(torch.float32).to('cuda')
        pt_feat_n = Variable(pt_feat_n).to(torch.float32).to('cuda')
        target_n = Variable(target_n).to(torch.long).to('cuda')
        target_ori_n = Variable(target_ori_n).to(torch.long).to('cuda')
        
        img_feat = torch.cat((img_feat_c, img_feat_n), dim=0)
        img_feat_v = torch.cat((img_feat_v_c, img_feat_v_n), dim=0)
        
        pt_feat = torch.cat((pt_feat_c, pt_feat_n), dim=0)
        target = torch.cat((target_c, target_n), dim=0)
        target_ori = torch.cat((target_ori_c, target_ori_n), dim=0)
        indices = torch.cat((indices_c, indices_n), dim=0)
        
        clean_num = img_feat_c.shape[0]
        noisy_num = img_feat_n.shape[0]
        mask_clean = torch.arange(clean_num+noisy_num, dtype=torch.int32).cuda()
        mask_clean = mask_clean < clean_num
         
        ##noisy label
        y_clean = torch.nn.functional.one_hot(target_c, args.num_classes).float().cuda()
        
        y_n = torch.nn.functional.one_hot(target, args.num_classes).float().cuda()
        _img_feat, _pt_feat = img_net_a(img_feat, img_feat_v), pt_net_a(pt_feat)
        _img_pred, _pt_pred, _joint_pred = model_a(_img_feat, _pt_feat)
        _img_feat_b, _pt_feat_b = img_net_b(img_feat, img_feat_v), pt_net_b(pt_feat)
        _img_pred_b, _pt_pred_b, _joint_pred_b = model_b(_img_feat_b, _pt_feat_b)
        y_joint_pred = (_joint_pred + _joint_pred_b) * 0.5
        y_joint_pred = _joint_pred
        
        y_img_pred = F.softmax(_img_pred, dim=1)
        y_pt_pred = F.softmax(_pt_pred, dim=1)
       
        
        y_joint_pred = F.softmax(_joint_pred, dim=1)
        
        
        
        
        weight_mask = None

        y_hat = y_joint_pred
      

        

        if not args.yema:
            with torch.no_grad():
                y_e = y_hat.clone().detach().cpu()
                y_hat = MA_epoch.update(y_e, indices,epoch, weight = weight_mask)
                y_hat = y_hat.cuda()
        
        if args.yrectified:
            y_final_one_hot = y_n
            y_final = target
            w = None
        else:
            y_final_one_hot = y_hat.clone().detach()
            y_final_one_hot[mask_clean,:] = y_clean
            y_final = torch.argmax(y_hat, dim=1)
            w = None
        
        _pt_pred_s = _pt_pred[mask_clean]
        _img_pred_s = _img_pred[mask_clean]
        _joint_pred_s = _joint_pred[mask_clean]
        _img_feat_s = _img_feat[mask_clean]
        _pt_feat_s = _pt_feat[mask_clean]
        y_final_one_hot_s = y_final_one_hot[mask_clean]
        y_final_s = y_final[mask_clean]
        prior = torch.ones(args.num_classes)/args.num_classes
        prior = prior.cuda()

        pt_pred_mean = torch.softmax(_pt_pred_s, dim=1).mean(0)
        pt_penalty = torch.sum(prior*torch.log(prior/pt_pred_mean))
        
        pt_pred = torch.softmax(_pt_pred_s, dim=1)
        pt = torch.sum(pt_pred * y_final_one_hot_s, dim=1)  # shape: [batch_size]
        log_pt = torch.sum(F.log_softmax(_pt_pred_s, dim=1) * y_final_one_hot_s, dim=1)  # [batch_size]
        focal_weight_pt = (1.0 - pt)  # gamma = 1
        pt_cls_loss = -torch.mean(focal_weight_pt * log_pt)+ pt_penalty
        
        img_pred_mean = torch.softmax(_img_pred_s, dim=1).mean(0)
        img_penalty = torch.sum(prior*torch.log(prior/img_pred_mean))
        img_pred = torch.softmax(_img_pred_s, dim=1)
        img = torch.sum(img_pred * y_final_one_hot_s, dim=1)  # shape: [batch_size]
        log_img = torch.sum(F.log_softmax(_img_pred_s, dim=1) * y_final_one_hot_s, dim=1)  # [batch_size]
        focal_weight_img = (1.0 - img)  # gamma = 1
        img_cls_loss = -torch.mean(focal_weight_img * log_img)+ img_penalty
        
           
        joint_pred_mean = torch.softmax(_joint_pred_s, dim=1).mean(0)
        joint_penalty = torch.sum(prior*torch.log(prior/joint_pred_mean))
        joint_pred = torch.softmax(_joint_pred_s, dim=1)
        joint = torch.sum(joint_pred * y_final_one_hot_s, dim=1)  # shape: [batch_size]
        log_joint = torch.sum(F.log_softmax(_joint_pred_s, dim=1) * y_final_one_hot_s, dim=1)  # [batch_size]
        focal_weight_joint = (1.0 - joint)  # gamma = 1
        joint_crc_loss = -torch.mean(focal_weight_joint * log_joint) + joint_penalty
        

        cls_loss_c = (pt_cls_loss + img_cls_loss + 2*joint_crc_loss)/2
 
        sem_loss_c, centers = sem_criterion(torch.cat((_img_feat_s, _pt_feat_s), dim = 0), torch.cat((y_final_s, y_final_s), dim = 0),  epoch)
        
        inst_loss_c = inst_criterion(torch.cat((_img_feat_s, _pt_feat_s), dim = 0))
        
        loss_c = args.w_ce_c * cls_loss_c + args.w_sem_c * sem_loss_c + args.w_inst_c * inst_loss_c
        # loss_c = args.w_cls_c * cls_loss_c  + args.w_sem_c * sem_loss_c
        
        _pt_pred_s = _pt_pred[~mask_clean]
        _img_pred_s = _img_pred[~mask_clean]
        _joint_pred_s = _joint_pred[~mask_clean]
        _img_feat_s = _img_feat[~mask_clean]
        _pt_feat_s = _pt_feat[~mask_clean]
        y_final_one_hot_s = y_final_one_hot[~mask_clean]
        y_final_s = y_final[~mask_clean]
        
        
        pt_pred_mean = torch.softmax(_pt_pred_s, dim=1).mean(0)
        pt_penalty = torch.sum(prior*torch.log(prior/pt_pred_mean))
        # pt_cls_loss = -torch.mean(torch.sum(F.log_softmax(_pt_pred_s, dim=1) * y_final_one_hot_s, dim=1)) + pt_penalty
        pt_pred = torch.softmax(_pt_pred_s, dim=1)
        pt = torch.sum(pt_pred * y_final_one_hot_s, dim=1)  # shape: [batch_size]
        log_pt = torch.sum(F.log_softmax(_pt_pred_s, dim=1) * y_final_one_hot_s, dim=1)  # [batch_size]
        focal_weight_pt = (1.0 - pt)  # gamma = 1
        pt_cls_loss = -torch.mean(focal_weight_pt * log_pt)+ pt_penalty
        
        # regularization     
        img_pred_mean = torch.softmax(_img_pred_s, dim=1).mean(0)
        img_penalty = torch.sum(prior*torch.log(prior/img_pred_mean))
        img_pred = torch.softmax(_img_pred_s, dim=1)
        img = torch.sum(img_pred * y_final_one_hot_s, dim=1)  # shape: [batch_size]
        log_img = torch.sum(F.log_softmax(_img_pred_s, dim=1) * y_final_one_hot_s, dim=1)  # [batch_size]
        focal_weight_img = (1.0 - img)  # gamma = 1
        img_cls_loss = -torch.mean(focal_weight_img * log_img)+ img_penalty
        
        # regularization     
        joint_pred_mean = torch.softmax(_joint_pred_s, dim=1).mean(0)
        joint_penalty = torch.sum(prior*torch.log(prior/joint_pred_mean))
        joint_pred = torch.softmax(_joint_pred_s, dim=1)
        joint = torch.sum(joint_pred * y_final_one_hot_s, dim=1)  # shape: [batch_size]
        log_joint = torch.sum(F.log_softmax(_joint_pred_s, dim=1) * y_final_one_hot_s, dim=1)  # [batch_size]
        focal_weight_joint = (1.0 - joint)  # gamma = 1
        joint_crc_loss = -torch.mean(focal_weight_joint * log_joint)+ joint_penalty
        

        cls_loss_n = (pt_cls_loss + img_cls_loss + 2*joint_crc_loss)/2
        sem_loss_n, centers = sem_criterion(torch.cat((_img_feat_s, _pt_feat_s), dim = 0), torch.cat((y_final_s, y_final_s), dim = 0),  epoch)
        
        inst_loss_n = inst_criterion(torch.cat((_img_feat_s, _pt_feat_s), dim = 0))
        
        loss_n = args.w_ce_n * cls_loss_n + args.w_sem_n * sem_loss_n + args.w_inst_n * inst_loss_n
        # loss_n =   args.w_inst_n * inst_loss_n + args.w_sem_n * sem_loss_n
        
        loss = loss_c + args.wn*loss_n
        # loss = loss_c 
        
        torch.autograd.set_detect_anomaly(True)
        loss.backward()

        optimizer_pt.step()
        optimizer_img.step()

        optimizer_model.step()

        optimizer_cmc.step()
        
        img_acc = calculate_accuracy(y_img_pred, target_ori)
        pt_acc = calculate_accuracy(y_pt_pred, target_ori)
  
        joint_pred = (y_img_pred+y_pt_pred)/2
        joint_acc = calculate_accuracy(joint_pred, target_ori)
        joint_m_acc = calculate_accuracy(y_joint_pred, target_ori)
        
        ema_acc = calculate_accuracy(y_final_one_hot, target_ori)
        

        ema_acc_n = calculate_accuracy(y_final_one_hot[~mask_clean], target_ori[~mask_clean])

        # wandb.log({
        #     "enire_Acc":ema_acc,
        #     "correct_acc":ema_acc_n,
        #     "epoch":epoch
        # })
    
        if iteration % args.per_print == 0:
            print("[%d]  img_acc: %.4f  pt_acc: %.4f  joint_acc: %.4f jm_acc: %.4f ema_acc: %.4f ema_acc_n: %f" % (iteration, img_acc, pt_acc, joint_acc, joint_m_acc, ema_acc, ema_acc_n)) 
            start_time = time.time()
        iteration = iteration + 1
  
        

    return iteration


def refinement_imgtxt(args,
             epoch, 
             img_net_a, 
             pt_net_a,
             model_a,
             img_net_b, 
             pt_net_b,
             model_b,
             train_trainloader_clean,
             train_trainloader_noisy,
             optimizer_img, 
             optimizer_pt, 
             optimizer_model, 
             optimizer_cmc, 
             inst_criterion, 
             sem_criterion,
             MA_epoch,
             iteration):


    pt_net_a.train()
    # mesh_net.train()
    model_a.train()
    img_net_a.train()
    pt_net_b.eval()
    # mesh_net.train()
    model_b.eval()
    img_net_b.eval()
    noisy_train_iter = iter(train_trainloader_noisy)
    start_time = time.time()

    
    
    for data in train_trainloader_clean:
        img_c, txt_c, label_c, label_ori_c, indices_c = data
        
        if label_c.shape[0]==1:
            continue
        
        try:
            img_n, txt_n, label_n, label_ori_n, indices_n = next(noisy_train_iter)
            assert label_n.shape[0] != 1
        except:
            noisy_train_iter = iter(train_trainloader_noisy)
            img_n, txt_n, label_n, label_ori_n, indices_n = next(noisy_train_iter)
        
        optimizer_img.zero_grad()
        optimizer_pt.zero_grad()
        # optimizer_mesh.zero_grad()
        optimizer_model.zero_grad()
        optimizer_cmc.zero_grad()
        
        img_feat_c = Variable(img_c).to(torch.float32).to('cuda')

        pt_feat_c = Variable(txt_c).to(torch.float32).to('cuda')
        # pt_feat_c = pt_feat_c.permute(0,2,1)
        target_c = label_c.argmax(dim=1)
        target_c = Variable(target_c).to(torch.long).to('cuda')
        target_ori_c = label_ori_c.argmax(dim=1)
        target_ori_c = Variable(target_ori_c).to(torch.long).to('cuda')
        

        img_feat_n = Variable(img_n).to(torch.float32).to('cuda')
        pt_feat_n = Variable(txt_n).to(torch.float32).to('cuda')
        # pt_feat_n = pt_feat_n.permute(0,2,1)
        target_n = label_n.argmax(dim=1)
        target_n = Variable(target_n).to(torch.long).to('cuda')
        target_ori_n = label_ori_n.argmax(dim=1)
        target_ori_n = Variable(target_ori_n).to(torch.long).to('cuda')
        
        img_feat = torch.cat((img_feat_c, img_feat_n), dim=0)
        pt_feat = torch.cat((pt_feat_c, pt_feat_n), dim=0)
        target = torch.cat((target_c, target_n), dim=0)
        target_ori = torch.cat((target_ori_c, target_ori_n), dim=0)
        indices = torch.cat((indices_c, indices_n), dim=0)
        
        clean_num = img_feat_c.shape[0]
        noisy_num = img_feat_n.shape[0]
        mask_clean = torch.arange(clean_num+noisy_num, dtype=torch.int32).cuda()
        mask_clean = mask_clean < clean_num
         
        ##noisy label
        y_clean = torch.nn.functional.one_hot(target_c, args.num_classes).float().cuda()
        
        y_n = torch.nn.functional.one_hot(target, args.num_classes).float().cuda()
        _img_feat, _pt_feat = img_net_a(img_feat), pt_net_a(pt_feat)
        _img_pred, _pt_pred, _joint_pred = model_a(_img_feat, _pt_feat)
        _img_feat_b, _pt_feat_b = img_net_b(img_feat), pt_net_b(pt_feat)
        _img_pred_b, _pt_pred_b, _joint_pred_b = model_b(_img_feat_b, _pt_feat_b)
        y_joint_pred = (_joint_pred + _joint_pred_b) * 0.5
        y_joint_pred = _joint_pred
        ##linear pred label
        y_img_pred = F.softmax(_img_pred, dim=1)
        # img_pred_labels = torch.argmax(y_img_pred, dim=1)
        y_pt_pred = F.softmax(_pt_pred, dim=1)
        # pt_pred_labels = torch.argmax(y_pt_pred, dim=1)
        
        y_joint_pred = F.softmax(_joint_pred, dim=1)
        # joint_pred_labels = torch.argmax(y_joint_pred, dim=1)
        
        

        weight_mask = None

        y_hat = y_joint_pred
      

        
        ##moving-average
        if not args.yema:
            with torch.no_grad():
                y_e = y_hat.clone().detach().cpu()
                # print(y_e.shape)
                y_hat = MA_epoch.update(y_e, indices,epoch, weight = weight_mask)
                y_hat = y_hat.cuda()
        
        if args.yrectified:
            y_final_one_hot = y_n
            y_final = target
            w = None
        else:
            y_final_one_hot = y_hat.clone().detach()
            y_final_one_hot[mask_clean,:] = y_clean
            y_final = torch.argmax(y_hat, dim=1)
            w = None
       
        _pt_pred_s = _pt_pred[mask_clean]
        _img_pred_s = _img_pred[mask_clean]
        _joint_pred_s = _joint_pred[mask_clean]
        _img_feat_s = _img_feat[mask_clean]
        _pt_feat_s = _pt_feat[mask_clean]
        y_final_one_hot_s = y_final_one_hot[mask_clean]
        y_final_s = y_final[mask_clean]
        prior = torch.ones(args.num_classes)/args.num_classes
        prior = prior.cuda()

        pt_pred_mean = torch.softmax(_pt_pred_s, dim=1).mean(0)
        pt_penalty = torch.sum(prior*torch.log(prior/pt_pred_mean))
        # pt_cls_loss = -torch.mean(torch.sum(F.log_softmax(_pt_pred_s, dim=1) * y_final_one_hot_s, dim=1)) + pt_penalty
        pt_pred = torch.softmax(_pt_pred_s, dim=1)
        pt = torch.sum(pt_pred * y_final_one_hot_s, dim=1)  # shape: [batch_size]
        log_pt = torch.sum(F.log_softmax(_pt_pred_s, dim=1) * y_final_one_hot_s, dim=1)  # [batch_size]
        focal_weight_pt = (1.0 - pt)  # gamma = 1
        pt_cls_loss = -torch.mean(focal_weight_pt * log_pt)+ pt_penalty
        
        img_pred_mean = torch.softmax(_img_pred_s, dim=1).mean(0)
        img_penalty = torch.sum(prior*torch.log(prior/img_pred_mean))
        img_pred = torch.softmax(_img_pred_s, dim=1)
        img = torch.sum(img_pred * y_final_one_hot_s, dim=1)  # shape: [batch_size]
        log_img = torch.sum(F.log_softmax(_img_pred_s, dim=1) * y_final_one_hot_s, dim=1)  # [batch_size]
        focal_weight_img = (1.0 - img)  # gamma = 1
        img_cls_loss = -torch.mean(focal_weight_img * log_img)+ img_penalty
        
           
        joint_pred_mean = torch.softmax(_joint_pred_s, dim=1).mean(0)
        joint_penalty = torch.sum(prior*torch.log(prior/joint_pred_mean))
        joint_pred = torch.softmax(_joint_pred_s, dim=1)
        joint = torch.sum(joint_pred * y_final_one_hot_s, dim=1)  # shape: [batch_size]
        log_joint = torch.sum(F.log_softmax(_joint_pred_s, dim=1) * y_final_one_hot_s, dim=1)  # [batch_size]
        focal_weight_joint = (1.0 - joint)  # gamma = 1
        joint_crc_loss = -torch.mean(focal_weight_joint * log_joint) + joint_penalty
        

        cls_loss_c = (pt_cls_loss + img_cls_loss + 2*joint_crc_loss)/2
 
    
        sem_loss_c, centers = sem_criterion(torch.cat((_img_feat_s, _pt_feat_s), dim = 0), torch.cat((y_final_s, y_final_s), dim = 0),  epoch)
        
        inst_loss_c = inst_criterion(torch.cat((_img_feat_s, _pt_feat_s), dim = 0))
        
        loss_c = args.w_ce_c * cls_loss_c + args.w_sem_c * sem_loss_c + args.w_inst_c * inst_loss_c
        # loss_c = args.w_cls_c * cls_loss_c  + args.w_sem_c * sem_loss_c
        
        _pt_pred_s = _pt_pred[~mask_clean]
        _img_pred_s = _img_pred[~mask_clean]
        _joint_pred_s = _joint_pred[~mask_clean]
        _img_feat_s = _img_feat[~mask_clean]
        _pt_feat_s = _pt_feat[~mask_clean]
        y_final_one_hot_s = y_final_one_hot[~mask_clean]
        y_final_s = y_final[~mask_clean]
        
        # print("标签修改策略后的与原来的数量对比：",(y_final_s == label_ori_n.argmax(dim=1).to(y_final_s.device)).sum().item())
        # print("标签修改策略后的与原来的数量对比：",(y_final_s == label_ori_n.argmax(dim=1).to(y_final_s.device)))
        # print(label_ori_n.shape[0])
        pt_pred_mean = torch.softmax(_pt_pred_s, dim=1).mean(0)
        pt_penalty = torch.sum(prior*torch.log(prior/pt_pred_mean))
        # pt_cls_loss = -torch.mean(torch.sum(F.log_softmax(_pt_pred_s, dim=1) * y_final_one_hot_s, dim=1)) + pt_penalty
        pt_pred = torch.softmax(_pt_pred_s, dim=1)
        pt = torch.sum(pt_pred * y_final_one_hot_s, dim=1)  # shape: [batch_size]
        log_pt = torch.sum(F.log_softmax(_pt_pred_s, dim=1) * y_final_one_hot_s, dim=1)  # [batch_size]
        focal_weight_pt = (1.0 - pt)  # gamma = 1
        pt_cls_loss = -torch.mean(focal_weight_pt * log_pt)+ pt_penalty
        
        # regularization     
        img_pred_mean = torch.softmax(_img_pred_s, dim=1).mean(0)
        img_penalty = torch.sum(prior*torch.log(prior/img_pred_mean))
        img_pred = torch.softmax(_img_pred_s, dim=1)
        img = torch.sum(img_pred * y_final_one_hot_s, dim=1)  # shape: [batch_size]
        log_img = torch.sum(F.log_softmax(_img_pred_s, dim=1) * y_final_one_hot_s, dim=1)  # [batch_size]
        focal_weight_img = (1.0 - img)  # gamma = 1
        img_cls_loss = -torch.mean(focal_weight_img * log_img)+ img_penalty
        
        # regularization     
        joint_pred_mean = torch.softmax(_joint_pred_s, dim=1).mean(0)
        joint_penalty = torch.sum(prior*torch.log(prior/joint_pred_mean))
        joint_pred = torch.softmax(_joint_pred_s, dim=1)
        joint = torch.sum(joint_pred * y_final_one_hot_s, dim=1)  # shape: [batch_size]
        log_joint = torch.sum(F.log_softmax(_joint_pred_s, dim=1) * y_final_one_hot_s, dim=1)  # [batch_size]
        focal_weight_joint = (1.0 - joint)  # gamma = 1
        joint_crc_loss = -torch.mean(focal_weight_joint * log_joint)+ joint_penalty
        

        cls_loss_n = (pt_cls_loss + img_cls_loss + 2*joint_crc_loss)/2
        sem_loss_n, centers = sem_criterion(torch.cat((_img_feat_s, _pt_feat_s), dim = 0), torch.cat((y_final_s, y_final_s), dim = 0),  epoch)
        
        inst_loss_n = inst_criterion(torch.cat((_img_feat_s, _pt_feat_s), dim = 0))
        
        loss_n = args.w_ce_n * cls_loss_n + args.w_sem_n * sem_loss_n + args.w_inst_n * inst_loss_n
        # loss_n =   args.w_inst_n * inst_loss_n + args.w_sem_n * sem_loss_n
        
        # loss = loss_c + args.wn*loss_n
        loss = loss_c 
        
        torch.autograd.set_detect_anomaly(True)
        loss.backward()

        optimizer_pt.step()
        optimizer_img.step()
        # optimizer_mesh.step()
        optimizer_model.step()

        optimizer_cmc.step()
        
        img_acc = calculate_accuracy(y_img_pred, target_ori)
        pt_acc = calculate_accuracy(y_pt_pred, target_ori)
  
        joint_pred = (y_img_pred+y_pt_pred)/2
        joint_acc = calculate_accuracy(joint_pred, target_ori)
        joint_m_acc = calculate_accuracy(y_joint_pred, target_ori)
        
        ema_acc = calculate_accuracy(y_final_one_hot, target_ori)
        

        ema_acc_n = calculate_accuracy(y_final_one_hot[~mask_clean], target_ori[~mask_clean])

        # wandb.log({
        #     "enire_Acc":ema_acc,
        #     "correct_acc":ema_acc_n,
        #     "epoch":epoch
        # })
    
        if iteration % args.per_print == 0:
            print("[%d]  img_acc: %.4f  pt_acc: %.4f  joint_acc: %.4f jm_acc: %.4f ema_acc: %.4f ema_acc_n: %f" % (iteration, img_acc, pt_acc, joint_acc, joint_m_acc, ema_acc, ema_acc_n)) 
            start_time = time.time()
        iteration = iteration + 1
  
        

    return iteration

