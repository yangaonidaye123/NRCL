from __future__ import division, absolute_import
import torch
from torch.autograd import Variable
import time
import warnings
from losses.YA_loss import YangaoLoss
warnings.filterwarnings('ignore',category=FutureWarning)

class NegEntropy(object):
    def __call__(self,outputs):
        probs = torch.softmax(outputs, dim=1)
        return torch.mean(torch.sum(probs.log()*probs, dim=1))
def warmup(args,
             epoch,
             img_net, 
             pt_net,
             model, 
             train_trainloader,
             optimizer_img, 
             optimizer_pt, 
             optimizer_model,
             optimizer_centloss, 
             ce, 
             inst_criterion, 
             sem_criterion):

    pt_net.train(True)
    model.train(True)
    img_net.train(True)
    # mesh_net.train(True)
    
    conf_penalty = NegEntropy()
    
    
    num_samples = len(train_trainloader.dataset)
    costs = torch.zeros(num_samples)
    iteration = epoch*len(train_trainloader)
    iteration_all = args.epochs*len(train_trainloader)
    start_time = time.time()
    
    for data in train_trainloader:

        img_list, pt_feat, target, target_ori, indices = data
        img_feat, img_feat_v = img_list[0], img_list[1] 
        img_feat = Variable(img_feat).to(torch.float32).to('cuda')
        img_feat_v = Variable(img_feat_v).to(torch.float32).to('cuda')
        pt_feat = Variable(pt_feat).to(torch.float32).to('cuda')
        target = Variable(target).to(torch.long).to('cuda')

        optimizer_img.zero_grad()
        optimizer_pt.zero_grad()
        optimizer_model.zero_grad()
        optimizer_centloss.zero_grad()
        
        _img_feat, _pt_feat = img_net(img_feat, img_feat_v), pt_net(pt_feat)
        _img_pred, _pt_pred, _joint_pred = model(_img_feat, _pt_feat)

        cost = None
        YA = YangaoLoss(operation='none')
        cost_img = YA(_img_pred,target)
        cost_pt = YA(_pt_pred,target)
        cost_joint = YA(_joint_pred,target)
        cost = cost_joint + (cost_img + cost_pt)/2.0
        indices = indices.to(torch.long)
        costs[indices] = cost.cpu().squeeze(1)

        pt_ce_loss = ce(_pt_pred, target)
        img_ce_loss = ce(_img_pred, target)
        joint_ce_loss = ce(_joint_pred, target)
        

        pt_penalty = conf_penalty(_img_pred)
        img_penalty = conf_penalty(_pt_pred)
        joint_penalty = conf_penalty(_joint_pred)
        
        penalty = pt_penalty+img_penalty+joint_penalty
        

        ce_loss = (pt_ce_loss + img_ce_loss+2*joint_ce_loss)/2
        

        sem_loss, centers = sem_criterion(torch.cat((_img_feat, _pt_feat), dim = 0), torch.cat((target, target), dim = 0), epoch)
        
        inst_loss = inst_criterion(torch.cat((_img_feat, _pt_feat), dim = 0))
        
        if args.noise_type=="asy":
            loss = args.w_ce_c * (ce_loss+penalty) + args.w_sem_c * sem_loss + args.w_inst_c * inst_loss
        else:
            loss = args.w_ce_c * ce_loss + args.w_sem_c * sem_loss + args.w_inst_c * inst_loss
            # loss =  args.w_sem_c * sem_loss
            # loss = args.w_cls_c * cls_loss +args.w_inst_c * inst_loss
        
        
        

        loss.backward()

        # optimizer_head.step()
        optimizer_img.step()
        optimizer_pt.step()
        optimizer_model.step()

        optimizer_centloss.step()

        if iteration % args.per_print == 0:
            print("[%d/%d]  loss: %f  sem_loss: %f  ce_loss: %f  inst_loss: %f time: %f" % (iteration, iteration_all, loss.item(), sem_loss.item(), ce_loss.item(), inst_loss.item(), time.time()-start_time))
            start_time = time.time()
            

        iteration = iteration + 1
        
    return iteration

def warmup_imgtxt(args,
             epoch,
             img_net, 
             pt_net,
             model, 
             train_trainloader,
             input_data_attr,
             optimizer_img, 
             optimizer_pt, 
             optimizer_model,
             optimizer_centloss, 
             ce, 
             inst_criterion, 
             sem_criterion):

    pt_net.train(True)
    model.train(True)
    img_net.train(True)
    # mesh_net.train(True)
    
    conf_penalty = NegEntropy()
    
    num_samples = input_data_attr['num_train']
    costs = torch.zeros(num_samples)
    iteration = epoch*len(train_trainloader)
    iteration_all = args.epochs*len(train_trainloader)
    start_time = time.time()
    
    for data in train_trainloader:
        # image, point cloud, noisy labels, original labels (True labels for val.).
        img, txt, labels, labels_ori, indices = data
        img_feat = Variable(img).to(torch.float32).to('cuda')
        
        
        pt_feat = Variable(txt).to(torch.float32).to('cuda')
        # pt_feat = pt_feat.permute(0,2,1)
        target = labels.argmax(dim=1)

        target = Variable(target).to(torch.long).to('cuda')

        optimizer_img.zero_grad()
        optimizer_pt.zero_grad()
        # optimizer_mesh.zero_grad()
        optimizer_model.zero_grad()
        optimizer_centloss.zero_grad()
        
        _img_feat, _pt_feat = img_net(img_feat), pt_net(pt_feat)
        # _mesh_feat = mesh_net(centers, corners, normals, neighbor_index)
        _img_pred, _pt_pred, _joint_pred = model(_img_feat, _pt_feat)


        cost = None
        YA = YangaoLoss(operation='none')
        cost_img = YA(_img_pred,target)
        cost_pt = YA(_pt_pred,target)
        cost_joint = YA(_joint_pred,target)
            
        cost = cost_joint + (cost_img + cost_pt)/2.0
            
        indices = indices.to(torch.long)
        costs[indices] = cost.cpu().squeeze(1)
        pt_ce_loss = ce(_pt_pred, target)
        img_ce_loss = ce(_img_pred, target)
        joint_ce_loss = ce(_joint_pred, target)
        
        # mesh_ce_loss = ce_criterion(_mesh_pred, target)
        pt_penalty = conf_penalty(_img_pred)
        img_penalty = conf_penalty(_pt_pred)
        joint_penalty = conf_penalty(_joint_pred)
        
        
        penalty = pt_penalty+img_penalty+joint_penalty
        

        ce_loss = (pt_ce_loss + img_ce_loss+2*joint_ce_loss)/2
        

        sem_loss, centers = sem_criterion(torch.cat((_img_feat, _pt_feat), dim = 0), torch.cat((target, target), dim = 0), epoch)
        
        inst_loss = inst_criterion(torch.cat((_img_feat, _pt_feat), dim = 0))
        
        if args.noise_type=="asy":
            loss = args.w_ce_c * (ce_loss+penalty) + args.w_sem_c * sem_loss + args.w_inst_c * inst_loss
        else:
            loss = args.w_ce_c * ce_loss + args.w_sem_c * sem_loss + args.w_inst_c * inst_loss
            # loss =  args.w_sem_c * sem_loss
            # loss = args.w_cls_c * cls_loss +args.w_inst_c * inst_loss
        
        
        

        loss.backward()

        # optimizer_head.step()
        optimizer_img.step()
        optimizer_pt.step()
        # optimizer_mesh.step()
        optimizer_model.step()

        optimizer_centloss.step()

        if iteration % args.per_print == 0:
            print("[%d/%d]  loss: %f  sem_loss: %f  ce_loss: %f  inst_loss: %f time: %f" % (iteration, iteration_all, loss.item(), sem_loss.item(), ce_loss.item(), inst_loss.item(), time.time()-start_time))
            start_time = time.time()
            

        iteration = iteration + 1
        
    return iteration

