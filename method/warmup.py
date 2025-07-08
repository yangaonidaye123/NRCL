from __future__ import division, absolute_import
import torch
from torch.autograd import Variable
import time
import wandb
import warnings
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
             cls_criterion, 
             inst_criterion, 
             sem_criterion):

    pt_net.train(True)
    model.train(True)
    img_net.train(True)
    # mesh_net.train(True)
    conf_penalty = NegEntropy()
    
    iteration = epoch*len(train_trainloader)
    iteration_all = args.epochs*len(train_trainloader)
    start_time = time.time()

    for data in train_trainloader:
        # img_feat, pt_feat, target, ori_label, index = data
        img_feat, pt_feat, target,label,   index = data
        # image, point cloud, noisy labels, original labels (True labels for val.).
        img_feat = Variable(img_feat).to(torch.float32).to('cuda')
        pt_feat = Variable(pt_feat).to(torch.float32).to('cuda')
        target = Variable(target).to(torch.long).to('cuda')
        label = Variable(label).to(torch.long).to('cuda')
        
        optimizer_img.zero_grad()
        optimizer_pt.zero_grad()
        # optimizer_mesh.zero_grad()
        optimizer_model.zero_grad()
        optimizer_centloss.zero_grad()
        
        _img_feat = img_net(img_feat)
        _pt_feat = pt_net(pt_feat)
        
        # _mesh_feat = mesh_net(centers, corners, normals, neighbor_index)
        _img_pred, _pt_pred, _joint_pred = model(_img_feat, _pt_feat)
        # _img_pred, _pt_pred, _vis_img_feat, _vis_pt_feat = model(_img_feat, _pt_feat)
        # print(_joint_pred.shape)
        # compute loss
        pt_cls_loss = cls_criterion(_pt_pred, target)
        img_cls_loss = cls_criterion(_img_pred, target)
        joint_cls_loss = cls_criterion(_joint_pred, target)
        
        # mesh_ce_loss = ce_criterion(_mesh_pred, target)
        pt_penalty = conf_penalty(_img_pred)
        img_penalty = conf_penalty(_pt_pred)
        joint_penalty = conf_penalty(_joint_pred)
        
        
        penalty = pt_penalty+img_penalty+joint_penalty
        # penalty = pt_penalty+img_penalty

        cls_loss = (pt_cls_loss + img_cls_loss + 2 * joint_cls_loss)/2


        sem_loss, centers = sem_criterion(torch.cat((_img_feat, _pt_feat), dim = 0), torch.cat((target, target), dim = 0), epoch)
        
        inst_loss = inst_criterion(torch.cat((_img_feat, _pt_feat), dim = 0))
        
        if args.noise_type=="asy":
            loss = args.w_cls_c * (cls_loss+penalty) + args.w_sem_c * sem_loss + args.w_inst_c * inst_loss
        else:
            loss = args.w_cls_c * cls_loss + args.w_sem_c * sem_loss + args.w_inst_c * inst_loss
        
        
        

        loss.backward()

        optimizer_img.step()
        optimizer_pt.step()
        optimizer_model.step()

        optimizer_centloss.step()


        if iteration % args.per_print == 0:
            print("[%d/%d]  loss: %f  sem_loss: %f  cls_loss: %f  inst_loss: %f time: %f" % (iteration, iteration_all, loss.item(), sem_loss.item(), cls_loss.item(), inst_loss.item(), time.time()-start_time))
            start_time = time.time()

        wandb.log({
                     "iteration":iteration,
                     "train_loss": loss.item(),
                     "center_loss": sem_loss.item(),
                     "cls_loss": cls_loss.item(),
                     "mg_loss": inst_loss.item()
                    })
            

        iteration = iteration + 1
        
    return iteration