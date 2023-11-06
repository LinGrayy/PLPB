from datetime import datetime
import os
import os.path as osp
import timeit
from torchvision.utils import make_grid
import time

import numpy as np
import pytz
import torch
from torch import nn
import torch.nn.functional as F

from tensorboardX import SummaryWriter

import tqdm
import socket
from metrics import *
from utils.Utils import *
import scipy.optimize as sopt

bceloss = torch.nn.BCELoss()
mseloss = torch.nn.MSELoss()

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']
        
def causal_structure_learning(X, lambda1, loss_type='l2', max_iter=100, h_tol=1e-8, rho_max=1e+16, w_threshold=0.3):
    # Solve min_W L(W; X)  s.t. h(W) = 0 using augmented Lagrangian.
    # Args:
    #     X (np.ndarray): [n, d] sample matrix
    #     lambda1 (float): l1 penalty parameter
    #     loss_type (str): l2, logistic, poisson
    #     max_iter (int): max num of dual ascent steps
    #     h_tol (float): exit if |h(w_est)| <= htol
    #     rho_max (float): exit if rho >= rho_max
    #     w_threshold (float): drop edge if |weight| < threshold
    # Returns:
    #     W_est (np.ndarray): [d, d] estimated DAG
    # 
        def _loss(W):
            """Evaluate value and gradient of loss."""
            M = X @ W
            print(X.shape, M.shape)
            if loss_type == 'l2':
                R = X - M
                loss = 0.5 / X.shape[2] * (R ** 2).sum()
                G_loss = - 1.0 / X.shape[2] * X.T @ R
            elif loss_type == 'logistic':
                loss = 1.0 / X.shape[2] * (np.logaddexp(0, M) - X * M).sum()
                G_loss = 1.0 / X.shape[20] * X.T @ (torch.sigmoid(M) - X)
            elif loss_type == 'poisson':
                S = np.exp(M)
                loss = 1.0 / X.shape[2] * (S - X * M).sum()
                G_loss = 1.0 / X.shape[2] * X.T @ (S - X)
            else:
                raise ValueError('unknown loss type')
            return loss, G_loss

        def _h(W):
            """Evaluate value and gradient of acyclicity constraint."""
            #     E = slin.expm(W * W)
            #     h = np.trace(E) - d
            M = np.eye(d) + W * W / d
            E = np.linalg.matrix_power(M, d - 1)
            h = (E.T * M).sum() - d
            G_h = E.T * W * 2
            return h, G_h

        def _adj(w):
            """Convert doubled variables ([2 d^2] array) back to original variables ([d, d] matrix)."""
            return (w[:d * d] - w[d * d:]).reshape([d, d])

        def _func(w):
            """Evaluate value and gradient of augmented Lagrangian for doubled variables ([2 d^2] array)."""
            W = _adj(w)
            loss, G_loss = _loss(W)
            h, G_h = _h(W)
            obj = loss + 0.5 * rho * h * h + alpha * h + lambda1 * w.sum()
            G_smooth = G_loss + (rho * h + alpha) * G_h
            g_obj = np.concatenate((G_smooth + lambda1, - G_smooth + lambda1), axis=None)
            return obj, g_obj

        print(X.shape)
        n, _,d,_ = X.shape
        w_est, rho, alpha, h = np.zeros(2 * d * d), 1.0, 0.0, np.inf  # double w_est into (w_pos, w_neg)
        bnds = [(0, 0) if i == j else (0, None) for _ in range(2) for i in range(d) for j in range(d)]
        for iter_j in range(max_iter):
            w_new, h_new = None, None
            print(iter_j)
            while rho < rho_max:
                sol = sopt.minimize(_func, w_est, method='L-BFGS-B', jac=True, bounds=bnds)
                w_new = sol.x
                h_new, _ = _h(_adj(w_new))
                if h_new > 0.25 * h:
                    rho *= 10
                else:
                    break
            w_est, h = w_new, h_new
            alpha += rho * h
            if h <= h_tol or rho >= rho_max:
                break
        W_est = _adj(w_est)
        # print(W_est)
        W_est[np.abs(W_est) < w_threshold] = 0
        # print(W_est)
        return W_est, h

class Trainer(object):

    def __init__(self, cuda, model_gen, model_dis, model_uncertainty_dis, optimizer_gen, optimizer_dis, optimizer_uncertainty_dis,
                 val_loader, domain_loaderS, domain_loaderT, out, max_epoch, stop_epoch=None,
                 lr_gen=1e-3, lr_dis=1e-3, lr_decrease_rate=0.1, interval_validate=None, batch_size=8, warmup_epoch=10):
        self.cuda = cuda
        self.warmup_epoch = warmup_epoch
        self.model_gen = model_gen
        self.model_dis2 = model_uncertainty_dis
        self.model_dis = model_dis
        self.optim_gen = optimizer_gen
        self.optim_dis = optimizer_dis
        self.optim_dis2 = optimizer_uncertainty_dis
        self.lr_gen = lr_gen
        self.lr_dis = lr_dis
        self.lr_decrease_rate = lr_decrease_rate
        self.batch_size = batch_size

        self.val_loader = val_loader
        self.domain_loaderS = domain_loaderS
        self.domain_loaderT = domain_loaderT
        self.time_zone = 'Asia/Hong_Kong'
        self.timestamp_start = \
            datetime.now(pytz.timezone(self.time_zone))

        if interval_validate is None:
            self.interval_validate = int(10)
        else:
            self.interval_validate = interval_validate

        self.out = out
        if not osp.exists(self.out):
            os.makedirs(self.out)

        self.log_headers = [
            'epoch',
            'iteration',
            'train/loss_seg',
            'train/cup_dice',
            'train/disc_dice',
            'train/loss_adv',
            'train/loss_D_same',
            'train/loss_D_diff',
            'valid/loss_CE',
            'valid/cup_dice',
            'valid/disc_dice',
            'elapsed_time',
        ]
        if not osp.exists(osp.join(self.out, 'log.csv')):
            with open(osp.join(self.out, 'log.csv'), 'w') as f:
                f.write(','.join(self.log_headers) + '\n')

        log_dir = os.path.join(self.out, 'tensorboard',
                               datetime.now().strftime('%b%d_%H-%M-%S') + '_' + socket.gethostname())
        self.writer = SummaryWriter(log_dir=log_dir)

        self.epoch = 0
        self.iteration = 0
        self.max_epoch = max_epoch
        self.stop_epoch = stop_epoch if stop_epoch is not None else max_epoch
        self.best_disc_dice = 0.0
        self.running_loss_tr = 0.0
        self.running_adv_diff_loss = 0.0
        self.running_adv_same_loss = 0.0
        self.best_mean_dice = 0.0
        self.best_epoch = -1


    def validate(self):
        training = self.model_gen.training
        self.model_gen.eval()

        val_loss = 0.0
        val_cup_dice = 0.0
        val_disc_dice = 0.0
        datanum_cnt = 0.0
        metrics = []
        with torch.no_grad():

            for batch_idx, sample in tqdm.tqdm(
                    enumerate(self.val_loader), total=len(self.val_loader),
                    desc='Valid iteration=%d' % self.iteration, ncols=80,
                    leave=False):
                data = sample['image']
                target_map = sample['map']
                target_boundary = sample['boundary']

                #data, target_map, aimg_source, target_boundary, dlabel_source = sample
                
                #data = torch.cat([data, aimg_source, data], dim=1)
                if self.cuda:
                    # data, target_map, aimg_source, target_boundary = data.cuda(), target_map.cuda(), aimg_source.cuda(), target_boundary.cuda()
                    data, target_map, target_boundary = data.cuda(), target_map.cuda(), target_boundary.cuda()
                    predictions, boundary, _ = self.model_gen(data)
                with torch.no_grad():
                    predictions, boundary, _ = self.model_gen(data)

                #loss = F.binary_cross_entropy_with_logits(predictions[:,0,:,:], target_map.to(torch.float))
                loss = F.binary_cross_entropy_with_logits(predictions, target_map)
            
                loss_data = loss.data.item()
                if np.isnan(loss_data):
                    raise ValueError('loss is nan while validating')
                val_loss += loss_data

                #dice_cup = dice_coeff(predictions[:,0,:,:], target_map)#dice_coeff_2label
                dice_cup = dice_coeff_2label(predictions, target_map)
                dice_disc = dice_coeff_2label(predictions, target_map)#
                # print('dice: ', dice_cup)
                val_cup_dice += np.sum(dice_cup)
                val_disc_dice += np.sum(dice_disc)
                datanum_cnt += float(dice_cup.shape[0])
            val_loss /= datanum_cnt
            val_cup_dice /= datanum_cnt
            val_disc_dice /= datanum_cnt
            metrics.append((val_loss, val_cup_dice))#val_disc_dice
            self.writer.add_scalar('val_data/loss_CE', val_loss, self.epoch * (len(self.domain_loaderS)))
            self.writer.add_scalar('val_data/val_CUP_dice', val_cup_dice, self.epoch * (len(self.domain_loaderS)))
            self.writer.add_scalar('val_data/val_DISC_dice', val_disc_dice, self.epoch * (len(self.domain_loaderS)))

            mean_dice = val_cup_dice #+ val_disc_dice
            is_best = mean_dice > self.best_mean_dice
            if is_best:
                self.best_epoch = self.epoch + 1
                self.best_mean_dice = mean_dice

                torch.save({
                    'epoch': self.epoch,
                    'iteration': self.iteration,
                    'arch': self.model_gen.__class__.__name__,
                    'optim_state_dict': self.optim_gen.state_dict(),
                    'optim_dis_state_dict': self.optim_dis.state_dict(),
                    'optim_dis2_state_dict': self.optim_dis2.state_dict(),
                    'model_state_dict': self.model_gen.state_dict(),
                    'model_dis_state_dict': self.model_dis.state_dict(),
                    'model_dis2_state_dict': self.model_dis2.state_dict(),
                    'learning_rate_gen': get_lr(self.optim_gen),
                    'learning_rate_dis': get_lr(self.optim_dis),
                    'learning_rate_dis2': get_lr(self.optim_dis2),
                    'best_mean_dice': self.best_mean_dice,
                }, osp.join(self.out, 'checkpoint_%d.pth.tar' % self.best_epoch))
            else:
                if (self.epoch + 1) >150 :#% 10 == 0:
                    torch.save({
                        'epoch': self.epoch,
                    'iteration': self.iteration,
                    'arch': self.model_gen.__class__.__name__,
                    'optim_state_dict': self.optim_gen.state_dict(),
                    'optim_dis_state_dict': self.optim_dis.state_dict(),
                    'optim_dis2_state_dict': self.optim_dis2.state_dict(),
                    'model_state_dict': self.model_gen.state_dict(),
                    'model_dis_state_dict': self.model_dis.state_dict(),
                    'model_dis2_state_dict': self.model_dis2.state_dict(),
                    'learning_rate_gen': get_lr(self.optim_gen),
                    'learning_rate_dis': get_lr(self.optim_dis),
                    'learning_rate_dis2': get_lr(self.optim_dis2),
                    'best_mean_dice': self.best_mean_dice,
                    }, osp.join(self.out, 'checkpoint_%d.pth.tar' % (self.epoch + 1)))


            with open(osp.join(self.out, 'log.csv'), 'a') as f:
                elapsed_time = (
                    datetime.now(pytz.timezone(self.time_zone)) -
                    self.timestamp_start).total_seconds()
                log = [self.epoch, self.iteration] + [''] * 5 + \
                       list(metrics) + [elapsed_time] + ['best model epoch: %d' % self.best_epoch]
                log = map(str, log)
                f.write(','.join(log) + '\n')
            self.writer.add_scalar('best_model_epoch', self.best_epoch, self.epoch * (len(self.domain_loaderS)))
            if training:
                self.model_gen.train()
                self.model_dis.train()
                self.model_dis2.train()



    def train_epoch(self):
        source_domain_label = 1
        target_domain_label = 0
        smooth = 1e-7
        self.model_gen.train()
        self.model_dis.train()
        self.model_dis2.train()
        self.running_seg_loss = 0.0
        self.running_adv_loss = 0.0
        self.running_dis_diff_loss = 0.0
        self.running_dis_same_loss = 0.0
        self.running_total_loss = 0.0
        self.running_cup_dice_tr = 0.0
        self.running_disc_dice_tr = 0.0
        loss_adv_diff_data = 0
        loss_D_same_data = 0
        loss_D_diff_data = 0

        domain_t_loader = enumerate(self.domain_loaderT)
        start_time = timeit.default_timer()
        for batch_idx, sampleS in tqdm.tqdm(
                enumerate(self.domain_loaderS), total=len(self.domain_loaderS),
                desc='Train epoch=%d' % self.epoch, ncols=80, leave=False):

            metrics = []

            iteration = batch_idx + self.epoch * len(self.domain_loaderS)
            self.iteration = iteration

            assert self.model_gen.training
            assert self.model_dis.training
            assert self.model_dis2.training

            self.optim_gen.zero_grad()
            self.optim_dis.zero_grad()
            self.optim_dis2.zero_grad()

            # 1. train generator with random images
            for param in self.model_dis.parameters():
                param.requires_grad = False
            for param in self.model_dis2.parameters():
                param.requires_grad = False
            for param in self.model_gen.parameters():
                param.requires_grad = True
            
            # device = torch.device('cuda:0')
            # cimg_source, target_map, aimg_source, target_boundary, dlabel_source = sampleS
            # cimg_source = cimg_source.to(device)
            # target_map = target_map.to(device)
            # aimg_source = aimg_source.to(device)
            # target_boundary = target_boundary.to(device)
            # dlabel_source = dlabel_source.to(device)
            # img_cat = torch.cat([cimg_source, aimg_source, cimg_source], dim=1)
            
            #oS, boundaryS, dpred_source = self.model_gen(img_cat)

            imageS = sampleS['image'].cuda()
            target_map = sampleS['map'].cuda()
            target_boundary = sampleS['boundary'].cuda()

            oS, boundaryS, feature = self.model_gen(imageS)

            loss_seg1 = bceloss(torch.sigmoid(oS), target_map)
            loss_seg2 = mseloss(torch.sigmoid(boundaryS), target_boundary)

            # loss_seg1 = bceloss(torch.sigmoid(oS)[:,0,:,:], target_map.to(torch.float))
            # loss_seg2 = mseloss(torch.sigmoid(boundaryS)[:,0,:,:], target_boundary.to(torch.float))

            #interp = nn.Upsample(size=(512,512), mode='bilinear',align_corners=True)
            #feature = interp(feature)
            #W_est, global_loss = causal_structure_learning(np.hstack((feature.data.cpu(), target_map.data.cpu())), lambda1=0)
            # loss_seg = loss_seg1 + loss_seg2
            loss_seg = loss_seg1 +loss_seg2#+ global_loss

            self.running_seg_loss += loss_seg.item()
            loss_seg_data = loss_seg.data.item()
            if np.isnan(loss_seg_data):
                raise ValueError('loss is nan while training')

            #cup_dice, disc_dice = dice_coeff_2label(oS, target_map)
            
            loss_seg.backward()
            self.optim_gen.step()

            # write image log
            if iteration % 30 == 0:
                grid_image = make_grid(
                    imageS[0, ...].clone().cpu().data, 1, normalize=True)
                self.writer.add_image('DomainS/image', grid_image, iteration)
                grid_image = make_grid(
                    target_map[0, 0, ...].clone().cpu().data, 1, normalize=True)
                self.writer.add_image('DomainS/target_cup', grid_image, iteration)
                # grid_image = make_grid(
                #     target_map[0, 1, ...].clone().cpu().data, 1, normalize=True)
                self.writer.add_image('DomainS/target_disc', grid_image, iteration)
                grid_image = make_grid(
                    target_boundary[0, 0, ...].clone().cpu().data, 1, normalize=True)
                self.writer.add_image('DomainS/target_boundary', grid_image, iteration)
                grid_image = make_grid(torch.sigmoid(oS)[0, 0, ...].clone().cpu().data, 1, normalize=True)
                self.writer.add_image('DomainS/prediction_cup', grid_image, iteration)
                # grid_image = make_grid(torch.sigmoid(oS)[0, 1, ...].clone().cpu().data, 1, normalize=True)
                # self.writer.add_image('DomainS/prediction_disc', grid_image, iteration)
                grid_image = make_grid(torch.sigmoid(boundaryS)[0, 0, ...].clone().cpu().data, 1, normalize=True)
                self.writer.add_image('DomainS/prediction_boundary', grid_image, iteration)

            self.writer.add_scalar('train_gen/loss_seg', loss_seg_data, iteration)

            metrics.append((loss_seg_data, loss_adv_diff_data, loss_D_same_data, loss_D_diff_data))
            metrics = np.mean(metrics, axis=0)

            with open(osp.join(self.out, 'log.csv'), 'a') as f:
                elapsed_time = (
                    datetime.now(pytz.timezone(self.time_zone)) -
                    self.timestamp_start).total_seconds()
                log = [self.epoch, self.iteration]  + \
                    metrics.tolist() + [''] * 5 + [elapsed_time]
                log = map(str, log)
                f.write(','.join(log) + '\n')

        self.running_seg_loss /= len(self.domain_loaderS)
        self.running_adv_diff_loss /= len(self.domain_loaderS)
        self.running_dis_same_loss /= len(self.domain_loaderS)
        self.running_dis_diff_loss /= len(self.domain_loaderS)

        stop_time = timeit.default_timer()

        print('\n[Epoch: %d] lr:%f,  Average segLoss: %f, '
              ' Average advLoss: %f, Average dis_same_Loss: %f, '
              'Average dis_diff_Lyoss: %f,'
              'Execution time: %.5f' %
              (self.epoch, get_lr(self.optim_gen), self.running_seg_loss,
               self.running_adv_diff_loss,
               self.running_dis_same_loss, self.running_dis_diff_loss, stop_time - start_time))

    
    def train(self):
        for epoch in tqdm.trange(self.epoch, self.max_epoch,
                                 desc='Train', ncols=80):
            self.epoch = epoch
            self.train_epoch()
            if self.stop_epoch == self.epoch:
                print('Stop epoch at %d' % self.stop_epoch)
                break

            if (epoch+1) % 100 == 0:
                _lr_gen = self.lr_gen * 0.2
                for param_group in self.optim_gen.param_groups:
                    param_group['lr'] = _lr_gen
            self.writer.add_scalar('lr_gen', get_lr(self.optim_gen), self.epoch * (len(self.domain_loaderS)))
            # if (self.epoch+1) % self.interval_validate == 0:
            if (self.epoch + 1) % 5 == 0:
                self.validate()
        self.writer.close()



