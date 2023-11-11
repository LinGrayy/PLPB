
#!/usr/bin/env python
import argparse
import os
import os.path as osp
import torch.nn.functional as F

import matplotlib.pyplot as plt
from PIL import Image
import torch
from torch.autograd import Variable
import tqdm
from advent import dataset
from dataloaders import fundus_dataloader as DL
from torch.utils.data import DataLoader
from dataloaders import custom_transforms as tr
from torchvision import transforms
from matplotlib.pyplot import imsave
from utils.Utils import *
from metrics import *
from datetime import datetime
import pytz
import networks.deeplabv3 as netd
import networks.deeplabv3_eval as netd_eval
import cv2
import torch.backends.cudnn as cudnn
import random
from tensorboardX import SummaryWriter
import torch.nn as nn
from Lovaszloss import lovasz_hinge
import imgaug.augmenters as iaa
import torchattacks
from scipy.ndimage import distance_transform_edt
bceloss = torch.nn.BCELoss(reduction='none')

def entropy_loss(v):
    """
        Entropy loss for probabilistic prediction vectors
        input: batch_size x channels x h x w
        output: batch_size x 1 x h x w
    """
    assert v.dim() == 4
    n, c, h, w = v.size()
    return -torch.sum(torch.mul(v, torch.log2(v + 1e-30))) / (n * h * w * np.log2(c))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-file', type=str, default='./logs/Domain1/1/checkpoint_180.pth.tar')
    parser.add_argument('--dataset', type=str, default='Domain1')
    parser.add_argument('--dataset_open', type=str, default='Domain4') #open domain
    parser.add_argument('--source', type=str, default='Domain3')
    parser.add_argument('-g', '--gpu', type=int, default=0)
    parser.add_argument('--data-dir', default='/mnt/data1/llr_data/Fundus/')
    parser.add_argument('--out-stride',type=int,default=16)
    parser.add_argument('--sync-bn',type=bool,default=True)
    parser.add_argument('--freeze-bn',type=bool,default=False)
    parser.add_argument(
        '--save-root-ent',
        type=str,
        default='./results/ent/',
        help='path to save ent',
    )
    parser.add_argument(
        '--save-root-mask',
        type=str,
        default='./results/mask/',
        help='path to save mask',
    )
    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    model_file = args.model_file

    # 1. dataset
    composed_transforms_train = transforms.Compose([
        tr.Resize(512),
        #tr.RandomFlip(),
        tr.add_salt_pepper_noise(),
        tr.adjust_light(),
        tr.eraser(),
        tr.Normalize_tf(),
        tr.ToTensor()
    ])
    composed_transforms_test = transforms.Compose([
        tr.Resize(512),
        #tr.RandomFlip(),
        tr.Normalize_tf(),
        tr.ToTensor()
    ])
    composed_transforms_test1 = transforms.Compose([
        tr.Resize(512),
        #tr.RandomFlip(),
        tr.Normalize_tf(),
        tr.ToTensor()
    ])
    db_train = DL.FundusSegmentation(base_dir=args.data_dir, dataset=args.dataset, split='train/ROIs', transform=composed_transforms_train)
    db_test = DL.FundusSegmentation(base_dir=args.data_dir, dataset=args.dataset, split='test/ROIs', transform=composed_transforms_test)
    db_source = DL.FundusSegmentation(base_dir=args.data_dir, dataset=args.source, split='train/ROIs', transform=composed_transforms_test)
    db_open = DL.FundusSegmentation(base_dir=args.data_dir, dataset=args.dataset_open, split='test/ROIs', transform=composed_transforms_test)
    
    train_loader = DataLoader(db_train, batch_size=4, shuffle=False, num_workers=1)
    test_loader = DataLoader(db_test, batch_size=1, shuffle=False, num_workers=1)
    open_loader = DataLoader(db_open, batch_size=1, shuffle=False, num_workers=1)
    source_loader = DataLoader(db_source, batch_size=1, shuffle=False, num_workers=1)

    # 2. model
    model = netd.DeepLab(num_classes=2, backbone='mobilenet', output_stride=args.out_stride, sync_bn=args.sync_bn, freeze_bn=args.freeze_bn)
    model_eval = netd_eval.DeepLab(num_classes=2, backbone='mobilenet', output_stride=args.out_stride, sync_bn=args.sync_bn, freeze_bn=args.freeze_bn).cuda()

    if torch.cuda.is_available():
        model = model.cuda()
    print('==> Loading %s model file: %s' %
          (model.__class__.__name__, model_file))
    checkpoint = torch.load(model_file)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.train()

    if args.dataset=="Domain2":
        npfilename = './results/mask/D2.npz'
        npfilename1 = './results/bound/bound_D2.npz'
    elif args.dataset=="Domain1":
        npfilename = './results/mask/D1.npz'
        npfilename1 = './results/bound/r-bound_D1.npz'
    elif args.dataset=="Domain4": # open domain
        npfilename = './results/mask/D1.npz'
        npfilename1 = './results/bound/bound_D1.npz'

    npdata = np.load(npfilename, allow_pickle=True)
    pseudo_label_dic = npdata['arr_0'].item()
    uncertain_dic = npdata['arr_1'].item()
    proto_pseudo_dic = npdata['arr_2'].item()
    
    npdata1 = np.load(npfilename1, allow_pickle=True)
    pseudo_bound_dic = npdata1['arr_0'].item()

    var_list = model.named_parameters()

    optim_gen = torch.optim.Adam(model.parameters(), lr=0.002, betas=(0.9, 0.99))
    best_val_cup_dice = 0.0;
    best_val_disc_dice = 0.0;
    best_avg = 0.0

    iter_num = 0
    for epoch_num in tqdm.tqdm(range(2), ncols=70):
        model.train()
        for batch_idx, (sample) in enumerate(train_loader):
            data, target, img_name = sample['image'], sample['map'], sample['img_name']
            target_boundary = sample['boundary']
            if torch.cuda.is_available():
                data, target, target_boundary = data.cuda(), target.cuda(), target_boundary.cuda()
            data, target, target_boundary = Variable(data), Variable(target), Variable(target_boundary)
            prediction, boundaryS, feature = model(data)
            
            num_classes = 2
            pred_s = prediction.permute(0, 2, 3, 1).contiguous().view(-1, num_classes)
            pred_s_softmax = F.softmax(pred_s, -1)
            prediction = torch.sigmoid(prediction)
            
            pseudo_label = [pseudo_label_dic.get(key) for key in img_name]
            uncertain_map = [uncertain_dic.get(key) for key in img_name]
            proto_pseudo = [proto_pseudo_dic.get(key) for key in img_name]
            pseudo_bound = [pseudo_bound_dic.get(key) for key in img_name]

            pseudo_label = torch.from_numpy(np.asarray(pseudo_label)).float().cuda()
            uncertain_map = torch.from_numpy(np.asarray(uncertain_map)).float().cuda()
            proto_pseudo = torch.from_numpy(np.asarray(proto_pseudo)).float().cuda()
            pseudo_bound = torch.from_numpy(np.asarray(pseudo_bound)).float().cuda()
            
            # generate adversarial samples
			# atk = torchattacks.PGD(model, eps=4/255, alpha=2/255, steps=4)
            # adv_untargeted = atk(data, target)
            # #cv2.imwrite(os.path.join('/mnt/data1/llr_data/Fundus/',args.dataset,'/test/image/'+ sample['img_name'][0]), cv2.resize(255*adv_untargeted.data.cpu().numpy()[0,0,:,:],(800,800)))
            # cv2.imwrite('/mnt/data1/llr_data/Fundus/PGD/TENT/Domain6/'+ sample['img_name'][0], cv2.resize(adv_untargeted.data.cpu().numpy()[0,0,:,:],(800,800)))
            # cv2.imwrite('/mnt/data1/llr_data/Fundus/PGD/TENT/Domain6/'+ sample['img_name'][1], cv2.resize(adv_untargeted.data.cpu().numpy()[1,0,:,:],(800,800)))
        
            for param in model.parameters():
                param.requires_grad = True
            optim_gen.zero_grad()

            target_0_obj = F.interpolate(pseudo_label[:,0:1,...], size=feature.size()[2:], mode='nearest')
            target_1_obj = F.interpolate(pseudo_label[:, 1:, ...], size=feature.size()[2:], mode='nearest')
            target_0_bck = 1.0 - target_0_obj;target_1_bck = 1.0 - target_1_obj

            mask_0_obj = torch.zeros([pseudo_label.shape[0], 1, pseudo_label.shape[2], pseudo_label.shape[3]]).cuda()
            mask_0_bck = torch.zeros([pseudo_label.shape[0], 1, pseudo_label.shape[2], pseudo_label.shape[3]]).cuda()
            mask_1_obj = torch.zeros([pseudo_label.shape[0], 1, pseudo_label.shape[2], pseudo_label.shape[3]]).cuda()
            mask_1_bck = torch.zeros([pseudo_label.shape[0], 1, pseudo_label.shape[2], pseudo_label.shape[3]]).cuda()
            mask_0_obj[uncertain_map[:, 0:1, ...] < 0.05] = 1.0
            mask_0_bck[uncertain_map[:, 0:1, ...] < 0.05] = 1.0
            mask_1_obj[uncertain_map[:, 1:, ...] < 0.05] = 1.0
            mask_1_bck[uncertain_map[:, 1:, ...] < 0.05] = 1.0
            mask = torch.cat((mask_0_obj*pseudo_label[:,0:1,...] + mask_0_bck*(1.0-pseudo_label[:,0:1,...]), mask_1_obj*pseudo_label[:,1:,...] + mask_1_bck*(1.0-pseudo_label[:,1:,...])), dim=1)

            mask_proto = torch.zeros([data.shape[0], 2, data.shape[2], data.shape[3]]).cuda()
            mask_proto[pseudo_label==proto_pseudo] = 1.0
            mask = mask*mask_proto

            # mask for pseudo boundary
            target_0_obj = F.interpolate(pseudo_bound[:,0:1,...], size=feature.size()[2:], mode='nearest')
            target_0_bck = 1.0 - target_0_obj;target_1_bck = 1.0 - target_1_obj
            mask_0_obj = torch.zeros([pseudo_bound.shape[0], 1, pseudo_bound.shape[2], pseudo_bound.shape[3]]).cuda()
            mask_0_bck = torch.zeros([pseudo_bound.shape[0], 1, pseudo_bound.shape[2], pseudo_bound.shape[3]]).cuda()
            mask_0_obj[uncertain_map[:, 0:1, ...] < 0.05] = 1.0
            mask_0_bck[uncertain_map[:, 0:1, ...] < 0.05] = 1.0
            mask1 = mask_0_obj*pseudo_bound[:,0:1,...] + mask_0_bck*(1.0-pseudo_bound[:,0:1,...])
            mask_proto1 = torch.zeros([data.shape[0], 1, data.shape[2], data.shape[3]]).cuda()
            mask_proto1[pseudo_label[:,0:1,...]==proto_pseudo[:,0:1,...]] = 1.0
            mask1 = mask1*mask_proto1

            sceloss = SCELoss()
            mseloss = torch.nn.MSELoss()

            loss_seg_pixel = bceloss(prediction, pseudo_label)
			loss_seg = torch.sum(loss_seg_pixel) / torch.sum(mask)
            loss_seg_bound = mseloss(torch.sigmoid(boundaryS), pseudo_bound) #Lb for pseudo boundary
            loss_ent = entropy_loss(prediction)
            loss = loss_seg + loss_seg_bound + 0.4*loss_ent
        
            loss.backward()
            optim_gen.step()
            iter_num = iter_num + 1

        #test
        model_eval.train()
        pretrained_dict = model.state_dict()
        model_dict = model_eval.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_eval.load_state_dict(pretrained_dict)

        val_cup_dice = 0.0;val_disc_dice = 0.0;datanum_cnt = 0.0
        cup_hd = 0.0; disc_hd = 0.0;datanum_cnt_cup = 0.0;datanum_cnt_disc = 0.0
        with torch.no_grad():
            for batch_idx, (sample) in enumerate(test_loader): #test_loader
                data, target, img_name = sample['image'], sample['map'], sample['img_name']
                
                if torch.cuda.is_available():
                    data, target = data.cuda(), target.cuda()
                data, target = Variable(data), Variable(target)
                prediction, boundary, _ = model_eval(data)
                prediction = torch.sigmoid(prediction)

                target_numpy = target.data.cpu()
                prediction = prediction.data.cpu()
                prediction[prediction>0.75] = 1;prediction[prediction <= 0.75] = 0
                im = np.array(target_numpy[:,0, ...]).transpose(1,2,0) *255
                im1 = np.array(prediction[:,0, ...]).transpose(1,2,0) *255
                
                cup_dice = dice_coefficient_numpy(prediction[:,0, ...], target_numpy[:, 0, ...])
                disc_dice = dice_coefficient_numpy(prediction[:,1, ...], target_numpy[:, 1, ...])

                for i in range(prediction.shape[0]):
                    hd_tmp = hd_numpy(prediction[i, 0, ...], target_numpy[i, 0, ...], get_hd)
                    if np.isnan(hd_tmp):
                        datanum_cnt_cup -= 1.0
                    else:
                        cup_hd += hd_tmp

                    hd_tmp = hd_numpy(prediction[i, 1, ...], target_numpy[i, 1, ...], get_hd)
                    if np.isnan(hd_tmp):
                        datanum_cnt_disc -= 1.0
                    else:
                        disc_hd += hd_tmp

                val_cup_dice += np.sum(cup_dice)
                val_disc_dice += np.sum(disc_dice)

                datanum_cnt += float(prediction.shape[0])
                datanum_cnt_cup += float(prediction.shape[0])
                datanum_cnt_disc += float(prediction.shape[0])

        val_cup_dice /= datanum_cnt
        val_disc_dice /= datanum_cnt
        cup_hd /= datanum_cnt_cup
        disc_hd /= datanum_cnt_disc
        if (val_cup_dice+val_disc_dice)/2.0>best_avg:
            best_val_cup_dice = val_cup_dice; best_val_disc_dice = val_disc_dice; best_avg = (val_cup_dice+val_disc_dice)/2.0
            best_cup_hd = cup_hd; best_disc_hd = disc_hd; best_avg_hd = (best_cup_hd+best_disc_hd)/2.0

        if not os.path.exists('./logs/train_target'):
            os.mkdir('./logs/train_target')
        if args.dataset == 'Domain1':
            savefile = './logs/train_target/' + 'D1_' + 'checkpoint_%d.pth.tar' % epoch_num
        elif args.dataset == 'Domain2':
            savefile = './logs/train_target/' + 'D2_' + 'checkpoint_%d.pth.tar' % epoch_num
        elif args.dataset == 'Domain4':
            savefile = './logs/train_target/' + 'D4_' + 'checkpoint_%d.pth.tar' % epoch_num
        if model_save:
            torch.save({
                'model_state_dict': model.state_dict(),
                'best_mean_dice': best_avg,
                'best_cup_dice': best_val_cup_dice,
                'best_disc_dice': best_val_disc_dice,
            }, savefile)

        print("cup: %.4f disc: %.4f  cup: %.4f disc: %.4f " %
              (val_cup_dice, val_disc_dice, cup_hd, disc_hd))
        print("best disc: %.4f best disc d: %.4f best cup: %.4f  best cup d: %.4f " %
              (best_val_disc_dice, best_disc_hd, best_val_cup_dice, best_cup_hd, ))
        model.train()



