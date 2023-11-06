
#!/usr/bin/env python

import argparse
import os
import os.path as osp
import torch.nn.functional as F

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

import torch
from torch.autograd import Variable
import tqdm
from dataloaders import fundus_dataloader as DL
from torch.utils.data import DataLoader
from dataloaders import custom_transformssemi as tr
from torchvision import transforms

from matplotlib.pyplot import imsave
from utils.Utils import *
from metrics import *
from datetime import datetime
import pytz
from networks.deeplabv3 import *
import cv2
import torch.backends.cudnn as cudnn
import random

bceloss = torch.nn.BCELoss()
seed = 3377
savefig = False
get_hd = False
if True:
    cudnn.benchmark = False
    cudnn.deterministic = True
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-file', type=str, default='./logs/source/source_model.pth.tar')
    parser.add_argument('--dataset', type=str, default='Domain2')
    parser.add_argument('--batchsize', type=int, default=8)
    parser.add_argument('--source', type=str, default='Domain3')
    parser.add_argument('-g', '--gpu', type=int, default=0)
    parser.add_argument('--data-dir', default='/mnt/data1/llr_data/Fundus/')
    parser.add_argument('--out-stride',type=int,default=16)
    parser.add_argument('--save-root-ent',type=str,default='./results/ent/')
    parser.add_argument('--save-root-mask',type=str,default='./results/mask/')
    parser.add_argument('--sync-bn',type=bool,default=True)
    parser.add_argument('--freeze-bn',type=bool,default=False)
    parser.add_argument('--test-prediction-save-path', type=str,default='./results/baseline/')

    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    model_file = args.model_file

    # 1. dataset
    composed_transforms_test = transforms.Compose([
        tr.Resize(512),
        tr.Normalize_tf1(),
        tr.ToTensor()
    ])
    db_train = DL.FundusSegmentation(base_dir=args.data_dir, dataset=args.dataset, split='train/ROIs', transform=composed_transforms_test)
    db_test = DL.FundusSegmentation(base_dir=args.data_dir, dataset=args.dataset, split='test/ROIs', transform=composed_transforms_test)
    db_source = DL.FundusSegmentation(base_dir=args.data_dir, dataset=args.source, split='train/ROIs', transform=composed_transforms_test)

    train_loader = DataLoader(db_train, batch_size=args.batchsize, shuffle=False, num_workers=1)
    test_loader = DataLoader(db_test, batch_size=args.batchsize, shuffle=False, num_workers=1)
    source_loader = DataLoader(db_source, batch_size=args.batchsize, shuffle=False, num_workers=1)

    # 2. model
    model = DeepLab(num_classes=2, backbone='mobilenet', output_stride=args.out_stride, sync_bn=args.sync_bn, freeze_bn=args.freeze_bn)

    if torch.cuda.is_available():
        model = model.cuda()
    print('==> Loading %s model file: %s' %
          (model.__class__.__name__, model_file))
    checkpoint = torch.load(model_file)

    model.load_state_dict(checkpoint['model_state_dict'])
    #model.load_state_dict({k.replace('module.',''):v for k,v in torch.load(model_file).items()})

    model.train()

    pseudo_bound_dic = {}

    with torch.no_grad():
        for batch_idx, (sample) in tqdm.tqdm(enumerate(train_loader),
                                             total=len(train_loader),
                                             ncols=80, leave=False):
            data, target, img_name = sample['image'], sample['map'], sample['img_name']
            if torch.cuda.is_available():
                data, target = data.cuda(), target.cuda()
            data, target = Variable(data), Variable(target)

            preds = torch.zeros([10, data.shape[0], 2, data.shape[2], data.shape[3]]).cuda()
            features = torch.zeros([10, data.shape[0], 305, 128, 128]).cuda()
            boundary = torch.zeros([10, data.shape[0], 1,  data.shape[2], data.shape[3]]).cuda()
            for i in range(10):
                with torch.no_grad():
                    preds[i,...], boundary[i,...], features[i,...] = model(data)
            preds1 = torch.sigmoid(preds)
            preds = torch.sigmoid(preds/2.0)
            prediction=torch.mean(preds1,dim=0)
            pseudo_label = prediction.clone()
            prediction=torch.mean(preds1,dim=0)

            b1 = torch.sigmoid(boundary)
            b = torch.mean(b1,dim=0)
            pseudo_bound = b.clone()
            # try bound threshold---no! boundary generation is different with pseudo labels
            # pseudo_bound[pseudo_bound > 0.75] = 1.0; pseudo_bound[pseudo_bound <= 0.75] = 0.0
            
            feature = torch.mean(features,dim=0)
            pseudo_bound = pseudo_bound.detach().cpu().numpy()
            for i in range(prediction.shape[0]):
                pseudo_bound_dic[img_name[i]] = pseudo_bound[i]
                

    if args.dataset=="Domain1":#pseudolabel_D1
        np.savez('./results/bound/r-bound_D1', pseudo_bound_dic)

    elif args.dataset=="Domain2":
        np.savez('./results/bound/r-bound_D2', pseudo_bound_dic)
    elif args.dataset=="RIGA":
        np.savez('/mnt/data1/llr_data/results/bound/test-bound_D6', pseudo_bound_dic)




