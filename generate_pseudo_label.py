
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
    parser.add_argument('--model-file', type=str, default='./logs/Domain2/1/checkpoint_200.pth.tar')
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
    db_train = DL.FundusSegmentationRIGA(base_dir=args.data_dir, dataset=args.dataset, split='train/', transform=composed_transforms_test)
    db_test = DL.FundusSegmentationRIGA(base_dir=args.data_dir, dataset=args.dataset, split='test/', transform=composed_transforms_test)
    db_source = DL.FundusSegmentationRIGA(base_dir=args.data_dir, dataset=args.source, split='train/ROIs', transform=composed_transforms_test)
    # db_train = DL.CellSegmentation(base_dir=args.data_dir, dataset=args.dataset, split='train/', transform=composed_transforms_test)
    # db_test = DL.CellSegmentation(base_dir=args.data_dir, dataset=args.dataset, split='test/', transform=composed_transforms_test)
    # db_source = DL.CellSegmentation(base_dir=args.data_dir, dataset=args.source, split='train/', transform=composed_transforms_test)

    train_loader = DataLoader(db_train, batch_size=args.batchsize, shuffle=False, num_workers=1,drop_last=True)
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

    pseudo_label_dic = {}
    uncertain_dic = {}
    proto_pseudo_dic = {}
    distance_0_obj_dic = {}
    distance_0_bck_dic = {}
    distance_1_bck_dic = {}
    distance_1_obj_dic = {}
    centroid_0_obj_dic = {}
    centroid_0_bck_dic = {}
    centroid_1_obj_dic = {}
    centroid_1_bck_dic = {}
    # fundus
    with torch.no_grad():
        for batch_idx, (sample) in tqdm.tqdm(enumerate(test_loader),
                                             total=len(train_loader),
                                             ncols=80, leave=False):
            data, target, img_name = sample['image'], sample['map'], sample['img_name']
            if torch.cuda.is_available():
                data, target = data.cuda(), target.cuda()
            data, target = Variable(data), Variable(target)

            preds = torch.zeros([10, data.shape[0], 2, data.shape[2], data.shape[3]]).cuda()
            features = torch.zeros([10, data.shape[0], 305, 128, 128]).cuda()
            for i in range(10):
                with torch.no_grad():
                    preds[i,...], _, features[i,...] = model(data)
            preds1 = torch.sigmoid(preds)
            preds = torch.sigmoid(preds/2.0)
            std_map = torch.std(preds,dim=0)

            prediction=torch.mean(preds1,dim=0)

            # complementary label generation
            # pred_s = preds[0,:,:,:,:].permute(0, 2, 3, 1).contiguous().view(-1, 2)
            # pred_s_softmax = F.softmax(pred_s, -1)
            # width = 1
            # k = 2 // 2 + random.randint(-width, width)
            # _, labels_neg = torch.topk(pred_s_softmax, 2, dim=1, sorted=True)
            # s_neg = torch.log(torch.clamp(1. - pred_s_softmax, min=1e-5, max=1.))
            # labels_neg = labels_neg[:, -1].squeeze().detach()
            # print(s_neg.shape)
            # print('negative label:', labels_neg.shape)

            # # save pseudo negative label image
            # mask_oc = construct_color_img(labels_neg[0][0,:,:])
            # mask_oc = cv2.cvtColor(mask_oc, cv2.COLOR_BGR2GRAY)
            # print(pseudo_label.shape)
            # mask_od = construct_color_img(labels_neg[0][1,:,:])
            # mask_od = cv2.cvtColor(mask_od, cv2.COLOR_BGR2GRAY)
            # cv2.imwrite(os.path.join('/mnt/data1/llr_data/results/pseudoneglabel/', args.dataset, 'oc',sample['img_name'][0]), mask_oc)
            # cv2.imwrite(os.path.join('/mnt/data1/llr_data/results/pseudoneglabel/', args.dataset, 'od',sample['img_name'][0]), mask_od)


            pseudo_label = prediction.clone()
            pseudo_label[pseudo_label > 0.75] = 1.0; pseudo_label[pseudo_label <= 0.75] = 0.0

            feature = torch.mean(features,dim=0)

            target_0_obj = F.interpolate(pseudo_label[:,0:1,...], size=feature.size()[2:], mode='nearest')
            target_1_obj = F.interpolate(pseudo_label[:, 1:, ...], size=feature.size()[2:], mode='nearest')
            prediction_small = F.interpolate(prediction, size=feature.size()[2:], mode='bilinear', align_corners=True)
            std_map_small = F.interpolate(std_map, size=feature.size()[2:], mode='bilinear', align_corners=True)
            target_0_bck = 1.0 - target_0_obj;target_1_bck = 1.0 - target_1_obj

            mask_0_obj = torch.zeros([std_map_small.shape[0], 1, std_map_small.shape[2], std_map_small.shape[3]]).cuda()
            mask_0_bck = torch.zeros([std_map_small.shape[0], 1, std_map_small.shape[2], std_map_small.shape[3]]).cuda()
            mask_1_obj = torch.zeros([std_map_small.shape[0], 1, std_map_small.shape[2], std_map_small.shape[3]]).cuda()
            mask_1_bck = torch.zeros([std_map_small.shape[0], 1, std_map_small.shape[2], std_map_small.shape[3]]).cuda()
            mask_0_obj[std_map_small[:, 0:1, ...] < 0.05] = 1.0
            mask_0_bck[std_map_small[:, 0:1, ...] < 0.05] = 1.0
            mask_1_obj[std_map_small[:, 1:, ...] < 0.05] = 1.0
            mask_1_bck[std_map_small[:, 1:, ...] < 0.05] = 1.0
            mask_0 = mask_0_obj + mask_0_bck
            mask_1 = mask_1_obj + mask_1_bck
            mask = torch.cat((mask_0, mask_1), dim=1)

            feature_0_obj = feature * target_0_obj*mask_0_obj;feature_1_obj = feature * target_1_obj*mask_1_obj
            feature_0_bck = feature * target_0_bck*mask_0_bck;feature_1_bck = feature * target_1_bck*mask_1_bck

            centroid_0_obj = torch.sum(feature_0_obj*prediction_small[:,0:1,...], dim=[0,2,3], keepdim=True)
            centroid_1_obj = torch.sum(feature_1_obj*prediction_small[:,1:,...], dim=[0,2,3], keepdim=True)
            centroid_0_bck = torch.sum(feature_0_bck*(1.0-prediction_small[:,0:1,...]), dim=[0,2,3], keepdim=True)
            centroid_1_bck = torch.sum(feature_1_bck*(1.0-prediction_small[:,1:,...]), dim=[0,2,3], keepdim=True)
            target_0_obj_cnt = torch.sum(mask_0_obj*target_0_obj*prediction_small[:,0:1,...], dim=[0,2,3], keepdim=True)
            target_1_obj_cnt = torch.sum(mask_1_obj*target_1_obj*prediction_small[:,1:,...], dim=[0,2,3], keepdim=True)
            target_0_bck_cnt = torch.sum(mask_0_bck*target_0_bck*(1.0-prediction_small[:,0:1,...]), dim=[0,2,3], keepdim=True)
            target_1_bck_cnt = torch.sum(mask_1_bck*target_1_bck*(1.0-prediction_small[:,1:,...]), dim=[0,2,3], keepdim=True)

            centroid_0_obj /= target_0_obj_cnt; centroid_1_obj /= target_1_obj_cnt
            centroid_0_bck /= target_0_bck_cnt; centroid_1_bck /= target_1_bck_cnt

            distance_0_obj = torch.sum(torch.pow(feature - centroid_0_obj, 2), dim=1, keepdim=True)
            distance_0_bck = torch.sum(torch.pow(feature - centroid_0_bck, 2), dim=1, keepdim=True)
            distance_1_obj = torch.sum(torch.pow(feature - centroid_1_obj, 2), dim=1, keepdim=True)
            distance_1_bck = torch.sum(torch.pow(feature - centroid_1_bck, 2), dim=1, keepdim=True)

            proto_pseudo_0 = torch.zeros([data.shape[0], 1, feature.shape[2], feature.shape[3]]).cuda()
            proto_pseudo_1 = torch.zeros([data.shape[0], 1, feature.shape[2], feature.shape[3]]).cuda()

            proto_pseudo_0[distance_0_obj < distance_0_bck] = 1.0
            proto_pseudo_1[distance_1_obj < distance_1_bck] = 1.0
            proto_pseudo = torch.cat((proto_pseudo_0, proto_pseudo_1), dim=1)
            proto_pseudo = F.interpolate(proto_pseudo, size=data.size()[2:], mode='nearest')

            debugc = 1

            pseudo_label = pseudo_label.detach().cpu().numpy()

            # save pseudo label image
            # mask_oc = construct_color_img(pseudo_label[0][0,:,:])
            # mask_oc = cv2.cvtColor(mask_oc, cv2.COLOR_BGR2GRAY)
            # print(pseudo_label.shape)
            # mask_od = construct_color_img(pseudo_label[0][1,:,:])
            # mask_od = cv2.cvtColor(mask_od, cv2.COLOR_BGR2GRAY)
            # cv2.imwrite(os.path.join('/mnt/data1/llr_data/results/pseudolabel/boundary', args.dataset, 'oc',sample['img_name'][0]), mask_oc)
            # cv2.imwrite(os.path.join('/mnt/data1/llr_data/results/pseudolabel/boundary', args.dataset, 'od',sample['img_name'][0]), mask_od)

            # save pseudo boundary image
            # mask_oc = construct_color_img(pseudo_label[0][0,:,:])
            # mask_oc = cv2.cvtColor(mask_oc, cv2.COLOR_BGR2GRAY)
            # img = np.zeros((512,512,3),np.uint8)
            # #img = Image.fromarray(img)
            # ret, thresh = cv2.threshold(mask_oc, 255, 255, 0)
            # contours, im = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) #第一个参数是轮廓
            # result_oc = cv2.drawContours(image=img, contours=contours, contourIdx=-1, color=(255, 255, 255), thickness=5)
            
            # mask_od = cv2.imread(os.path.join('/mnt/data1/llr_data/results/pseudolabel/Domain2/oc',sample['img_name'][0]))
            # mask_od = cv2.cvtColor(mask_od, cv2.COLOR_BGR2GRAY)
            # ret, thresh = cv2.threshold(mask_od, 127, 255, 0)
            # contours, im = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) #第一个参数是轮廓
            # result_od = cv2.drawContours(image=img, contours=contours, contourIdx=-1, color=(255, 255, 255), thickness=5)
            # cv2.imwrite(os.path.join('/mnt/data1/llr_data/results/pseudolabel/Domain2/boundary/oc',sample['img_name'][0]), result_oc)
            # cv2.imwrite(os.path.join('/mnt/data1/llr_data/results/pseudolabel/Domain2/boundary/od',sample['img_name'][0]), result_od)


            std_map = std_map.detach().cpu().numpy()
            proto_pseudo = proto_pseudo.detach().cpu().numpy()
            distance_0_obj = distance_0_obj.detach().cpu().numpy()
            distance_0_bck = distance_0_bck.detach().cpu().numpy()
            distance_1_obj = distance_1_obj.detach().cpu().numpy()
            distance_1_bck = distance_1_bck.detach().cpu().numpy()
            centroid_0_obj = centroid_0_obj.detach().cpu().numpy()
            centroid_0_bck = centroid_0_bck.detach().cpu().numpy()
            centroid_1_obj = centroid_1_obj.detach().cpu().numpy()
            centroid_1_bck = centroid_1_bck.detach().cpu().numpy()
            for i in range(prediction.shape[0]):
                pseudo_label_dic[img_name[i]] = pseudo_label[i]
                uncertain_dic[img_name[i]] = std_map[i]
                proto_pseudo_dic[img_name[i]] = proto_pseudo[i]
                distance_0_obj_dic[img_name[i]] = distance_0_obj[i]
                distance_0_bck_dic[img_name[i]] = distance_0_bck[i]
                distance_1_obj_dic[img_name[i]] = distance_1_obj[i]
                distance_1_bck_dic[img_name[i]] = distance_1_bck[i]
                centroid_0_obj_dic[img_name[i]] = centroid_0_obj
                centroid_0_bck_dic[img_name[i]] = centroid_0_bck
                centroid_1_obj_dic[img_name[i]] = centroid_1_obj
                centroid_1_bck_dic[img_name[i]] = centroid_1_bck
    #cell
    # with torch.no_grad():
    #     for batch_idx, (sample) in tqdm.tqdm(enumerate(train_loader),
    #                                          total=len(train_loader),
    #                                          ncols=80, leave=False):
    #         data, target, img_name = sample['image'], sample['map'], sample['img_name']
    #         if torch.cuda.is_available():
    #             data, target = data.cuda(), target.cuda()
    #         data, target = Variable(data), Variable(target)

    #         preds = torch.zeros([10, data.shape[0], 1, data.shape[2], data.shape[3]]).cuda()
    #         features = torch.zeros([10, data.shape[0], 305, 128, 128]).cuda()
    #         for i in range(10):
    #             with torch.no_grad():
    #                 preds[i,...], _, features[i,...] = model(data)
    #         preds1 = torch.sigmoid(preds)
    #         preds = torch.sigmoid(preds/2.0)
    #         std_map = torch.std(preds,dim=0)

    #         prediction=torch.mean(preds1,dim=0)
    #         pseudo_label = prediction.clone()
    #         pseudo_label[pseudo_label > 0.75] = 1.0; pseudo_label[pseudo_label <= 0.75] = 0.0

    #         feature = torch.mean(features,dim=0)

    #         target_0_obj = F.interpolate(pseudo_label[:,0:1,...], size=feature.size()[2:], mode='nearest')
    #         #target_1_obj = F.interpolate(pseudo_label[:, 1:, ...], size=feature.size()[2:], mode='nearest')
    #         prediction_small = F.interpolate(prediction, size=feature.size()[2:], mode='bilinear', align_corners=True)
    #         std_map_small = F.interpolate(std_map, size=feature.size()[2:], mode='bilinear', align_corners=True)
    #         target_0_bck = 1.0 - target_0_obj#;target_1_bck = 1.0 - target_1_obj

    #         mask_0_obj = torch.zeros([std_map_small.shape[0], 1, std_map_small.shape[2], std_map_small.shape[3]]).cuda()
    #         mask_0_bck = torch.zeros([std_map_small.shape[0], 1, std_map_small.shape[2], std_map_small.shape[3]]).cuda()
    #         # mask_1_obj = torch.zeros([std_map_small.shape[0], 1, std_map_small.shape[2], std_map_small.shape[3]]).cuda()
    #         # mask_1_bck = torch.zeros([std_map_small.shape[0], 1, std_map_small.shape[2], std_map_small.shape[3]]).cuda()
    #         mask_0_obj[std_map_small[:, 0:1, ...] < 0.05] = 1.0
    #         mask_0_bck[std_map_small[:, 0:1, ...] < 0.05] = 1.0
    #         # mask_1_obj[std_map_small[:, 1:, ...] < 0.05] = 1.0
    #         # mask_1_bck[std_map_small[:, 1:, ...] < 0.05] = 1.0
    #         mask_0 = mask_0_obj + mask_0_bck
    #         # mask_1 = mask_1_obj + mask_1_bck
    #         mask = mask_0

    #         feature_0_obj = feature * target_0_obj*mask_0_obj#;feature_1_obj = feature * target_1_obj*mask_1_obj
    #         feature_0_bck = feature * target_0_bck*mask_0_bck#;feature_1_bck = feature * target_1_bck*mask_1_bck

    #         centroid_0_obj = torch.sum(feature_0_obj*prediction_small[:,0:1,...], dim=[0,2,3], keepdim=True)
    #         # centroid_1_obj = torch.sum(feature_1_obj*prediction_small[:,1:,...], dim=[0,2,3], keepdim=True)
    #         centroid_0_bck = torch.sum(feature_0_bck*(1.0-prediction_small[:,0:1,...]), dim=[0,2,3], keepdim=True)
    #         # centroid_1_bck = torch.sum(feature_1_bck*(1.0-prediction_small[:,1:,...]), dim=[0,2,3], keepdim=True)
    #         target_0_obj_cnt = torch.sum(mask_0_obj*target_0_obj*prediction_small[:,0:1,...], dim=[0,2,3], keepdim=True)
    #         #target_1_obj_cnt = torch.sum(mask_1_obj*target_1_obj*prediction_small[:,1:,...], dim=[0,2,3], keepdim=True)
    #         target_0_bck_cnt = torch.sum(mask_0_bck*target_0_bck*(1.0-prediction_small[:,0:1,...]), dim=[0,2,3], keepdim=True)
    #         #target_1_bck_cnt = torch.sum(mask_1_bck*target_1_bck*(1.0-prediction_small[:,1:,...]), dim=[0,2,3], keepdim=True)

    #         centroid_0_obj /= target_0_obj_cnt#; centroid_1_obj /= target_1_obj_cnt
    #         centroid_0_bck /= target_0_bck_cnt#; centroid_1_bck /= target_1_bck_cnt

    #         distance_0_obj = torch.sum(torch.pow(feature - centroid_0_obj, 2), dim=1, keepdim=True)
    #         distance_0_bck = torch.sum(torch.pow(feature - centroid_0_bck, 2), dim=1, keepdim=True)
    #         # distance_1_obj = torch.sum(torch.pow(feature - centroid_1_obj, 2), dim=1, keepdim=True)
    #         # distance_1_bck = torch.sum(torch.pow(feature - centroid_1_bck, 2), dim=1, keepdim=True)

    #         proto_pseudo_0 = torch.zeros([data.shape[0], 1, feature.shape[2], feature.shape[3]]).cuda()
    #         #proto_pseudo_1 = torch.zeros([data.shape[0], 1, feature.shape[2], feature.shape[3]]).cuda()

    #         proto_pseudo_0[distance_0_obj < distance_0_bck] = 1.0
    #         #proto_pseudo_1[distance_1_obj < distance_1_bck] = 1.0
    #         proto_pseudo = proto_pseudo_0
    #         proto_pseudo = F.interpolate(proto_pseudo, size=data.size()[2:], mode='nearest')

    #         debugc = 1

    #         pseudo_label = pseudo_label.detach().cpu().numpy()
    #         std_map = std_map.detach().cpu().numpy()
    #         proto_pseudo = proto_pseudo.detach().cpu().numpy()
    #         distance_0_obj = distance_0_obj.detach().cpu().numpy()
    #         distance_0_bck = distance_0_bck.detach().cpu().numpy()
    #         # distance_1_obj = distance_1_obj.detach().cpu().numpy()
    #         # distance_1_bck = distance_1_bck.detach().cpu().numpy()
    #         centroid_0_obj = centroid_0_obj.detach().cpu().numpy()
    #         centroid_0_bck = centroid_0_bck.detach().cpu().numpy()
    #         # centroid_1_obj = centroid_1_obj.detach().cpu().numpy()
    #         # centroid_1_bck = centroid_1_bck.detach().cpu().numpy()
    #         for i in range(prediction.shape[0]):
    #             pseudo_label_dic[img_name[i]] = pseudo_label[i]
    #             uncertain_dic[img_name[i]] = std_map[i]
    #             proto_pseudo_dic[img_name[i]] = proto_pseudo[i]
    #             distance_0_obj_dic[img_name[i]] = distance_0_obj[i]
    #             distance_0_bck_dic[img_name[i]] = distance_0_bck[i]
    #             # distance_1_obj_dic[img_name[i]] = distance_1_obj[i]
    #             # distance_1_bck_dic[img_name[i]] = distance_1_bck[i]
    #             centroid_0_obj_dic[img_name[i]] = centroid_0_obj
    #             centroid_0_bck_dic[img_name[i]] = centroid_0_bck
    #             # centroid_1_obj_dic[img_name[i]] = centroid_1_obj
    #             # centroid_1_bck_dic[img_name[i]] = centroid_1_bck

    if args.dataset=="Domain1":#pseudolabel_D1
        np.savez('/mnt/data1/llr_data/results/cell/pseudolabel_D1', pseudo_label_dic, uncertain_dic, proto_pseudo_dic,
                         distance_0_obj_dic, distance_0_bck_dic, distance_1_obj_dic, distance_1_bck_dic,
                         centroid_0_obj_dic, centroid_0_bck_dic, centroid_1_obj_dic, centroid_1_bck_dic
                         )

    elif args.dataset=="Domain2":
        np.savez('/mnt/data1/llr_data/results/pseudolabel_D2', pseudo_label_dic, uncertain_dic, proto_pseudo_dic,
                         distance_0_obj_dic, distance_0_bck_dic, distance_1_obj_dic, distance_1_bck_dic,
                         centroid_0_obj_dic, centroid_0_bck_dic, centroid_1_obj_dic, centroid_1_bck_dic
                         )
    elif args.dataset=="RIGA":
        np.savez('/mnt/data1/llr_data/results/prototype/test_D6', pseudo_label_dic, uncertain_dic, proto_pseudo_dic,
                         distance_0_obj_dic, distance_0_bck_dic, distance_1_obj_dic, distance_1_bck_dic,
                         centroid_0_obj_dic, centroid_0_bck_dic, centroid_1_obj_dic, centroid_1_bck_dic
                         )




