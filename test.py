import os
import cv2
import time
import numpy as np
from tensorboardX import SummaryWriter
import torch
import torch.optim
import torch.distributed as dist
import torchvision.utils as vutils
from torch.utils.data import DataLoader
import cocoa_dataloader
import cocoa_val_dataloader 
import KINS_dataloader
import models
import utils
import tqdm
import argparse
import warnings
import imageio
import csv
warnings.filterwarnings("ignore")

def bbox_iou(box1, box2, x1y1x2y2=False, eps=1e-7,mode = "IOU"):
    flag = 0 
    # Get the coordinates of bounding boxes
    box1 = torch.from_numpy(box1)
    box2 = torch.from_numpy(box2)
    if x1y1x2y2:  # x1, y1, x2, y2 = box1
        b1_x1, b1_y1, b1_x2, b1_y2 = box1[0], box1[1], box1[2], box1[3]
        b2_x1, b2_y1, b2_x2, b2_y2 = box2[0], box2[1], box2[2], box2[3]
    else:  # transform from xywh to xyxy
        b1_x1, b1_x2 = box1[0], box1[0] + box1[2]
        b1_y1, b1_y2 = box1[1], box1[1] + box1[3]
        b2_x1, b2_x2 = box2[0], box2[0] + box2[2]
        b2_y1, b2_y2 = box2[1], box2[1] + box2[3] 

    # Intersection area
    inter = (torch.min(b1_x2, b2_x2) - torch.max(b1_x1, b2_x1)).clamp(0) * \
            (torch.min(b1_y2, b2_y2) - torch.max(b1_y1, b2_y1)).clamp(0)
    # Union Area
    w1, h1 = b1_x2 - b1_x1, b1_y2 - b1_y1 + eps
    w2, h2 = b2_x2 - b2_x1, b2_y2 - b2_y1 + eps
    union = w1 * h1 + w2 * h2 - inter + eps
    if mode == 'IOU':
        # print('IOU')
        iou = inter / union
        return iou
    else:
        print('AP')
        AP = inter/(w1*h1)
        return AP

def main(args):
    if args.mode == 1:
        params = {}
        params['optim'] = 'SGD'
        params['lr'] = 0.001
        params['weight_decay'] = 0.0001
        params['use_rgb'] = False
        params['inmask_weight'] = 5.
        params['in_ch'] = 2
        params['num_cls'] = 2
        annfile_path='data/KINS/update_test_2020.json'
        image_path = 'data/KINS/testing/image_2/'
        model = models.PartialCompletionMask(params = params)
        model.load_state(args.model_path)
        # train_dataset = KINS_dataloader.KINSloader_complete_layout(dataset='KINS',annfile_path=annfile_path,image_path = image_path)
        train_dataset = KINS_dataloader.KINSloader_rgb(dataset='KINS',annfile_path=annfile_path,image_path = image_path)
        train_loader = DataLoader(train_dataset,batch_size=1,shuffle=False,num_workers=1)
        source_total_IOU = 0
        source_big_IOU = 0 
        source_small_IOU = 0 
        small_num = 0
        big_num = 0
        result_total_IOU = 0
        result_big_IOU = 0 
        result_small_IOU = 0 
        total = 0 
        gt = 0 
        ALCN = 0
        test_num = 0
        AP50,AP55,AP60,AP65,AP70,AP75,AP80,AP85,AP90,AP95 = 0,0,0,0,0,0,0,0,0,0
        for idx, input in enumerate(tqdm.tqdm(train_loader)):
            occ_num = 0
     
            source_mask, target_mask, eraser_mask, source_category, eraser_category,image,flag = input
            source_bbox, target_bbox = utils.mask_to_bbox(source_mask[0][0].numpy()),utils.mask_to_bbox(target_mask[0].numpy())
            IOU = bbox_iou(np.array(source_bbox),np.array(target_bbox))  
   
            model.set_input(mask=source_mask, eraser=eraser_mask, target=target_mask,source_category=source_category,eraser_category=eraser_category,image=image)
            tensor_dict, loss_dict = model.forward_only()
            result_tensor = tensor_dict['mask_tensors']
            result_bbox = utils.mask_to_bbox(result_tensor[2][0][0].cpu().numpy())
            IOU2 = bbox_iou(np.array(result_bbox),np.array(target_bbox))

    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='coco',
                        help='training dataset')
    parser.add_argument('--model_path', type=str, default = '',
                        help='which epoch to load')
    parser.add_argument('--model_number', type=int, default='0')
    parser.add_argument('--mode', type=int, default='1')
    parser.add_argument('--IOU', type=float, default='0.75')
    args = parser.parse_args()
    main(args)