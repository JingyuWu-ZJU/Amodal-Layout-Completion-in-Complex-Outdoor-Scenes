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
import models
import utils
import tqdm
import warnings
import KINS_dataloader
from torchvision.utils import make_grid
# warnings.filterwarnings("ignore")
# import datasets
# import inference as infer
params = {}
params['optim'] = 'SGD'
params['lr'] = 0.001
params['weight_decay'] = 0.0001
params['use_rgb'] = False
params['inmask_weight'] = 0.2
params['in_ch'] = 2
params['num_cls'] = 2

annfile_path='data/KINS/update_train_2020.json' #using category or not 
annfile_path2 = 'data/KINS/instances_train.json'
d-a-c_path = 'divide-and-conquer-result.json'
image_path = 'data/KINS/training/image_2/'
Epoch = 200
model = models.PartialCompletionMask(params = params)
# tb_logger = SummaryWriter('events2')

lr_scheduler = utils.StepLRScheduler(model.optim,[16000,32000, 48000, 64000],[0,1,0.1, 0.1, 0.1],0.001,[],[],last_iter= -1)

# train_dataset = cocoa_dataloader.CocoAloader(dataset='cocoa',annfile_path=train_annfile_path,eraser_setter=eraser_setter,size = 256,eraser_front_prob = 0.8,mode = 'overlap')
# train_loader = DataLoader(train_dataset,batch_size=32,shuffle=True,num_workers=2)

# val_dataset = cocoa_dataloader.CocoAloader(dataset='cocoa',annfile_path=val_annfile_path,eraser_setter=eraser_setter,size = 256,eraser_front_prob = 0.8,mode = 'overlap')
# val_loader = DataLoader(val_dataset,batch_size=16,shuffle=False,num_workers=1)
train_dataset = KINS_dataloader.KINSloader_category(dataset='KINS',annfile_path=annfile_path2,image_path = image_path)
train_loader = DataLoader(train_dataset,batch_size=64,shuffle=True,num_workers=1)

# val_dataset = KINS_dataloader.KINSloader_category(dataset='KINS',annfile_path='data/KINS/update_test_2020.json',image_path = 'data/KINS/testing/image_2/')
# val_loader = DataLoader(val_dataset,batch_size=16,shuffle=True,num_workers=1)

for epoch in range(Epoch):
    print('epoch:'+str(epoch))
    for idx,input in enumerate(tqdm.tqdm(train_loader)):
        source_mask, target_mask, eraser_mask, source_category, eraser_category = input
        lr_scheduler.step(epoch*(len(train_loader))+idx+1)
        curr_lr = lr_scheduler.get_lr()[0]
 
        # loss_dict = model.train(source_mask,target_mask,eraser_mask)
  
        model.set_input(mask=source_mask, eraser=eraser_mask, target=target_mask,source_category=source_category,eraser_category=eraser_category)
        loss_dict = model.step()
        # tb_logger.add_scalar('train_loss',loss_dict['loss'].item(),epoch*(len(train_loader))+idx+1) 
        # if  (idx+1) == len(train_loader):
        #     model.save_state("",epoch = epoch,Iter = idx+1)
        # #val
        # if (idx+1) == len(train_loader):
        #     model.switch_to('eval')
        #     all_together = []
        #     for idx2,input2 in enumerate(val_loader):
        #         source_mask2, target_mask2, eraser_mask2, source_category2, eraser_category2 = input2
        #         model.set_input(mask=source_mask2, eraser=eraser_mask2, target=target_mask2,source_category=source_category2,eraser_category=eraser_category2)
        #         tensor_dict, loss_dict = model.forward_only()
        #         all_together.append(utils.visualize_tensor(tensor_dict,[0,0,0],[1,1,1]))
        #         all_together = torch.cat(all_together, dim=2)
        #         grid = vutils.make_grid(all_together,nrow=1,normalize=True,range=(0, 255),scale_each=False)
        #         tb_logger.add_image('Image_on_val', grid,epoch*(len(train_loader))+idx+1)
        #         break
        #     model.switch_to('train')

