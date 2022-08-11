import numpy as np

import torch
import torch.nn as nn

import utils
import inference as infer
from . import SingleStageModel
from . import MaskWeightedCrossEntropyLoss

import pdb

class PartialCompletionMask(SingleStageModel):

    def __init__(self, params, load_pretrain=None, dist_model=False):
        super(PartialCompletionMask, self).__init__(params, dist_model)
        self.params = params
        self.use_rgb = params['use_rgb']

        # loss
        self.criterion = MaskWeightedCrossEntropyLoss(
            inmask_weight=params['inmask_weight'],
            outmask_weight=1.)

    def set_input(self,mask=None, eraser=None, target=None,source_category=None,eraser_category=None,image=None):
        # self.rgb = rgb.cuda()
        self.mask = mask.cuda()
        self.eraser = eraser.cuda()
        self.target = target.cuda()
        self.source_category = source_category.cuda()
        self.eraser_category = eraser_category.cuda()
        # self.image = image.cuda()

    def forward_only(self, ret_loss=True):
        with torch.no_grad():
            if self.use_rgb:
                output = self.model(torch.cat([self.mask, self.eraser], dim=1), self.rgb)
            else:
                # output = self.model(torch.cat([self.mask, self.eraser], dim=1))
                # output = self.model(torch.cat([self.mask, self.eraser], dim=1),torch.cat([self.source_category,self.eraser_category],dim=1))
                # output = self.model(torch.cat([self.mask, self.eraser,self.image], dim=1))
                output = self.model(torch.cat([self.mask, self.eraser], dim=1),self.source_category)
                # output = self.model(torch.cat([self.mask, self.eraser], dim=1),self.eraser_category)
            if output.shape[2] != self.mask.shape[2]:
                output = nn.functional.interpolate(
                    output, size=self.mask.shape[2:4],
                    mode="bilinear", align_corners=True)
        comp = output.argmax(dim=1, keepdim=True).float()
        comp[self.eraser == 0] = (self.mask > 0).float()[self.eraser == 0]

        vis_combo = (self.mask > 0).float()
        vis_combo[self.eraser == 1] = 0.5
        vis_target = self.target.cpu().clone().float()
        if vis_target.max().item() == 255:
            vis_target[vis_target == 255] = 0.5
        vis_target = vis_target.unsqueeze(1)
        if self.use_rgb:
            cm_tensors = [self.rgb]
        else:
            cm_tensors = []
        ret_tensors = {'common_tensors': cm_tensors,
                       'mask_tensors': [self.mask, vis_combo, comp, vis_target]}
        if ret_loss:
            loss = self.criterion(output, self.target, self.eraser.squeeze(1)) / self.world_size
            return ret_tensors, {'loss': loss}
        else:
            return ret_tensors

    def step(self):
        if self.use_rgb:
            output = self.model(torch.cat([self.mask, self.eraser], dim=1), self.rgb)
        else:
            # output = self.model(torch.cat([self.mask, self.eraser], dim=1))
            output = self.model(torch.cat([self.mask, self.eraser], dim=1),self.source_category)
            # output = self.model(torch.cat([self.mask, self.eraser], dim=1),self.eraser_category)
        loss = self.criterion(output, self.target, self.eraser.squeeze(1)) / self.world_size
        self.optim.zero_grad()
        loss.backward()
        # utils.average_gradients(self.model)
        self.optim.step()
        return {'loss': loss}

