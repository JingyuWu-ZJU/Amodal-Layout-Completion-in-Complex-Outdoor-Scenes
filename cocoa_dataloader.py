import numpy as np
try:
    import mc
except Exception:
    pass
import cv2
import os
from PIL import Image

import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms

import utils
import reader
import imageio


class CocoAloader(Dataset):
    def __init__(self,dataset,annfile_path,eraser_setter,size = 256,eraser_front_prob = 0.8,mode = 'overlap'):
        self.dataset = dataset
        self.data_reader = reader.COCOADataset(annfile_path)
        self.eraser_setter = utils.EraserSetter(eraser_setter)
        self.sz = size
        self.eraser_front_prob = eraser_front_prob
        self.mode = mode #phase
        self.load_rgb = False
        self.enlarge_box = 3.
        self.shift = [-0.2, 0.2]
        self.scale = [0.8, 1.2]
        self.flip = False

    def __len__(self):
        return self.data_reader.get_instance_length()


    def _load_image(self, fn):
        if self.memcached:
            try:
                img_value = mc.pyvector()
                self.mclient.Get(fn, img_value)
                img_value_str = mc.ConvertBuffer(img_value)
                img = utils.pil_loader(img_value_str)
            except:
                print('Read image failed ({})'.format(fn))
                raise Exception("Exit")
            else:
                return img
        else:
            return Image.open(fn).convert('RGB')

    def _get_inst(self, idx, load_rgb=False, randshift=False):
        modal, bbox, category, imgfn, _ = self.data_reader.get_instance(idx)
        centerx = bbox[0] + bbox[2] / 2.
        centery = bbox[1] + bbox[3] / 2.
        size = max([np.sqrt(bbox[2] * bbox[3] * self.enlarge_box), bbox[2] * 1.1, bbox[3] * 1.1])
        if size < 5 or np.all(modal == 0):
            return self._get_inst(
                np.random.choice(len(self)), load_rgb=load_rgb, randshift=randshift)

        # shift & scale aug
        if self.mode  == 'overlap':
            if randshift:
                centerx += np.random.uniform(*self.shift) * size
                centery += np.random.uniform(*self.shift) * size
            size /= np.random.uniform(*self.scale)
        
        # crop
        new_bbox = [int(centerx - size / 2.), int(centery - size / 2.), int(size), int(size)]
        modal = cv2.resize(utils.crop_padding(modal, new_bbox, pad_value=(0,)),
            (self.sz, self.sz), interpolation=cv2.INTER_NEAREST)
        bbox_new = utils.mask_to_bbox(modal)
        if load_rgb:
            rgb = np.array(self._load_image(os.path.join(
                self.config['{}_image_root'.format(self.phase)], imgfn))) # uint8
            rgb = cv2.resize(utils.crop_padding(rgb, new_bbox, pad_value=(0,0,0)),
                (self.sz, self.sz), interpolation=cv2.INTER_CUBIC)
            if flip:
                rgb = rgb[:, ::-1, :]
            rgb = torch.from_numpy(rgb.astype(np.float32).transpose((2, 0, 1)) / 255.)
            rgb = self.img_transform(rgb) # CHW

        if load_rgb:
            return modal, category, rgb, bbox_new
        else:
            return modal, category, None, bbox_new

    def __getitem__(self, idx):

        randidx = np.random.choice(len(self))
        modal, category, rgb,_ = self._get_inst(idx=idx, load_rgb=False, randshift=True) # modal, uint8 {0, 1}
        category = 1

        eraser, _, _,_ = self._get_inst(randidx, load_rgb=False, randshift=False)
        eraser = self.eraser_setter(modal, eraser) # uint8 {0, 1}

        # erase
        erased_modal = modal.copy().astype(np.float32)
        if np.random.rand() < self.eraser_front_prob:
            erased_modal[eraser == 1] = 0 # eraser above modal
        else:
            eraser[modal == 1] = 0 # eraser below modal
        erased_modal = erased_modal * category

        target_bbox = utils.mask_to_bbox(modal)
        erased_bbox = utils.mask_to_bbox(erased_modal)
        eraser_bbox = utils.mask_to_bbox(eraser)

        erased_mask = np.zeros((self.sz,self.sz)).astype(np.float32)
        erased_mask[erased_bbox[1]:erased_bbox[1]+erased_bbox[3]+1,erased_bbox[0]:erased_bbox[0]+erased_bbox[2]+1] = 1

        eraser_mask = np.zeros((self.sz,self.sz)).astype(np.float32)
        eraser_mask[eraser_bbox[1]:eraser_bbox[1]+eraser_bbox[3]+1,eraser_bbox[0]:eraser_bbox[0]+eraser_bbox[2]+1] = 1
        
        target_mask = np.zeros((self.sz,self.sz)).astype(np.float32)
        target_mask[target_bbox[1]:target_bbox[1]+target_bbox[3]+1, target_bbox[0]:target_bbox[0]+target_bbox[2]+1] = 1
 
        eraser_tensor = torch.from_numpy(eraser_mask.astype(np.float32)).unsqueeze(0) # 1HW
        # erase rgb
        if rgb is not None:
            rgb = rgb * (1 - eraser_tensor)
        else:
            rgb = torch.zeros((3, self.sz, self.sz), dtype=torch.float32) # 3HW
        erased_modal_tensor = torch.from_numpy(erased_mask.astype(np.float32)).unsqueeze(0) # 1HW
        target = torch.from_numpy(target_mask.astype(np.int)) # HW
        return rgb, erased_modal_tensor, eraser_tensor, target

