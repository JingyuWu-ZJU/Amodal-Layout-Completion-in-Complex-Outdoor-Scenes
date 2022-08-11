import os
import numpy as np
import cvbase as cvb
import cv2
from pycocotools.coco import COCO
import pycocotools.mask as mask_utils
import pdb
import copy
from torch.utils.data import Dataset
from torchvision import transforms
import tqdm
from torch.utils.data import DataLoader
from PIL import Image
import utils
import torch
import imageio
import random
import reader
from PIL import ImageDraw
import layout_painting

#KINS origial json loader
def ori_make_json_dict(imgs, anns):
    imgs_dict = {}
    anns_dict = {}
    for ann in anns:
        image_id = ann["image_id"]
        if not image_id in anns_dict:
            anns_dict[image_id] = []
            anns_dict[image_id].append(ann)
        else:
            anns_dict[image_id].append(ann)

    for img in imgs:
        image_id = img['id']
        imgs_dict[image_id] = img['file_name']

    return imgs_dict, anns_dict

#not using category
def make_json_dict(imgs, anns):
    imgs_dict = {}
    inmodal_anns_dict = {}
    amodal_anns_dict = {}
    image = {}
    all_inmodal_dict = {}
    all_amodal_dict = {}
    data_len = 0
    for ann in tqdm.tqdm(anns):
        image_id = ann["image_id"]
        oco_id = ann["oco_id"]
        if not image_id in inmodal_anns_dict:
            inmodal_anns_dict[image_id] = {}
            amodal_anns_dict[image_id] = {}
        
            inmodal_anns_dict[image_id][oco_id] = []
            amodal_anns_dict[image_id][oco_id] = []
     

            inmodal_anns_dict[image_id][oco_id].append(ann["i_bbox"])
            amodal_anns_dict[image_id][oco_id].append(ann['a_bbox'])

        else:
            if not oco_id in inmodal_anns_dict[image_id]:
                inmodal_anns_dict[image_id][oco_id] = []
                amodal_anns_dict[image_id][oco_id] = []
          
            inmodal_anns_dict[image_id][oco_id].append(ann["i_bbox"])
            amodal_anns_dict[image_id][oco_id].append(ann['a_bbox'])

    for image in inmodal_anns_dict:
        for oco in inmodal_anns_dict[image]:
            if len(inmodal_anns_dict[image][oco]) > 1:
                all_inmodal_dict[data_len] = {image:inmodal_anns_dict[image][oco]}
                all_amodal_dict[data_len] = {image:amodal_anns_dict[image][oco]}
                data_len = data_len + 1
    for img in imgs:
        image_id = img['id']
        imgs_dict[image_id] = img['file_name']
    return imgs_dict, all_inmodal_dict, all_amodal_dict

# add category
def new_make_json_dict(imgs, anns):
    ### add category
    imgs_dict = {}
    inmodal_anns_dict = {}
    amodal_anns_dict = {}
    category_anns_dict = {}
    image = {}
    all_inmodal_dict = {}
    all_amodal_dict = {}
    all_category_dict = {}
    data_len = 0
  
    for ann in tqdm.tqdm(anns):
        image_id = ann["image_id"]
        oco_id = ann["oco_id"]
        if not image_id in inmodal_anns_dict:
            inmodal_anns_dict[image_id] = {}
            amodal_anns_dict[image_id] = {}
            category_anns_dict[image_id] = {}

            inmodal_anns_dict[image_id][oco_id] = []
            amodal_anns_dict[image_id][oco_id] = []
            category_anns_dict[image_id][oco_id] = []

            inmodal_anns_dict[image_id][oco_id].append(ann["i_bbox"])
            amodal_anns_dict[image_id][oco_id].append(ann['a_bbox'])
            category_anns_dict[image_id][oco_id].append(ann['category_id'])

        else:
            if not oco_id in inmodal_anns_dict[image_id]:
                inmodal_anns_dict[image_id][oco_id] = []
                amodal_anns_dict[image_id][oco_id] = []
                category_anns_dict[image_id][oco_id] = []
          
            inmodal_anns_dict[image_id][oco_id].append(ann["i_bbox"])
            amodal_anns_dict[image_id][oco_id].append(ann['a_bbox'])
            category_anns_dict[image_id][oco_id].append(ann['category_id'])

    for image in inmodal_anns_dict:
        for oco in inmodal_anns_dict[image]:
            if len(inmodal_anns_dict[image][oco]) > 1:
                all_inmodal_dict[data_len] = {image:inmodal_anns_dict[image][oco]}
                all_amodal_dict[data_len] = {image:amodal_anns_dict[image][oco]}
                all_category_dict[data_len] = {image:category_anns_dict[image][oco]}
                data_len = data_len + 1

    for img in imgs:
        image_id = img['id']
        imgs_dict[image_id] = img['file_name']
    return imgs_dict, all_inmodal_dict, all_amodal_dict, all_category_dict

def divide_and_conquer(imgs,anns):
    imgs_dict, anns_dict = ori_make_json_dict(imgs, anns)
    Groups = {}
    number = 0
    for imgid in tqdm.tqdm(imgs_dict):
        modal_bbox_list = []
        amodal_bbox_list = []
        category_list = []
        group = {}
        group_num = 0
        
        for objectid in anns_dict[imgid]:
            modal_bbox_list.append(objectid['inmodal_bbox'])
            amodal_bbox_list.append(objectid['bbox'])
            category_list.append(objectid['category_id'])
        # print(modal_bbox_list)
        modal = []
        amodal = []
        category = []
        for i in range(len(modal_bbox_list)):
            modaltemp = []
            amodaltemp = []
            catetemp = []
            modaltemp.append(modal_bbox_list[i])
            amodaltemp.append(amodal_bbox_list[i])
            catetemp.append(category_list[i])
            for j in range(i+1,len(modal_bbox_list)):
                bbox1 = modal_bbox_list[i]
                bbox2 = modal_bbox_list[j]
                x,y,w,h = bbox2
            
                if bbox_iou(np.array(bbox1),np.array(bbox2)) or bbox_iou(np.array(bbox1),np.array([x-1,y,w,h])) or  bbox_iou(np.array(bbox1),np.array([x,y-1,w,h])) or bbox_iou(np.array(bbox1),np.array([x,y,w+1,h])) or bbox_iou(np.array(bbox1),np.array([x,y,w,h+1])):
                    modaltemp.append(modal_bbox_list[j])
                    amodaltemp.append(amodal_bbox_list[j])
                    catetemp.append(category_list[j])
            modal.append(modaltemp)
            amodal.append(amodaltemp)
            category.append(catetemp)
        for k in range(len(modal)):
            if modal[k] == []:
                continue
            for j in range(len(modal)):
                flag = 0
                if j == k:
                    continue
                for bbox_2 in modal[j]:
                    if bbox_2 in modal[k]:
                        flag = 1
                if flag == 1:
                    for bbox_2 in modal[j]:
                        if bbox_2 not in modal[k]:
                            modal[k].append(bbox_2)  
                    modal[j] = []
                    for l in range(len(amodal[j])):
                        if amodal[j][l] not in amodal[k]:
                            amodal[k].append(amodal[j][l])
                            category[k].append(category[j][l])
                    amodal[j] = []
                    category[j] = []
                    break 

        for m in range(len(modal)):
            if modal[m] == [] or len(modal[m]) == 1:
                continue  
            Groups[number] = {'image_id':imgid,'category': category[m],'modal_bbox':modal[m],'amodal_bbox':amodal[m]}   
            number = number + 1 
    with open('divide-and-conquer-result.json','w+') as file1:
        json.dump(Groups,file1)

def read_dac(imgs,anns,path = 'divide-and-conquer-result.json'):
    imgs_dict, _ = ori_make_json_dict(imgs, anns)
    anns = cvb.load(path)
    anns_amodal_dict = {}
    anns_inmodal_dict = {}
    anns_category_dict = {}
    for ann in tqdm.tqdm(anns):
        anns_inmodal_dict[int(ann)] = {anns[ann]['image_id']: anns[ann]['modal_bbox']}
        anns_amodal_dict[int(ann)] = {anns[ann]['image_id']: anns[ann]['amodal_bbox']}
        anns_category_dict[int(ann)] = {anns[ann]['image_id']: anns[ann]['category']}
    return  imgs_dict, anns_inmodal_dict, anns_amodal_dict, anns_category_dict

#input 6,256,256 for test
class KINSloader_6256256(Dataset):
    def __init__(self,dataset,annfile_path,image_path):
        self.dataset = dataset
        self.image_path = image_path
        anns = cvb.load(annfile_path)
        self.imgs_info = anns['images']
        self.anns_info = anns["annotations"]
        self.anns_amodal_dict = {}
        self.anns_inmodal_dict = {}
        self.anns_category_dict = {}
        self.imgs_dict, self.anns_inmodal_dict, self.anns_amodal_dict = make_json_dict(self.imgs_info, self.anns_info)
 
    def __len__(self):
        return len(self.anns_inmodal_dict)

    def __getitem__(self, idx): 
        # all the same 
        for image_id in self.anns_inmodal_dict[idx]:
            img_fn = self.imgs_dict[image_id]
            inmodal_bboxes = self.anns_inmodal_dict[idx][image_id]
            amodal_bboxes = self.anns_amodal_dict[idx][image_id]
        image = Image.open(self.image_path+img_fn).convert('RGB')
        W, H = image.size
        minx,miny,maxx,maxy = inmodal_bboxes[0]
        maxx,maxy = 0,0
        if len(inmodal_bboxes)>6:
            inmodal_bboxes = inmodal_bboxes[:6]
            amodal_bboxes = amodal_bboxes[:6]
        # #changing 2 
        source_mask = np.zeros((6,H,W)).astype(np.float32)
        target_mask = np.zeros((6,H,W)).astype(np.float32)
        eraser_mask = np.zeros((6,H,W)).astype(np.float32)
        for i in range(len(inmodal_bboxes)):
            source_mask[i,inmodal_bboxes[i][1]:inmodal_bboxes[i][1]+inmodal_bboxes[i][3]+1,inmodal_bboxes[i][0]:inmodal_bboxes[i][0]+inmodal_bboxes[i][2]+1] = 1
            target_mask[i,amodal_bboxes[i][1]:amodal_bboxes[i][1]+amodal_bboxes[i][3]+1,amodal_bboxes[i][0]:amodal_bboxes[i][0]+amodal_bboxes[i][2]+1] = 1
            minx = inmodal_bboxes[i][0] if minx>inmodal_bboxes[i][0]    else minx
            miny = inmodal_bboxes[i][1] if miny>inmodal_bboxes[i][1]    else miny
            maxx = inmodal_bboxes[i][0]+inmodal_bboxes[i][2] if maxx<(inmodal_bboxes[i][0]+inmodal_bboxes[i][2])    else maxx
            maxy = inmodal_bboxes[i][1]+inmodal_bboxes[i][3] if maxy<(inmodal_bboxes[i][1]+inmodal_bboxes[i][3])    else maxy
            for j in range(len(inmodal_bboxes)):
                if i == j:
                    continue
                else:
                    eraser_mask[i,inmodal_bboxes[j][1]:inmodal_bboxes[j][1]+inmodal_bboxes[j][3]+1,inmodal_bboxes[j][0]:inmodal_bboxes[j][0]+inmodal_bboxes[j][2]+1] = 1


        new_bbox = [minx,miny,maxx-minx,maxy-miny]
        size = max([np.sqrt(new_bbox[2] * new_bbox[3] * 2.), new_bbox[2] * 1.1, new_bbox[3] * 1.1])
        center_x, center_y  = new_bbox[0] + new_bbox[2]/2, new_bbox[1] + new_bbox[3]/2
        new_bbox[2], new_bbox[3] = size, size
        new_bbox[0], new_bbox[1] = center_x - size/2, center_y - size/2

        source_mask = cv2.resize(utils.crop_padding(source_mask.transpose((1,2,0)), new_bbox, pad_value=(0,0,0,0,0,0)),(256, 256), interpolation=cv2.INTER_NEAREST)
        target_mask = cv2.resize(utils.crop_padding(target_mask.transpose((1,2,0)), new_bbox, pad_value=(0,0,0,0,0,0)),(256, 256), interpolation=cv2.INTER_NEAREST)
        eraser_mask = cv2.resize(utils.crop_padding(eraser_mask.transpose((1,2,0)), new_bbox, pad_value=(0,0,0,0,0,0)),(256, 256), interpolation=cv2.INTER_NEAREST)
        
        # temp = source_mask.reshape(-1, 6).argmax(axis=1).reshape(256, 256)
        # imageio.imsave('temp.jpg',temp)
        source_mask_tensor = torch.from_numpy(source_mask.transpose((2,0,1)).astype(np.float32))
        target_mask_tensor = torch.from_numpy(target_mask.transpose((2,0,1)).astype(np.int64))
        eraser_mask_tensor = torch.from_numpy(eraser_mask.transpose((2,0,1)).astype(np.int64))
       
        return source_mask_tensor,target_mask_tensor, eraser_mask_tensor

#self-supervised learning dataloader
class KINSloader_self_supervised(Dataset):
    def __init__(self,dataset,annfile_path,image_path):
        self.dataset = dataset
        self.image_path = image_path
        anns = cvb.load(annfile_path)
        self.imgs_info = anns['images']
        self.anns_info = anns["annotations"]
        self.data_reader = reader.KINSLVISDataset(self.dataset, annfile_path)
        config = {}
        config['min_overlap'] = 0.3
        config['max_overlap'] = 0.8
        config['min_cut_ratio'] = 0.001
        config['max_cut_ratio'] = 0.9
        self.eraser_setter = utils.EraserSetter(config)
        self.eraser_front_prob = 0.9
        self.phase = 'train'
        self.size = 256


    def __len__(self):
        return self.data_reader.get_instance_length()

    def _get_inst(self, idx,randshift=False):
        modal, bbox, category, imgfn, _ = self.data_reader.get_instance(idx)
        image = Image.open(self.image_path+imgfn).convert('RGB')
        W, H = image.size
        if bbox[2]*bbox[3]/(W*H)<0.01:
            flag = 1
        else:
            flag = 0
        centerx = bbox[0] + bbox[2] / 2.
        centery = bbox[1] + bbox[3] / 2.
        size = max([np.sqrt(bbox[2] * bbox[3] * 4.), bbox[2] * 2.5, bbox[3] * 2.5])
        
        if size < 5 or np.all(modal == 0):
            return self._get_inst(
                np.random.choice(len(self)), randshift=randshift)
        # shift & scale aug
        if self.phase  == 'train':
            if randshift:
                centerx += np.random.uniform(-0.2, 0.2) * size
                centery += np.random.uniform(-0.2, 0.2) * size
            size /= np.random.uniform(0.8, 1.2)
        # crop
        new_bbox = [int(centerx - size / 2.), int(centery - size / 2.), int(size), int(size)]
        modal = cv2.resize(utils.crop_padding(modal, new_bbox, pad_value=(0,)),
            (self.size, self.size), interpolation=cv2.INTER_NEAREST) 

        return modal, category,flag

    def __getitem__(self, idx): 
        modal, category,flag = self._get_inst(idx, randshift=True) # modal, uint8 {0, 1}

        target_bbox = utils.mask_to_bbox(modal)
        target_mask = np.zeros((self.size,self.size)).astype(np.float32)
    
        target_mask[target_bbox[1]:target_bbox[1]+target_bbox[3], target_bbox[0]:target_bbox[0]+target_bbox[2]] = 1

        source_mask = np.zeros((self.size,self.size)).astype(np.float32)
        eraser_mask = np.zeros((self.size,self.size)).astype(np.float32)
        
        random_num = np.random.randint(1,3)
        for i in range(random_num):
        # while 1:
            randidx = np.random.choice(len(self))
            eraser, _ ,_= self._get_inst(randidx, randshift=False)
            eraser = self.eraser_setter(modal, eraser) # uint8 {0, 1}
            eraser_bbox = utils.mask_to_bbox(eraser)
            eraser_mask[eraser_bbox[1]:eraser_bbox[1]+eraser_bbox[3],eraser_bbox[0]:eraser_bbox[0]+eraser_bbox[2]] = 1
            # erase
            erased_modal = modal.copy().astype(np.float32)
            if np.random.rand() < self.eraser_front_prob:
                erased_modal[eraser == 1] = 0 # eraser above modal
            else:
                eraser[modal == 1] = 0 # eraser below modal
            erased_modal = erased_modal * 1

        source_bbox = utils.mask_to_bbox(erased_modal)
        # if source_bbox == target_bbox:
        #     continue
        # else:
        #     break

        source_mask[source_bbox[1]:source_bbox[1]+source_bbox[3],source_bbox[0]:source_bbox[0]+source_bbox[2]] = 1     
        eraser_tensor = torch.from_numpy(eraser_mask.astype(np.float32)).unsqueeze(0) # 1HW
        source_tensor = torch.from_numpy(source_mask.astype(np.float32)).unsqueeze(0) # 1HW
        target_tensor = torch.from_numpy(target_mask.astype(np.int64)) # HW
        source_category = torch.tensor(category).view(1)
        flag = torch.tensor(flag)
        return source_tensor, target_tensor, eraser_tensor, source_category,flag

#supervised learning dataloader using source category
class KINSloader(Dataset):
    def __init__(self,dataset,annfile_path,image_path):
        self.dataset = dataset
        self.image_path = image_path
        anns = cvb.load(annfile_path)
        self.imgs_info = anns['images']
        self.anns_info = anns["annotations"]
        self.anns_amodal_dict = {}
        self.anns_inmodal_dict = {}
        self.imgs_dict, self.anns_inmodal_dict, self.anns_amodal_dict, self.anns_category_dict = new_make_json_dict(self.imgs_info, self.anns_info)
     
    def __len__(self):
        return len(self.anns_inmodal_dict)

    def __getitem__(self, idx): 
        # all the same 
        for image_id in self.anns_inmodal_dict[idx]:
            img_fn = self.imgs_dict[image_id]
            inmodal_bboxes = self.anns_inmodal_dict[idx][image_id]
            amodal_bboxes = self.anns_amodal_dict[idx][image_id]
            categories = self.anns_category_dict[idx][image_id]

        image = Image.open(self.image_path+img_fn).convert('RGB')
        W, H = image.size
        minx,miny,maxx,maxy = inmodal_bboxes[0]
        maxx,maxy = 0,0
        if len(inmodal_bboxes)>6:
            inmodal_bboxes = inmodal_bboxes[:6]
            amodal_bboxes = amodal_bboxes[:6]
            categories = categories[:6]
        #changing 1
        source_mask = np.zeros((H,W)).astype(np.float32)
        target_mask = np.zeros((H,W)).astype(np.float32)
        eraser_mask = np.zeros((H,W)).astype(np.float32)

        randidx = random.randint(1, len(inmodal_bboxes)-1)
        source_mask[inmodal_bboxes[randidx][1]:inmodal_bboxes[randidx][1]+inmodal_bboxes[randidx][3],
                    inmodal_bboxes[randidx][0]:inmodal_bboxes[randidx][0]+inmodal_bboxes[randidx][2]] = 1
        target_mask[amodal_bboxes[randidx][1]:amodal_bboxes[randidx][1]+amodal_bboxes[randidx][3],
                    amodal_bboxes[randidx][0]:amodal_bboxes[randidx][0]+amodal_bboxes[randidx][2]] = 1
        source_category = torch.tensor(categories[randidx]).view(1)
        eraser_category = []
        for i in range(len(inmodal_bboxes)):
            minx = inmodal_bboxes[i][0] if minx>inmodal_bboxes[i][0]    else minx
            miny = inmodal_bboxes[i][1] if miny>inmodal_bboxes[i][1]    else miny
            maxx = inmodal_bboxes[i][0]+inmodal_bboxes[i][2] if maxx<(inmodal_bboxes[i][0]+inmodal_bboxes[i][2])    else maxx
            maxy = inmodal_bboxes[i][1]+inmodal_bboxes[i][3] if maxy<(inmodal_bboxes[i][1]+inmodal_bboxes[i][3])    else maxy
            if i == randidx:
                continue
            eraser_mask[inmodal_bboxes[i][1]:inmodal_bboxes[i][1]+inmodal_bboxes[i][3],
                        inmodal_bboxes[i][0]:inmodal_bboxes[i][0]+inmodal_bboxes[i][2]] = 1
            # eraser_category.append(categories[i])

        new_bbox = [minx,miny,maxx-minx,maxy-miny]
        size = max([np.sqrt(new_bbox[2] * new_bbox[3] * 2.5), new_bbox[2] * 1.1, new_bbox[3] * 1.1])
        center_x, center_y  = new_bbox[0] + new_bbox[2]/2, new_bbox[1] + new_bbox[3]/2
        new_bbox[2], new_bbox[3] = size, size
        new_bbox[0], new_bbox[1] = center_x - size/2, center_y - size/2

        source_mask = cv2.resize(utils.crop_padding(source_mask, new_bbox, pad_value=(0,)),(256, 256), interpolation=cv2.INTER_NEAREST)
        target_mask = cv2.resize(utils.crop_padding(target_mask, new_bbox, pad_value=(0,)),(256, 256), interpolation=cv2.INTER_NEAREST)
        eraser_mask = cv2.resize(utils.crop_padding(eraser_mask, new_bbox, pad_value=(0,)),(256, 256), interpolation=cv2.INTER_NEAREST)
        source_mask_tensor = torch.from_numpy(source_mask.astype(np.float32)).unsqueeze(0)
        target_mask_tensor = torch.from_numpy(target_mask.astype(np.int64))
        eraser_mask_tensor = torch.from_numpy(eraser_mask.astype(np.int64)).unsqueeze(0)

        return source_mask_tensor, target_mask_tensor, eraser_mask_tensor, source_category

#supervised learning dataloader using all category
class KINSloader_category(Dataset):
    def __init__(self,dataset,annfile_path,image_path):
        self.dataset = dataset
        self.image_path = image_path
        anns = cvb.load(annfile_path)
        self.imgs_info = anns['images']
        self.anns_info = anns["annotations"]
        self.anns_amodal_dict = {}
        self.anns_inmodal_dict = {}
        self.imgs_dict, self.anns_inmodal_dict, self.anns_amodal_dict, self.anns_category_dict = read_dac(imgs = self.imgs_info,anns = self.anns_info)
        
        anns2 = cvb.load('data/KINS/update_train_2020.json')    
        self.imgs_info2 = anns2['images']
        self.anns_info2 = anns2["annotations"]
        self.imgs_dict, self.anns_inmodal_dict, self.anns_amodal_dict, self.anns_category_dict = new_make_json_dict(self.imgs_info2, self.anns_info2)


    def __len__(self):
        return len(self.anns_inmodal_dict)

    def __getitem__(self, idx): 
        # all the same 
        for image_id in self.anns_inmodal_dict[idx]:
            img_fn = self.imgs_dict[image_id]
            inmodal_bboxes = self.anns_inmodal_dict[idx][image_id]
            amodal_bboxes = self.anns_amodal_dict[idx][image_id]
            categories = self.anns_category_dict[idx][image_id]
   
        image = Image.open(self.image_path+img_fn).convert('RGB')
        W, H = image.size
        minx,miny,maxx,maxy = inmodal_bboxes[0]
        maxx,maxy = 0,0
        if len(inmodal_bboxes)>6:
            inmodal_bboxes = inmodal_bboxes[:6]
            amodal_bboxes = amodal_bboxes[:6]
            categories = categories[:6]
        #changing 1
        source_mask = np.zeros((H,W)).astype(np.float32)
        target_mask = np.zeros((H,W)).astype(np.float32)
        eraser_mask = np.zeros((H,W)).astype(np.float32)

        randidx = random.randint(1, len(inmodal_bboxes)-1)
        source_mask[inmodal_bboxes[randidx][1]:inmodal_bboxes[randidx][1]+inmodal_bboxes[randidx][3],
                    inmodal_bboxes[randidx][0]:inmodal_bboxes[randidx][0]+inmodal_bboxes[randidx][2]] = 1
        target_mask[amodal_bboxes[randidx][1]:amodal_bboxes[randidx][1]+amodal_bboxes[randidx][3],
                    amodal_bboxes[randidx][0]:amodal_bboxes[randidx][0]+amodal_bboxes[randidx][2]] = 1
        source_category = torch.tensor(categories[randidx]).view(1)
        eraser_category = []
        for i in range(len(inmodal_bboxes)):
            minx = inmodal_bboxes[i][0] if minx>inmodal_bboxes[i][0]    else minx
            miny = inmodal_bboxes[i][1] if miny>inmodal_bboxes[i][1]    else miny
            maxx = inmodal_bboxes[i][0]+inmodal_bboxes[i][2] if maxx<(inmodal_bboxes[i][0]+inmodal_bboxes[i][2])    else maxx
            maxy = inmodal_bboxes[i][1]+inmodal_bboxes[i][3] if maxy<(inmodal_bboxes[i][1]+inmodal_bboxes[i][3])    else maxy
            if i == randidx:
                continue
            eraser_mask[inmodal_bboxes[i][1]:inmodal_bboxes[i][1]+inmodal_bboxes[i][3],
                        inmodal_bboxes[i][0]:inmodal_bboxes[i][0]+inmodal_bboxes[i][2]] = 1
            eraser_category.append(categories[i])

        for _ in range(len(eraser_category),5):
            eraser_category.append(0)
        new_bbox = [minx,miny,maxx-minx,maxy-miny]
        size = max([np.sqrt(new_bbox[2] * new_bbox[3] * 2.5), new_bbox[2] * 1.1, new_bbox[3] * 1.1])
        center_x, center_y  = new_bbox[0] + new_bbox[2]/2, new_bbox[1] + new_bbox[3]/2
        new_bbox[2], new_bbox[3] = size, size
        new_bbox[0], new_bbox[1] = center_x - size/2, center_y - size/2

        source_mask = cv2.resize(utils.crop_padding(source_mask, new_bbox, pad_value=(0,)),(256, 256), interpolation=cv2.INTER_NEAREST)
        target_mask = cv2.resize(utils.crop_padding(target_mask, new_bbox, pad_value=(0,)),(256, 256), interpolation=cv2.INTER_NEAREST)
        eraser_mask = cv2.resize(utils.crop_padding(eraser_mask, new_bbox, pad_value=(0,)),(256, 256), interpolation=cv2.INTER_NEAREST)
        
        
        source_mask_tensor = torch.from_numpy(source_mask.astype(np.float32)).unsqueeze(0)
        target_mask_tensor = torch.from_numpy(target_mask.astype(np.int64))
        eraser_mask_tensor = torch.from_numpy(eraser_mask.astype(np.int64)).unsqueeze(0)

        eraser_category = torch.tensor(eraser_category)
        return source_mask_tensor, target_mask_tensor, eraser_mask_tensor, source_category, eraser_category

#supervised learning dataloader save pic
class KINSloader_savepic(Dataset):
    def __init__(self,dataset,annfile_path,image_path):
        self.dataset = dataset
        self.image_path = image_path
        anns = cvb.load(annfile_path)
        self.imgs_info = anns['images']
        self.anns_info = anns["annotations"]
        self.anns_amodal_dict = {}
        self.anns_inmodal_dict = {}
        self.imgs_dict, self.anns_inmodal_dict, self.anns_amodal_dict, self.anns_category_dict = new_make_json_dict(self.imgs_info, self.anns_info)
        self.transform_img = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((256,256))
            ])
    def __len__(self):
        return len(self.anns_inmodal_dict)

    def __getitem__(self, idx): 
        # all the same 
        random.seed(1)
        for image_id in self.anns_inmodal_dict[idx]:
            img_fn = self.imgs_dict[image_id]
            inmodal_bboxes = self.anns_inmodal_dict[idx][image_id]
            amodal_bboxes = self.anns_amodal_dict[idx][image_id]
            categories = self.anns_category_dict[idx][image_id]
        # print(img_fn)
        random.seed(1)
        image = Image.open(self.image_path+img_fn).convert('RGB')
        image_ori = Image.open(self.image_path+img_fn).convert('RGB')
        image_2 = Image.open(self.image_path+img_fn).convert('RGB')
        image_3 = Image.open(self.image_path+img_fn).convert('RGB')

        W, H = image.size
        minx,miny,maxx,maxy = inmodal_bboxes[0]
        maxx,maxy = 0,0
        if len(inmodal_bboxes)>6:
            inmodal_bboxes = inmodal_bboxes[:6]
            amodal_bboxes = amodal_bboxes[:6]
            categories = categories[:6]
        #changing 1
        source_mask = np.zeros((H,W)).astype(np.float32)
        target_mask = np.zeros((H,W)).astype(np.float32)
        eraser_mask = np.zeros((H,W)).astype(np.float32)
        
        layout_pic = Image.new('RGB', (W, H), (255, 255, 255))
        layout_2 = Image.new('RGB', (W, H), (255, 255, 255))

        a = ImageDraw.ImageDraw(image)
        b = ImageDraw.ImageDraw(layout_pic)
        c = ImageDraw.ImageDraw(image_2)
        d = ImageDraw.ImageDraw(image_3)
        e = ImageDraw.ImageDraw(layout_2)

        randidx = random.randint(1, len(inmodal_bboxes)-1)
        source_mask[inmodal_bboxes[randidx][1]:inmodal_bboxes[randidx][1]+inmodal_bboxes[randidx][3],
                    inmodal_bboxes[randidx][0]:inmodal_bboxes[randidx][0]+inmodal_bboxes[randidx][2]] = 1
        target_mask[amodal_bboxes[randidx][1]:amodal_bboxes[randidx][1]+amodal_bboxes[randidx][3],
                    amodal_bboxes[randidx][0]:amodal_bboxes[randidx][0]+amodal_bboxes[randidx][2]] = 1
        
        a.rectangle((inmodal_bboxes[randidx][0],inmodal_bboxes[randidx][1],inmodal_bboxes[randidx][0]+inmodal_bboxes[randidx][2],
                    inmodal_bboxes[randidx][1]+inmodal_bboxes[randidx][3]),outline='gold',width=2)
        b.rectangle((inmodal_bboxes[randidx][0],inmodal_bboxes[randidx][1],inmodal_bboxes[randidx][0]+inmodal_bboxes[randidx][2],
                    inmodal_bboxes[randidx][1]+inmodal_bboxes[randidx][3]),outline='gold',width=2)
        # b.rectangle((amodal_bboxes[randidx][0],amodal_bboxes[randidx][1],amodal_bboxes[randidx][0]+amodal_bboxes[randidx][2],
        #             amodal_bboxes[randidx][1]+amodal_bboxes[randidx][3]),outline='red',width=2)   
        c.rectangle((amodal_bboxes[randidx][0],amodal_bboxes[randidx][1],amodal_bboxes[randidx][0]+amodal_bboxes[randidx][2],
                    amodal_bboxes[randidx][1]+amodal_bboxes[randidx][3]),outline='red',width=3)    
        c.rectangle((inmodal_bboxes[randidx][0],inmodal_bboxes[randidx][1],inmodal_bboxes[randidx][0]+inmodal_bboxes[randidx][2],
                    inmodal_bboxes[randidx][1]+inmodal_bboxes[randidx][3]),outline='gold',width=5)
        e.rectangle((amodal_bboxes[randidx][0],amodal_bboxes[randidx][1],amodal_bboxes[randidx][0]+amodal_bboxes[randidx][2],
                    amodal_bboxes[randidx][1]+amodal_bboxes[randidx][3]),outline='red',width=3)  
        source_category = torch.tensor(categories[randidx]).view(1)
        eraser_category = []
        for i in range(len(inmodal_bboxes)):
            minx = inmodal_bboxes[i][0] if minx>inmodal_bboxes[i][0]    else minx
            miny = inmodal_bboxes[i][1] if miny>inmodal_bboxes[i][1]    else miny
            maxx = inmodal_bboxes[i][0]+inmodal_bboxes[i][2] if maxx<(inmodal_bboxes[i][0]+inmodal_bboxes[i][2])    else maxx
            maxy = inmodal_bboxes[i][1]+inmodal_bboxes[i][3] if maxy<(inmodal_bboxes[i][1]+inmodal_bboxes[i][3])    else maxy
            if i == randidx:
                continue
            eraser_mask[inmodal_bboxes[i][1]:inmodal_bboxes[i][1]+inmodal_bboxes[i][3],
                        inmodal_bboxes[i][0]:inmodal_bboxes[i][0]+inmodal_bboxes[i][2]] = 1
            eraser_category.append(categories[i])
            # a.rectangle((inmodal_bboxes[i][0],inmodal_bboxes[i][1],inmodal_bboxes[i][0]+inmodal_bboxes[i][2],
            #         inmodal_bboxes[i][1]+inmodal_bboxes[i][3]),outline='lime',width=5)
            b.rectangle((inmodal_bboxes[i][0],inmodal_bboxes[i][1],inmodal_bboxes[i][0]+inmodal_bboxes[i][2],
                    inmodal_bboxes[i][1]+inmodal_bboxes[i][3]),outline='lime',width=2)
            # c.rectangle((inmodal_bboxes[i][0],inmodal_bboxes[i][1],inmodal_bboxes[i][0]+inmodal_bboxes[i][2],
            #         inmodal_bboxes[i][1]+inmodal_bboxes[i][3]),outline='green',width=2)
            d.rectangle((inmodal_bboxes[i][0],inmodal_bboxes[i][1],inmodal_bboxes[i][0]+inmodal_bboxes[i][2],
                    inmodal_bboxes[i][1]+inmodal_bboxes[i][3]),outline='lime',width=5)       
            # e.rectangle((inmodal_bboxes[i][0],inmodal_bboxes[i][1],inmodal_bboxes[i][0]+inmodal_bboxes[i][2],
            #         inmodal_bboxes[i][1]+inmodal_bboxes[i][3]),outline='green',width=2) 
        for _ in range(len(eraser_category),5):
            eraser_category.append(0)
        new_bbox = [minx,miny,maxx-minx,maxy-miny]
        size = max([np.sqrt(new_bbox[2] * new_bbox[3] * 2.5), new_bbox[2] * 1.1, new_bbox[3] * 1.1])
        if size < 50:
            print('repeat')
            self.__getitem__(random.randint(1,len(self.anns_inmodal_dict)))
        center_x, center_y  = new_bbox[0] + new_bbox[2]/2, new_bbox[1] + new_bbox[3]/2
        new_bbox[2], new_bbox[3] = size, size
        new_bbox[0], new_bbox[1] = center_x - size/2, center_y - size/2

        source_mask = cv2.resize(utils.crop_padding(source_mask, new_bbox, pad_value=(0,)),(256, 256), interpolation=cv2.INTER_NEAREST)
        target_mask = cv2.resize(utils.crop_padding(target_mask, new_bbox, pad_value=(0,)),(256, 256), interpolation=cv2.INTER_NEAREST)
        eraser_mask = cv2.resize(utils.crop_padding(eraser_mask, new_bbox, pad_value=(0,)),(256, 256), interpolation=cv2.INTER_NEAREST)

        source_mask_tensor = torch.from_numpy(source_mask.astype(np.float32)).unsqueeze(0)
        target_mask_tensor = torch.from_numpy(target_mask.astype(np.int64))
        eraser_mask_tensor = torch.from_numpy(eraser_mask.astype(np.int64)).unsqueeze(0)

        eraser_category = torch.tensor(eraser_category)
        
        image = image.crop((new_bbox[0],new_bbox[1],new_bbox[0]+new_bbox[2],new_bbox[1]+new_bbox[3]))
        layout_pic = layout_pic.crop((new_bbox[0],new_bbox[1],new_bbox[0]+new_bbox[2],new_bbox[1]+new_bbox[3]))
        image_ori = image_ori.crop((new_bbox[0],new_bbox[1],new_bbox[0]+new_bbox[2],new_bbox[1]+new_bbox[3]))
        image_2 = image_2.crop((new_bbox[0],new_bbox[1],new_bbox[0]+new_bbox[2],new_bbox[1]+new_bbox[3]))
        image_3 = image_3.crop((new_bbox[0],new_bbox[1],new_bbox[0]+new_bbox[2],new_bbox[1]+new_bbox[3]))
        layout_2 = layout_2.crop((new_bbox[0],new_bbox[1],new_bbox[0]+new_bbox[2],new_bbox[1]+new_bbox[3]))

        image = self.transform_img(image)
        layout_pic = self.transform_img(layout_pic)
        image_ori = self.transform_img(image_ori)
        image_2 = self.transform_img(image_2)
        image_3 = self.transform_img(image_3)
        layout_2 = self.transform_img(layout_2)

        # new_bbox = torch.tensor(new_bbox)
        return source_mask_tensor, target_mask_tensor, eraser_mask_tensor, source_category, eraser_category, image, layout_pic,image_ori,image_2,image_3,layout_2
       

#supervised learning dataloader with rgb
class KINSloader_rgb(Dataset):
    def __init__(self,dataset,annfile_path,image_path):
        self.dataset = dataset
        self.image_path = image_path
        anns = cvb.load(annfile_path)
        self.imgs_info = anns['images']
        self.anns_info = anns["annotations"]
        self.anns_amodal_dict = {}
        self.anns_inmodal_dict = {}
        self.imgs_dict, self.anns_inmodal_dict, self.anns_amodal_dict, self.anns_category_dict = new_make_json_dict(self.imgs_info, self.anns_info)
        self.transform_img = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((256,256))
            ])
    def __len__(self):
        return len(self.anns_inmodal_dict)

    def __getitem__(self, idx): 
        # all the same 
        random.seed(1)
        for image_id in self.anns_inmodal_dict[idx]:
            img_fn = self.imgs_dict[image_id]
            inmodal_bboxes = self.anns_inmodal_dict[idx][image_id]
            amodal_bboxes = self.anns_amodal_dict[idx][image_id]
            categories = self.anns_category_dict[idx][image_id]
   
        image = Image.open(self.image_path+img_fn).convert('RGB')
        W, H = image.size
        minx,miny,maxx,maxy = inmodal_bboxes[0]
        maxx,maxy = 0,0
        if len(inmodal_bboxes)>6:
            inmodal_bboxes = inmodal_bboxes[:6]
            amodal_bboxes = amodal_bboxes[:6]
            categories = categories[:6]
        #changing 1
        source_mask = np.zeros((H,W)).astype(np.float32)
        target_mask = np.zeros((H,W)).astype(np.float32)
        eraser_mask = np.zeros((H,W)).astype(np.float32)

        randidx = random.randint(1, len(inmodal_bboxes)-1)
    
        area = (inmodal_bboxes[randidx][2] * inmodal_bboxes[randidx][3])/(W*H)
        if area>0.0 and area<0.01 :
            flag = torch.tensor([1])
        else:
            flag = torch.tensor([0])
        source_mask[inmodal_bboxes[randidx][1]:inmodal_bboxes[randidx][1]+inmodal_bboxes[randidx][3],
                    inmodal_bboxes[randidx][0]:inmodal_bboxes[randidx][0]+inmodal_bboxes[randidx][2]] = 1
        target_mask[amodal_bboxes[randidx][1]:amodal_bboxes[randidx][1]+amodal_bboxes[randidx][3],
                    amodal_bboxes[randidx][0]:amodal_bboxes[randidx][0]+amodal_bboxes[randidx][2]] = 1
        source_category = torch.tensor(categories[randidx]).view(1)
        eraser_category = []
        for i in range(len(inmodal_bboxes)):
            minx = inmodal_bboxes[i][0] if minx>inmodal_bboxes[i][0]    else minx
            miny = inmodal_bboxes[i][1] if miny>inmodal_bboxes[i][1]    else miny
            maxx = inmodal_bboxes[i][0]+inmodal_bboxes[i][2] if maxx<(inmodal_bboxes[i][0]+inmodal_bboxes[i][2])    else maxx
            maxy = inmodal_bboxes[i][1]+inmodal_bboxes[i][3] if maxy<(inmodal_bboxes[i][1]+inmodal_bboxes[i][3])    else maxy
            if i == randidx:
                continue
            eraser_mask[inmodal_bboxes[i][1]:inmodal_bboxes[i][1]+inmodal_bboxes[i][3],
                        inmodal_bboxes[i][0]:inmodal_bboxes[i][0]+inmodal_bboxes[i][2]] = 1
            eraser_category.append(categories[i])

        for _ in range(len(eraser_category),5):
            eraser_category.append(0)
        new_bbox = [minx,miny,maxx-minx,maxy-miny]
        size = max([np.sqrt(new_bbox[2] * new_bbox[3] * 2.5), new_bbox[2] * 1.1, new_bbox[3] * 1.1])
        center_x, center_y  = new_bbox[0] + new_bbox[2]/2, new_bbox[1] + new_bbox[3]/2
        new_bbox[2], new_bbox[3] = size, size
        new_bbox[0], new_bbox[1] = center_x - size/2, center_y - size/2

        source_mask = cv2.resize(utils.crop_padding(source_mask, new_bbox, pad_value=(0,)),(256, 256), interpolation=cv2.INTER_NEAREST)
        target_mask = cv2.resize(utils.crop_padding(target_mask, new_bbox, pad_value=(0,)),(256, 256), interpolation=cv2.INTER_NEAREST)
        eraser_mask = cv2.resize(utils.crop_padding(eraser_mask, new_bbox, pad_value=(0,)),(256, 256), interpolation=cv2.INTER_NEAREST)
        image_ori = image.crop((new_bbox[0],new_bbox[1],new_bbox[0]+new_bbox[2],new_bbox[1]+new_bbox[3]))
        
        source_mask_tensor = torch.from_numpy(source_mask.astype(np.float32)).unsqueeze(0)
        target_mask_tensor = torch.from_numpy(target_mask.astype(np.int64))
        eraser_mask_tensor = torch.from_numpy(eraser_mask.astype(np.int64)).unsqueeze(0)
        
        eraser_category = torch.tensor(eraser_category)
        image_ori = self.transform_img(image_ori)
        return source_mask_tensor, target_mask_tensor, eraser_mask_tensor, source_category, eraser_category, image_ori, flag

#using all layout to train
class KINSloader_alllayout(Dataset):
    def __init__(self,dataset,annfile_path,image_path):
        self.dataset = dataset
        self.image_path = image_path
        anns = cvb.load(annfile_path)
        self.imgs_info = anns['images']
        self.anns_info = anns["annotations"]
        self.anns_amodal_dict = {}
        self.anns_inmodal_dict = {}
        self.imgs_dict, self.anns_inmodal_dict, self.anns_amodal_dict, self.anns_category_dict = new_make_json_dict(self.imgs_info, self.anns_info)
        self.transform_img = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((256,256))
            ])
    def __len__(self):
        return len(self.anns_inmodal_dict)

    def __getitem__(self, idx): 
        # all the same 
        save_img_path = 'data/KINS-yolo/test/'+ str(idx) + '.png'
        save_txt_path = 'data/KINS-yolo/test/'+ str(idx) + '.txt'
        for image_id in self.anns_inmodal_dict[idx]:
            img_fn = self.imgs_dict[image_id]
            inmodal_bboxes = self.anns_inmodal_dict[idx][image_id]
            amodal_bboxes = self.anns_amodal_dict[idx][image_id]
            categories = self.anns_category_dict[idx][image_id]
   
        image = Image.open(self.image_path+img_fn).convert('RGB')
        W, H = image.size
        minx,miny,maxx,maxy = inmodal_bboxes[0]
        maxx,maxy = 0,0
        if len(inmodal_bboxes)>6:
            inmodal_bboxes = inmodal_bboxes[:6]
            amodal_bboxes = amodal_bboxes[:6]
            categories = categories[:6]

        randidx = random.randint(1, len(inmodal_bboxes)-1)

        for i in range(len(inmodal_bboxes)):
            minx = inmodal_bboxes[i][0] if minx>inmodal_bboxes[i][0]    else minx
            miny = inmodal_bboxes[i][1] if miny>inmodal_bboxes[i][1]    else miny
            maxx = inmodal_bboxes[i][0]+inmodal_bboxes[i][2] if maxx<(inmodal_bboxes[i][0]+inmodal_bboxes[i][2])    else maxx
            maxy = inmodal_bboxes[i][1]+inmodal_bboxes[i][3] if maxy<(inmodal_bboxes[i][1]+inmodal_bboxes[i][3])    else maxy
  
        new_bbox = [minx,miny,maxx-minx,maxy-miny]
        size = max([np.sqrt(new_bbox[2] * new_bbox[3] * 2.5), new_bbox[2] * 1.1, new_bbox[3] * 1.1])
        center_x, center_y  = new_bbox[0] + new_bbox[2]/2, new_bbox[1] + new_bbox[3]/2
        new_bbox[2], new_bbox[3] = size, size
        new_bbox[0], new_bbox[1] = center_x - size/2, center_y - size/2

        bbox_256 = []
        for j in range(len(inmodal_bboxes)):
            eraser_mask = np.zeros((H,W)).astype(np.float32)
            eraser_mask[inmodal_bboxes[j][1]:inmodal_bboxes[j][1]+inmodal_bboxes[j][3],
                        inmodal_bboxes[j][0]:inmodal_bboxes[j][0]+inmodal_bboxes[j][2]] = 1
            eraser_mask = cv2.resize(utils.crop_padding(eraser_mask, new_bbox, pad_value=(0,)),(256, 256), interpolation=cv2.INTER_NEAREST)
            bbox_temp = utils.mask_to_bbox(eraser_mask)
            bbox_256.append(bbox_temp)
        
        # eraser_mask = np.zeros((H,W)).astype(np.float32)
        # eraser_mask[amodal_bboxes[randidx][1]:amodal_bboxes[randidx][1]+amodal_bboxes[randidx][3],
        #             amodal_bboxes[randidx][0]:amodal_bboxes[randidx][0]+amodal_bboxes[randidx][2]] = 1
        # eraser_mask = cv2.resize(utils.crop_padding(eraser_mask, new_bbox, pad_value=(0,)),(256, 256), interpolation=cv2.INTER_NEAREST)
        # bbox_temp = utils.mask_to_bbox(eraser_mask)
        # bbox_256.append(bbox_temp)
        image_ori = image.crop((new_bbox[0],new_bbox[1],new_bbox[0]+new_bbox[2],new_bbox[1]+new_bbox[3]))
        image_ori = image_ori.resize((256,256))
        # imageio.imsave(save_img_path,image_ori)
        with open(save_txt_path,'w+') as file1:
            for k in range(len(bbox_256)):
                x,y,w,h = bbox_256[k][0]/256,bbox_256[k][1]/256,bbox_256[k][2]/256,bbox_256[k][3]/256
                c_x,c_y,w,h = x+(w/2), y+(h/2),w,h
                file1.write(str(categories[k]-1) + ' '+str(c_x)+' '+str(c_y)+' '+str(w)+' '+str(h)+'\n')
        # with open('data/KINS-yolo/val.txt','a') as file2:
        #     file2.write(save_img_path+'\n')
        

        return 1
        # return source_mask_tensor, target_mask_tensor, eraser_mask_tensor, source_category, eraser_category, image_ori

#supervise learning using all complete layout
class KINSloader_complete_layout(Dataset):
    def __init__(self,dataset,annfile_path,image_path):
        self.dataset = dataset
        self.image_path = image_path
        anns = cvb.load('data/KINS/instances_train.json')
        self.imgs_info = anns['images']
        self.anns_info = anns["annotations"]
        self.anns_amodal_dict = {}
        self.anns_inmodal_dict = {}
        self.anns_dict = {}
        self.imgs_dict, self.anns_dict = ori_make_json_dict(self.imgs_info, self.anns_info)
        self.img_id = []
        for img_id in self.imgs_dict.keys():
            self.img_id.append(img_id)

    def __len__(self):
        return len(self.img_id)

    def __getitem__(self, idx): 
        # all the same 
        img_id = self.img_id[idx]
        img_name = self.imgs_dict[img_id]
        categories = []
        inmodal_bboxes = []
        amodal_bboxes = [] 
        image = Image.open(self.image_path+img_name).convert('RGB')
        W, H = image.size
        anns = self.anns_dict[img_id]
        for ann in anns:
            categories.append(ann['category_id'])
            inmodal_bboxes.append(ann['inmodal_bbox'])
            amodal_bboxes.append(ann['bbox'])
        # print(categories)
        if len(inmodal_bboxes)>6:
            inmodal_bboxes = inmodal_bboxes[:6]
            amodal_bboxes = amodal_bboxes[:6]
            categories = categories[:6]
        
        source_mask = np.zeros((H,W)).astype(np.float32)
        target_mask = np.zeros((H,W)).astype(np.float32)
        eraser_mask = np.zeros((H,W)).astype(np.float32)
        
        if len(inmodal_bboxes) == 1:
            randidx = 0
        else:
            randidx = random.randint(1, len(inmodal_bboxes)-1)
        source_mask[inmodal_bboxes[randidx][1]:inmodal_bboxes[randidx][1]+inmodal_bboxes[randidx][3],
                    inmodal_bboxes[randidx][0]:inmodal_bboxes[randidx][0]+inmodal_bboxes[randidx][2]] = 1
        target_mask[amodal_bboxes[randidx][1]:amodal_bboxes[randidx][1]+amodal_bboxes[randidx][3],
                    amodal_bboxes[randidx][0]:amodal_bboxes[randidx][0]+amodal_bboxes[randidx][2]] = 1
        area = (inmodal_bboxes[randidx][2] * inmodal_bboxes[randidx][3])/(W*H)
        if area>0.0 and area<0.01 :
            flag = torch.tensor([1])
        else:
            flag = torch.tensor([0])
        source_category = torch.tensor(categories[randidx]).view(1)
        eraser_category = []

        for i in range(len(inmodal_bboxes)):
            if i == randidx:
                continue
            eraser_mask[inmodal_bboxes[i][1]:inmodal_bboxes[i][1]+inmodal_bboxes[i][3],
                        inmodal_bboxes[i][0]:inmodal_bboxes[i][0]+inmodal_bboxes[i][2]] = 1
            eraser_category.append(categories[i])
        for _ in range(len(eraser_category),5):
            eraser_category.append(0)
        source_mask = cv2.resize(source_mask,(256, 256), interpolation=cv2.INTER_NEAREST)
        target_mask = cv2.resize(target_mask,(256, 256), interpolation=cv2.INTER_NEAREST)
        eraser_mask = cv2.resize(eraser_mask,(256, 256), interpolation=cv2.INTER_NEAREST)

        source_mask_tensor = torch.from_numpy(source_mask.astype(np.float32)).unsqueeze(0)
        target_mask_tensor = torch.from_numpy(target_mask.astype(np.int64))
        eraser_mask_tensor = torch.from_numpy(eraser_mask.astype(np.int64)).unsqueeze(0)
        eraser_category = torch.tensor(eraser_category)
        return source_mask_tensor, target_mask_tensor, eraser_mask_tensor, source_category, eraser_category,flag

class KINSloader_lostgan(Dataset):
    def __init__(self,dataset,annfile_path,image_path):
        self.dataset = dataset
        self.image_path = image_path
        anns = cvb.load(annfile_path)
        self.imgs_info = anns['images']
        self.anns_info = anns["annotations"]
        self.anns_amodal_dict = {}
        self.anns_inmodal_dict = {}
        self.imgs_dict, self.anns_inmodal_dict, self.anns_amodal_dict, self.anns_category_dict = new_make_json_dict(self.imgs_info, self.anns_info)
        self.transform_img = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((256,256))
            ])
    def __len__(self):
        return len(self.anns_inmodal_dict)

    def __getitem__(self, idx): 
        # all the same        
        save_img_path = 'data/KINS-yolo/test/'+ str(idx) + '.png'
        save_txt_path = 'data/KINS-yolo/test/'+ str(idx) + '.txt'
        for image_id in self.anns_inmodal_dict[idx]:
            img_fn = self.imgs_dict[image_id]
            inmodal_bboxes = self.anns_inmodal_dict[idx][image_id]
            amodal_bboxes = self.anns_amodal_dict[idx][image_id]
            categories = self.anns_category_dict[idx][image_id]
    
        image = Image.open(self.image_path+img_fn).convert('RGB')
        W, H = image.size
        minx,miny,maxx,maxy = inmodal_bboxes[0]
        maxx,maxy = 0,0
        if len(inmodal_bboxes)>6:
            inmodal_bboxes = inmodal_bboxes[:6]
            amodal_bboxes = amodal_bboxes[:6]
            categories = categories[:6]

        randidx = random.randint(1, len(inmodal_bboxes)-1)

        for i in range(len(inmodal_bboxes)):
            minx = inmodal_bboxes[i][0] if minx>inmodal_bboxes[i][0]    else minx
            miny = inmodal_bboxes[i][1] if miny>inmodal_bboxes[i][1]    else miny
            maxx = inmodal_bboxes[i][0]+inmodal_bboxes[i][2] if maxx<(inmodal_bboxes[i][0]+inmodal_bboxes[i][2])    else maxx
            maxy = inmodal_bboxes[i][1]+inmodal_bboxes[i][3] if maxy<(inmodal_bboxes[i][1]+inmodal_bboxes[i][3])    else maxy
  
        new_bbox = [minx,miny,maxx-minx,maxy-miny]
        size = max([np.sqrt(new_bbox[2] * new_bbox[3] * 2.5), new_bbox[2] * 1.1, new_bbox[3] * 1.1])
        center_x, center_y  = new_bbox[0] + new_bbox[2]/2, new_bbox[1] + new_bbox[3]/2
        new_bbox[2], new_bbox[3] = size, size
        new_bbox[0], new_bbox[1] = center_x - size/2, center_y - size/2

        bbox_256 = []
        for j in range(len(inmodal_bboxes)):
            eraser_mask = np.zeros((H,W)).astype(np.float32)
            eraser_mask[inmodal_bboxes[j][1]:inmodal_bboxes[j][1]+inmodal_bboxes[j][3],
                        inmodal_bboxes[j][0]:inmodal_bboxes[j][0]+inmodal_bboxes[j][2]] = 1
            eraser_mask = cv2.resize(utils.crop_padding(eraser_mask, new_bbox, pad_value=(0,)),(256, 256), interpolation=cv2.INTER_NEAREST)
            bbox_temp = utils.mask_to_bbox(eraser_mask)
            bbox_256.append(bbox_temp)

        # eraser_mask = np.zeros((H,W)).astype(np.float32)
        # eraser_mask[amodal_bboxes[randidx][1]:amodal_bboxes[randidx][1]+amodal_bboxes[randidx][3],
        #             amodal_bboxes[randidx][0]:amodal_bboxes[randidx][0]+amodal_bboxes[randidx][2]] = 1
        # eraser_mask = cv2.resize(utils.crop_padding(eraser_mask, new_bbox, pad_value=(0,)),(256, 256), interpolation=cv2.INTER_NEAREST)
        # bbox_temp = utils.mask_to_bbox(eraser_mask)
        # bbox_256.append(bbox_temp)
        # image_ori = image.crop((new_bbox[0],new_bbox[1],new_bbox[0]+new_bbox[2],new_bbox[1]+new_bbox[3]))
        # image_ori = image_ori.resize((256,256))
        # imageio.imsave(save_img_path,image_ori)
        # with open(save_txt_path,'w+') as file1:
        #     for k in range(len(bbox_256)):
        #         x,y,w,h = bbox_256[k][0]/256,bbox_256[k][1]/256,bbox_256[k][2]/256,bbox_256[k][3]/256
        #         c_x,c_y,w,h = x+(w/2), y+(h/2),w,h
        #         file1.write(str(categories[k]-1) + ' '+str(c_x)+' '+str(c_y)+' '+str(w)+' '+str(h)+'\n')
        # with open('data/KINS-yolo/val.txt','a') as file2:
        #     file2.write(save_img_path+'\n')
        return categories,bbox_256


