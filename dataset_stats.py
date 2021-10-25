import argparse
import datetime
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, DistributedSampler

import datasets
import util.misc as utils
from datasets import build_dataset, get_coco_api_from_dataset

from PIL import Image, ImageFont, ImageDraw
import torchvision.transforms.functional as F

mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

def denorm(tensor):
    for t, m, s in zip(tensor, mean, std):
        t.mul_(s).add_(m)
        # The normalize code -> t.sub_(m).div_(s)
    return tensor
    
def normalized_tensor_to_PIL(img):
    return F.to_pil_image(denorm(img))
    
def showimg(img, trg):
    cols, rows = img.size
    draw = ImageDraw.Draw(img)
    
    for i in trg["boxes"]:
        cx, cy, w, h = i
        cx, w = cx * cols, w * cols
        cy, h = cy * rows, h * rows
        draw.rectangle(((int(cx - w/2), int(cy - h/2)), (int(cx + w/2), int(cy + h/2))), outline ="red")
     
    img.show()

VAL_LABELS = {
        "noobject": 0,      #0  
        "person": 0,        #1  
        "bird": 0,          #2
        "cat": 0,           #3
        "cow": 0,           #4
        "dog": 0,           #5  
        "horse": 0,         #6
        "sheep": 0,         #7
        "aeroplane": 0,     #8    
        "bicycle": 0,       #9
        "boat": 0,          #10
        "bus": 0,           #11
        "car": 0,           #12 
        "motorbike": 0,     #13   
        "train": 0,         #14 
        "bottle": 0,        #15
        "chair": 0,         #16 
        "diningtable": 0,   #17 
        "pottedplant": 0,   #18 
        "sofa": 0,          #19 
        "tvmonitor": 0      #20
        }
        
TRAIN_LABELS = {
        "noobject": 0,      #0  
        "person": 0,        #1  
        "bird": 0,          #2
        "cat": 0,           #3
        "cow": 0,           #4
        "dog": 0,           #5  
        "horse": 0,         #6
        "sheep": 0,         #7
        "aeroplane": 0,     #8    
        "bicycle": 0,       #9
        "boat": 0,          #10
        "bus": 0,           #11
        "car": 0,           #12 
        "motorbike": 0,     #13   
        "train": 0,         #14 
        "bottle": 0,        #15
        "chair": 0,         #16 
        "diningtable": 0,   #17 
        "pottedplant": 0,   #18 
        "sofa": 0,          #19 
        "tvmonitor": 0      #20
        }
        
LABELS_LIST = list(VAL_LABELS)

def get_args_parser():
    parser = argparse.ArgumentParser('Dataset statistics', add_help=False)

    # dataset parameters
    parser.add_argument('--dataset_file', default='coco')
    parser.add_argument('--coco_path', type=str)
    parser.add_argument('--coco_panoptic_path', type=str)
    parser.add_argument('--remove_difficult', action='store_true')
    parser.add_argument('--masks', action='store_true',
                        help="Train segmentation head if the flag is provided")

    return parser
   

def main(args):
    print('\nBUILDING DATASETS\n')

    dataset_train = build_dataset(image_set='train', args=args)
    dataset_val = build_dataset(image_set='val', args=args)
    
    print('\nDONE BUILDING DATASETS\n')

    print(f'Train dataset length: {dataset_train.__len__()}')
    print(f'Val dataset length: {dataset_val.__len__()}\n')
    
    # VAL DATASET STATS
    
    max_density = -1
    min_density = 10000
    
    for i in range(dataset_val.__len__()):
        img, trg = dataset_val.__getitem__(i)
        
        for j in range(len(trg["labels"])):
            VAL_LABELS[LABELS_LIST[int(trg["labels"][j])]] += 1
    
    max_density, min_density, max_id, min_id = dataset_val.__getdensity__()
    
    print(f'Val dataset max density: {max_density} and min density: {min_density}\n')
    
    dataset_val.__showitem__(max_id)
    dataset_val.__showitem__(min_id)
    
    print('Val dataset class instances:')
    for i in range(1, len(VAL_LABELS)):
        print(f'{LABELS_LIST[i]} : {VAL_LABELS[LABELS_LIST[i]]}')
        
    max_img, max_trg = dataset_val.__getitem__(max_id)
    min_img, min_trg = dataset_val.__getitem__(min_id)
    
    showimg(normalized_tensor_to_PIL(max_img), max_trg)
    showimg(normalized_tensor_to_PIL(min_img), min_trg)
        
    # TRAIN DATASET STATS

    max_density = -1
    min_density = 10000
    
    for i in range(dataset_train.__len__()):
        img, trg = dataset_train.__getitem__(i)
        
        for j in range(len(trg["labels"])):
            TRAIN_LABELS[LABELS_LIST[int(trg["labels"][j])]] += 1
    
    max_density, min_density, max_id, min_id = dataset_train.__getdensity__()
    
    print(f'\nTrain dataset max density: {max_density} and min density: {min_density}\n')
    
    dataset_train.__showitem__(max_id)
    dataset_train.__showitem__(min_id)
    
    print('Train dataset class instances:')
    for i in range(1, len(VAL_LABELS)):
        print(f'{LABELS_LIST[i]} : {TRAIN_LABELS[LABELS_LIST[i]]}')
        
    max_img, max_trg = dataset_train.__getitem__(max_id)
    min_img, min_trg = dataset_train.__getitem__(min_id)
    
    showimg(normalized_tensor_to_PIL(max_img), max_trg)
    showimg(normalized_tensor_to_PIL(min_img), min_trg)
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser('Pascal VOC dataset statistics', parents=[get_args_parser()])
    args = parser.parse_args()
    
    main(args)