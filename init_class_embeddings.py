import argparse
import datetime
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

import util.misc as utils
from models.backbone import build_backbone

from PIL import Image
import torchvision.transforms as T
import requests
from util.misc import NestedTensor, nested_tensor_from_tensor_list
import glob
import os


def get_args_parser():
    parser = argparse.ArgumentParser('Set embedding backbone', add_help=False) 
    parser.add_argument('--lr_backbone', default=0, type=float)
    parser.add_argument('--batch_size', default=1, type=int)

    # * Backbone
    parser.add_argument('--backbone', default='resnet50', type=str,
                        help="Name of the convolutional backbone to use")
    parser.add_argument('--dilation', action='store_true',
                        help="If true, we replace stride with dilation in the last convolutional block (DC5)")
    parser.add_argument('--position_embedding', default='sine', type=str, choices=('sine', 'learned'),
                        help="Type of positional embedding to use on top of the image features")

    # * Transformer
    parser.add_argument('--hidden_dim', default=256, type=int,
                        help="Size of the embeddings (dimension of the transformer)")
    parser.add_argument('--nheads', default=8, type=int,
                        help="Number of attention heads inside the transformer's attentions")
    parser.add_argument('--num_queries', default=100, type=int,
                        help="Number of query slots")

    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')    

    return parser
    

def main(args):
    print(args)

    device = torch.device(args.device)

    backbone = build_backbone(args)
    backbone.eval()
    
    sample_transform = T.Compose([
        T.Resize((480,480)),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])  
    
    reverse_preprocess = T.Compose([
        T.ToPILImage(),
    ])
    
    # average pooling for class sample
    avgpool = nn.AdaptiveAvgPool2d((1, 1))
    
    # align the backbone channels to hidden dim
    sample2query = nn.Linear(backbone.num_channels, args.hidden_dim)
    
    samples = []
    
    for filename in glob.glob('class_embeddings/*.jpg'):
        image = Image.open(filename)
        samples.append(image)
        
        # only for verification
        # image.show()
        # trans = sample_transform(image)
        # samples.append(trans.to(device).unsqueeze(0))
        # image = reverse_preprocess(trans)
        # image.show()
    
    samples = torch.stack([sample_transform(sample) for sample in samples], dim=0).unsqueeze(0)
    print(samples.shape)
    
    samples = samples.repeat(args.batch_size, 1, 1, 1, 1).contiguous()
    print(samples.shape)
    
    bs, num_embs, _, _, _ = samples.shape
    print(bs, num_embs)
    
    samples = samples.flatten(0, 1)
    print(samples.shape)
    
    sample_feature = backbone(samples)
    print(avgpool(sample_feature[-1]).shape)
    
    sample_feature_gt = avgpool(sample_feature[-1]).flatten(1)
    print(sample_feature_gt.shape)
    
    # (40, 2048)->(40, 256)->(2, 20, 256)->(2, 100, 256)->(100, 2, 256)
    sample_feature = sample2query(sample_feature_gt) \
            .view(bs, num_embs, -1) \
            .repeat_interleave(args.num_queries // num_embs, dim=1) \
            .permute(1, 0, 2) \
            .contiguous()

    print(sample_feature.shape)
    
    sample_feature = sample_feature.detach()
    torch.save(sample_feature, 'class_embeddings/class_embeddings_db')
    
    # for verification
    loaded = torch.load('class_embeddings/class_embeddings_db')
    
    print(loaded.shape)
   

if __name__ == '__main__':
    parser = argparse.ArgumentParser('DETR class embedding script', parents=[get_args_parser()])
    args = parser.parse_args()
    main(args)