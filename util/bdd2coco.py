import os
import json
import argparse
from tqdm import tqdm
import cv2
import shutil
import glob

TOTAL_SMALL = 0
TOTAL_MEDIUM = 0
TOTAL_LARGE = 0

parser = argparse.ArgumentParser(description='bdd2coco')
parser.add_argument('--bdd_dir', type=str, default='bdd100k')
cfg = parser.parse_args()

src_val_dir = os.path.join(cfg.bdd_dir, 'labels', 'bdd100k_labels_images_val.json')
src_train_dir = os.path.join(cfg.bdd_dir, 'labels', 'bdd100k_labels_images_train.json')

os.makedirs(os.path.join(cfg.bdd_dir, 'labels_coco'), exist_ok=True)

dst_val_dir = os.path.join(cfg.bdd_dir, 'labels_coco', 'bdd100k_val_subset_3_classes.json')
dst_train_dir = os.path.join(cfg.bdd_dir, 'labels_coco', 'bdd100k_train_subset_3_classes.json')


# subset imgs dir
images_dir = 'bdd100k\\images\\100k'

os.makedirs(os.path.join(images_dir, 'train2017'), exist_ok=True)
os.makedirs(os.path.join(images_dir, 'val2017'), exist_ok=True)

src_val_img_dir = os.path.join(images_dir, 'val')
src_train_img_dir = os.path.join(images_dir, 'train')

dst_val_img_dir = os.path.join(images_dir, 'val2017')
dst_train_img_dir = os.path.join(images_dir, 'train2017')


def bdd2coco_detection(dataset, labeled_images, save_dir):
  global TOTAL_SMALL
  global TOTAL_MEDIUM
  global TOTAL_LARGE
  
  # amount of s/m/l objects per image set (train or val)
  set_small = 0
  set_medium = 0
  set_large = 0
  
  # amount of images with ONLY s/m/l objects per image set
  set_only_small = 0 # val: 120        train: 933
  set_only_medium = 0 # val: 30         train: 183
  set_only_large = 0 # val: 20         train: 139
  
  # val and train dataset length
  if dataset == 'val':
    set_len = 20
    src_img_dir = src_val_img_dir
    dst_img_dir = dst_val_img_dir
  else:
    set_len = 100
    src_img_dir = src_train_img_dir
    dst_img_dir = dst_train_img_dir

  non_priority = ["person", "car", "traffic light", "traffic sign"]

  bdd_dict = {"categories":
    [
      {"supercategory": "N/A", "id": 1, "name": "pedestrian"},
      {"supercategory": "N/A", "id": 2, "name": "car"},
      {"supercategory": "N/A", "id": 3, "name": "rider"},
      # {"supercategory": "N/A", "id": 4, "name": "bus"},
      {"supercategory": "N/A", "id": 4, "name": "truck"},
      # {"supercategory": "N/A", "id": 6, "name": "bicycle"},
      # {"supercategory": "N/A", "id": 7, "name": "motorcycle"},
      # {"supercategory": "N/A", "id": 8, "name": "traffic light"},
      # {"supercategory": "N/A", "id": 9, "name": "traffic sign"},
      # {"supercategory": "N/A", "id": 10, "name": "train"},
    ]}
    
  coco_dict = {"categories":
    [
      {"supercategory": "N/A", "id": 1, "name": "person", "count" : 0},
      {"supercategory": "N/A", "id": 2, "name": "car", "count" : 0},
      # {"supercategory": "N/A", "id": 3, "name": "bus", "count" : 0},
      {"supercategory": "N/A", "id": 3, "name": "truck", "count" : 0},
      # {"supercategory": "N/A", "id": 5, "name": "bicycle", "count" : 0},
      # {"supercategory": "N/A", "id": 6, "name": "motorcycle", "count" : 0},
      # {"supercategory": "N/A", "id": 7, "name": "traffic light", "count" : 0},
      # {"supercategory": "N/A", "id": 8, "name": "traffic sign", "count" : 0},
      # {"supercategory": "N/A", "id": 9, "name": "train", "count" : 0},
    ]}

  bdd_label_dict = {i['name']: i['id'] for i in bdd_dict['categories']}
  coco_label_dict = {i['name']: i['id'] for i in coco_dict['categories']}
  coco_count_dict = {i['name']: i['count'] for i in coco_dict['categories']}
  subset_count_dict = {i['id']: [i['count'], i['name']] for i in coco_dict['categories']}

  images = list()
  annotations = list()
  ignore_categories = set()

  counter = 0
  for i in tqdm(labeled_images):
  
    # amount of s/m/l objects per current image
    img_small = 0
    img_medium = 0
    img_large = 0
    
    # ok = 0
    
    annotations_img = list()
    priority = 0
  
    counter += 1
    image = dict()
    image['file_name'] = i['name']
    image['height'] = 720
    image['width'] = 1280

    image['id'] = counter
    
    empty_image = True

    if 'labels' in i:
        for l in i['labels']:
          annotation = dict()
          if l['category'] in bdd_label_dict.keys():
            empty_image = False
            annotation["iscrowd"] = 0
            annotation["image_id"] = image['id']
            x1 = round(l['box2d']['x1'], 2)
            y1 = round(l['box2d']['y1'], 2)
            x2 = round(l['box2d']['x2'], 2)
            y2 = round(l['box2d']['y2'], 2)
            annotation['bbox'] = [x1, y1, round(x2 - x1, 2), round(y2 - y1, 2)]
            annotation['area'] = float((x2 - x1) * (y2 - y1))
            
            # count s/m/l objects
            if annotation['area'] <= 1024 and annotation['area'] >= 500:
                set_small += 1
                img_small += 1
            if annotation['area'] > 1024 and annotation['area'] < 9216:
                set_medium += 1
                img_medium += 1
            if annotation['area'] >= 9216:
                set_large += 1
                img_large += 1
            
            # merge pedestrian and rider into person category
            # otherwise keep the same category names
            if l['category'] == 'rider' or l['category'] == 'pedestrian':
                cat = 'person'
            else:
                cat = l['category']
               
            # # apply priorities based on object category counts
            # if coco_count_dict[cat] < 10:
                # priority += 1000
            # elif coco_count_dict[cat] > 20:
                # priority -= 1
                
            # # apply priorities based on object category
            # if cat in non_priority:
                # priority -= 10
                
            # if cat not in non_priority:
                # priority += 1
            if cat == 'truck':
                priority += 9
            elif cat == 'person':
                priority += 1
              
            annotation['category_id'] = coco_label_dict[cat]
            coco_count_dict[cat] += 1
                
            # annotation['ignore'] = 0
            annotation['id'] = int(l['id'])
            annotation['segmentation'] = [[x1, y1, x1, y2, x2, y2, x2, y1]]
            
            if annotation['area'] >= 500:
                annotations.append(annotation)
          else:
            ignore_categories.add(l['category'])

    if empty_image:
      continue
      
    # count images with only s/m/l objects
    if img_small > 0 and img_medium == 0 and img_large == 0:
        set_only_small += 1
        # ok = 1
        
    if img_small == 0 and img_medium > 0 and img_large == 0:
        set_only_medium += 1
        
    if img_small == 0 and img_medium == 0 and img_large > 0:
        set_only_large += 1
      
    image['priority'] = priority
    
    images.append(image)
    # annotations.extend(annotations_img)
    
    # if ok == 1:
    # if dataset == 'val' and len(images) < 20:
      # images.append(image)
      # annotations.extend(annotations_img)
    # if dataset == 'train' and len(images) < 100:
      # images.append(image)
      # annotations.extend(annotations_img)
  
  subset_small = subset_medium = subset_large = 0
  
  images = sorted(images, key = lambda i: i['priority'],reverse=True)
  
  priority_ids = [[image['id'], image['priority']] for image in images[:set_len]]
  print(f'\n priority_ids length: {len(priority_ids)}')
  # print(f' priority_ids (id, priority): {priority_ids}\n')
  priority_annotations = [annotation for annotation in annotations if annotation["image_id"] in [item[0] for item in priority_ids]] 
  
  for annotation in priority_annotations:
    subset_count_dict[annotation['category_id']][0] += 1
    
    if annotation['area'] <= 1024:
        subset_small += 1
    if annotation['area'] > 1024 and annotation['area'] < 9216:
        subset_medium += 1
    if annotation['area'] >= 9216:
        subset_large += 1

    
  #cleanup json
  for image in images:
    image.pop("priority")

  for dic in coco_dict["categories"]:
    dic.pop("count")

  coco_dict["images"] = images[:set_len]
  coco_dict["annotations"] = priority_annotations  
  # coco_dict["type"] = "instances"
  
  print(f" {'set_only_small':<15}={set_only_small:<7} {'set_only_medium':<15}={set_only_medium:<7} {'set_only_large':<15}={set_only_large:<7}")
  print(f" {'set_small':<15}={set_small:<7} {'set_medium':<15}={set_medium:<7} {'set_large':<15}={set_large:<7}")
  print(f" {'subset_small':<15}={subset_small:<7} {'subset_medium':<15}={subset_medium:<7} {'subset_large':<15}={subset_large:<7}")
  
  print(f"\n {'subset category count':<25}: {subset_count_dict.values()}")
  print(f" {'total category count':<25}: {coco_count_dict}\n")
  
  TOTAL_SMALL += set_small
  TOTAL_MEDIUM += set_medium
  TOTAL_LARGE += set_large

  print('ignored categories: ', ignore_categories)
  print('saving...')
  with open(save_dir, "w") as file:
    json.dump(coco_dict, file)
    
  # subset imgs dir
  print('\ncreating subset image directory')
  
  for image in coco_dict["images"]:
    img = os.path.join(src_img_dir,image['file_name'])
    shutil.copy(img, dst_img_dir)

  print('Done.')


def main():

  # create BDD validation set detections in COCO format
  print('Loading validation set...')
  with open(src_val_dir) as f:
    val_labels = json.load(f)
  print('Converting validation set...\n')
  bdd2coco_detection('val', val_labels, dst_val_dir)

  # create BDD training set detections in COCO format
  print('\nLoading training set...')
  with open(src_train_dir) as f:
    train_labels = json.load(f)
  print('Converting training set...\n')
  bdd2coco_detection('train', train_labels, dst_train_dir)
  
  print(f'\n total_small    = {TOTAL_SMALL}    total_medium    = {TOTAL_MEDIUM}    total_large    = {TOTAL_LARGE}\n')


if __name__ == '__main__':
  main()
