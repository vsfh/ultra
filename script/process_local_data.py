import shutil
import os
import glob
from natsort import natsorted
import cv2
import numpy as np
import sys
import json
sys.path.append('.')
sys.path.append('..')
from detect.dataset import smile_cls, face_cls
IMG_NUM = 1500
def make_sub_dataset():
    for i in range(20):
        print(i)
        folder_name = f'/home/vsfh/data/cls/image/train/{i:02d}'
        os.makedirs(folder_name.replace('train','train_sub'),exist_ok=True)
        img_list = natsorted(glob.glob(f'{folder_name}/*.jpg'))
        img_number = IMG_NUM // 5 if f'{i:02d}' in smile_cls or f'{i:02d}' in face_cls else IMG_NUM
        for j in range(img_number):
            src = img_list[j]
            shutil.copy(src, src.replace('train','train_sub'))
            
def flip_04():
    folder_name = '/home/vsfh/data/cls/image/train_sub/04'
    os.makedirs(folder_name.replace('/04','/04_flip'),exist_ok=True)
    os.makedirs('/mnt/hdd/data/cls/network_res/04_flip',exist_ok=True)
    
    for file in glob.glob(f'{folder_name}/*.jpg'):
        img = np.array(cv2.imread(file))
        img_name = file.split('/')[-1].split('.')[0]
        flip_img = np.flipud(img)
        cv2.imwrite(file.replace('/04/','/04_flip/'), flip_img)
        
        new_bbox = []
        bbox = json.load(open(f'/mnt/hdd/data/cls/network_res/04_ori/{img_name}.json'))['xyxy']
        h, w = img.shape[0], img.shape[1]
        new_bbox.append(max(0, bbox[0]))
        new_bbox.append(h - max(0, bbox[3]))
        new_bbox.append(min(w, bbox[2]))
        new_bbox.append(h - min(h, bbox[1]))
        new_json = {'xyxy':new_bbox}
        json.dump(new_json, open(f'/mnt/hdd/data/cls/network_res/04_flip/{img_name}.json', 'w'))

        
def flip_00():
    folder_name = '/home/vsfh/data/cls/image/train_sub/00'
    os.makedirs(folder_name.replace('/00','/00_flip'),exist_ok=True)
    os.makedirs('/mnt/hdd/data/cls/network_res/00_flip',exist_ok=True)
    
    for file in glob.glob(f'{folder_name}/*.jpg'):
        img = np.array(cv2.imread(file))
        img_name = file.split('/')[-1].split('.')[0]
        flip_img = np.fliplr(img)
        cv2.imwrite(file.replace('/00/','/00_flip/'), flip_img)
        
        new_bbox = []
        bbox = json.load(open(f'/mnt/hdd/data/cls/network_res/00/{img_name}.json'))['xyxy']
        h, w = img.shape[0], img.shape[1]
        new_bbox.append(w - max(0, bbox[2]))
        new_bbox.append(max(0, bbox[1]))
        new_bbox.append(w - min(w, bbox[0]))
        new_bbox.append(min(h, bbox[3]))
        new_json = {'xyxy':new_bbox}
        json.dump(new_json, open(f'/mnt/hdd/data/cls/network_res/00_flip/{img_name}.json', 'w'))
        
def remove_img_wo_label():
    label_list = [os.path.basename(a).replace('json', 'jpg') for a in glob.glob('/mnt/hdd/data/cls/network_res/*/*.json')]
    a = 0
    for i in range(20):
        folder_name = f'/home/vsfh/data/cls/image/train_sub/{i:02d}'
        img_list = natsorted(glob.glob(f'{folder_name}/*.jpg'))
        for src in img_list:
            if not os.path.basename(src) in label_list:
                a += 1
                print(src)
    print(a)
# make_sub_dataset()
flip_04()
flip_00()
# remove_img_wo_label()
            
        