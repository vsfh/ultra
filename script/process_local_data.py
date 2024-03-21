import shutil
import os
import glob
from natsort import natsorted
import cv2
import numpy as np

def make_sub_dataset():
    for i in range(20):
        folder_name = f'/home/vsfh/data/cls/image/train/{i:02d}'
        os.makedirs(folder_name.replace('train','train_sub'),exist_ok=True)
        img_list = natsorted(glob.glob(f'{folder_name}/*.jpg'))
        for j in range(300):
            src = img_list[j]
            shutil.copy(src, src.replace('train','train_sub'))
            
def rot_04():
    folder_name = '/home/vsfh/data/cls/image/train_sub/04'
    for file in glob.glob(f'{folder_name}/*.jpg'):
        img = np.array(cv2.imread(file))
        flip_img = np.flipud(img)
        cv2.imwrite(file.replace('04','04_rot'), flip_img)
        
rot_04()
            
        