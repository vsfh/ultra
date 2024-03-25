import os
from os.path import join as opj
import numpy as np
import json
import cv2
from math import cos, sin
from scipy.spatial.transform import Rotation as R
import shutil
import glob
if __name__=='__main__':
    bboxes = []
    def get_bbox(txt_path, img_path):
        arr = open(txt_path,'r').readlines()
        for tarr in arr:
            ys = np.array(tarr.split(), dtype=np.float32)[2:][::2]
            xs = np.array(tarr.split(), dtype=np.float32)[1:][::2]
            bboxes.append((xs.min(),ys.min(),xs.max(),ys.max()))
        bboxes_arr = np.array(bboxes)
        img = cv2.imread(img_path)
        w, h = img.shape[:2]
        bbox = (int(bboxes_arr[:,0].min()*h),int(bboxes_arr[:,1].min()*w),int(bboxes_arr[:,2].max()*h),int(bboxes_arr[:,3].max()*w))
        return bbox, img
    path = '/home/vsfh/data/xiaoyapian/images/*.jpg'
    for name in glob.glob(path):
        bbox, img = get_bbox(name, name.replace('jpg','txt'))
        json.dump(open(name.replace('jpg','json').replace('images','19')))
    # img = cv2.rectangle(img,bbox[:2],bbox[-2:],color=255,thickness=5)
    # cv2.imwrite('a.jpg', img)
    pass