import cv2
import os
import sys
import torch
import glob
from ultralytics.data.augment import LetterBox
import numpy as np
from natsort import natsorted
from PIL import Image
import torchvision.transforms as T
class ClassificationDataset(torch.utils.data.Dataset):
    def __init__(self, mode='train'):
        self.path = '/data/shenfeihong/classification/brace'
        if mode=='train':
            self.images = natsorted(glob.glob(os.path.join(self.path, '*', 'none','*.jpg')))[:300]+\
                        natsorted(glob.glob(os.path.join(self.path, '*', 'not_none','*.jpg')))[:300]
        else:
            self.images = natsorted(glob.glob(os.path.join(self.path, '*', 'none','*.jpg')))[300:]+\
                        natsorted(glob.glob(os.path.join(self.path, '*', 'not_none','*.jpg')))[300:]

        self.letterbox = LetterBox((640, 640))
        # self.transform = T.Compose([T.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5.)),T.ToTensor(),T.Normalize(mean=0, std=1)])
        self.transform = T.Compose([T.ToTensor(),T.Normalize(mean=0, std=1)])
        
    def __len__(self):
        return len(self.images)
    
    def _format_img(self, img):
        """Format the image for YOLO from Numpy array to PyTorch tensor."""
        if len(img.shape) < 3:
            img = np.expand_dims(img, -1)
        img = np.ascontiguousarray(img.transpose(2, 0, 1)[::-1])
        img = torch.from_numpy(img)
        return img
    
    def __getitem__(self, index, new_shape=(640,640), show=False):
        image_path = self.images[index]
        if 'not_none' in image_path:
            a = 1
        else:
            a = 0
        im = cv2.imread(image_path)
        im = self.letterbox(image=im)
        im = Image.fromarray(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
        im = self.transform(im)

        return {'img': im, 'cls': a}