
from itertools import repeat
from multiprocessing.pool import ThreadPool
from pathlib import Path

import cv2
import numpy as np
import torch


from ultralytics.utils import LOCAL_RANK, NUM_THREADS, TQDM, colorstr, is_dir_writeable
from ultralytics.utils.ops import resample_segments

from ultralytics.data.augment import Compose, Format, Instances, LetterBox, v8_transforms
from ultralytics.data.base import BaseDataset
from ultralytics.data.utils import HELP_URL, LOGGER, get_hash

DATASET_CACHE_VERSION = '1.0.3'

import glob
import math
import os
import random
from copy import deepcopy
from multiprocessing.pool import ThreadPool
from pathlib import Path
from typing import Any, Optional
import json
from scipy.spatial.transform import Rotation as R

import cv2
import numpy as np
import psutil
from torch.utils.data import Dataset

from ultralytics.utils import LOCAL_RANK, LOGGER, NUM_THREADS, TQDM

from ultralytics.data.utils import HELP_URL, IMG_FORMATS

from ultralytics.utils import yaml_load, IterableSimpleNamespace
from ultralytics.utils.ops import xyxy2xywh
from PIL import Image
from ultralytics.data.dataset import BaseDataset

smile_cls = ['05','07','10','13','15']
face_cls = ['06','08','11','14','16']
project = {
    '00':['ceph',2],
    '01':['bite',8],
    '02':['pano',1],
    '03':['upper',3],
    '04':['lower',4],
    '09':['right',5],
    '12':['front',6],
    '17':['left',7],
    '19':['small',0],
    '20':['f-ceph',11],
}

from ultralytics.utils.instance import Bboxes
class BboxesPose(Bboxes):
    def rot_90(self, M, angle):
        n = len(self.bboxes)
        new_bbox = np.zeros_like(self.bboxes)
        x1y1 = np.ones((n,3))
        x2y2 = np.ones((n,3))
        
        if angle == 270:
            x1y1[:,:2] = self.bboxes[:,[0,3]]
            x2y2[:,:2] = self.bboxes[:,[2,1]]
        
        elif angle == 90:
            x1y1[:,:2] = self.bboxes[:,[2,1]]
            x2y2[:,:2] = self.bboxes[:,[0,3]]    

        elif angle == 180:
            x1y1[:,:2] = self.bboxes[:,[2,3]]
            x2y2[:,:2] = self.bboxes[:,[0,1]]
        
        elif angle == 0:
            return
        else:
            print('error rot angle')
        
        new_bbox[:,:2] = (x1y1 @ M.T)[:,:2]
        new_bbox[:,-2:] = (x2y2 @ M.T)[:,:2]
        
        self.bboxes = new_bbox
class InstancesPose(Instances):
    def __init__(self, bboxes, poses, segments=None, keypoints=None, bbox_format='xywh', normalized=True) -> None:
        """
        Args:
            bboxes (ndarray): bboxes with shape [N, 4].
            segments (list | ndarray): segments.
            keypoints (ndarray): keypoints(x, y, visible) with shape [N, 17, 3].
        """
        self._bboxes = BboxesPose(bboxes=bboxes, format=bbox_format)
        self.poses = poses
        self.keypoints = keypoints
        self.normalized = normalized
        self.segments = segments
        
    def rot_90_bbox(self, M, degree):
        self.poses[:, -1] = degree/270

        self._bboxes.rot_90(M, degree)
        
    def __getitem__(self, index) -> 'Instances':
        """
        Retrieve a specific instance or a set of instances using indexing.

        Args:
            index (int, slice, or np.ndarray): The index, slice, or boolean array to select
                                               the desired instances.

        Returns:
            Instances: A new Instances object containing the selected bounding boxes,
                       segments, and keypoints if present.

        Note:
            When using boolean indexing, make sure to provide a boolean array with the same
            length as the number of instances.
        """
        segments = self.segments[index] if len(self.segments) else self.segments
        keypoints = self.keypoints[index] if self.keypoints is not None else None
        bboxes = self.bboxes[index]
        bbox_format = self._bboxes.format
        poses = self.poses[index]
        
        return InstancesPose(
            bboxes=bboxes,
            poses=poses,
            segments=segments,
            keypoints=keypoints,
            bbox_format=bbox_format,
            normalized=self.normalized,
        ) 

from scipy.ndimage import rotate
class RandomRot:
    """Resize image and padding for detection, instance segmentation, pose"""

    def __init__(self):
        pass
    
    def __call__(self, labels=None, image=None):
        if labels is None:
            labels = {}
        img = labels.get('img') if image is None else image

        shape = img.shape[:2]  # current shape [height, width]

        R = np.eye(3)
        angle = random.choice([0, 180, 90, 270])  # add 90deg rotations to small rotations
        
        # if 4 in labels['cls']:
        #     R[:2] = cv2.getRotationMatrix2D(angle=180, center=(int(shape[0]//2), int(shape[1]//2)), scale=1)
        #     img = cv2.warpAffine(img, R[:2], dsize=shape, borderValue=(0, 0, 0))
            
        R[:2] = cv2.getRotationMatrix2D(angle=angle, center=(int(shape[0]//2), int(shape[1]//2)), scale=1)
        # img = cv2.warpAffine(image, R[:2], dsize=shape, borderValue=(0, 0, 0))
        if angle == 180:
            img = img[::-1, ::-1]
        elif angle == 270:
            img = np.transpose(img, axes=(1, 0, 2))[:, ::-1]
        elif angle == 90:
            img = np.transpose(img, axes=(1, 0, 2))[::-1]
        # img = rotate(image, angle=-angle, axes=(0, 1), reshape=False, mode='reflect') 

        labels['instances'].convert_bbox(format='xyxy')
        labels['instances'].denormalize(*labels['img'].shape[:2][::-1])
        labels['instances'].rot_90_bbox(R, angle)
        labels['img'] = img
        return labels

class FormatPose:
    def __init__(self, **kwargs):
        self.format = Format(**kwargs)
        self.pose_dim = 2
    def __call__(self, labels):
        former_label = self.format(labels.copy())
        instances = labels.pop('instances')
        nl = len(instances)
        if nl:
            poses_list = []
            for i in range(len(instances.poses)):
                single_pose = instances.poses[i]
                if self.pose_dim == 1 or self.pose_dim==2:
                    poses_list.append(single_pose)
                elif self.pose_dim == 6:
                    matrix = R.from_euler('xyz', single_pose, degrees=True).as_matrix()
                    poses_list.append([np.concatenate((matrix[:,0], matrix[:,1]),0)])
            poses_arr = np.concatenate(poses_list, 0, dtype=np.float32)
                
            former_label['poses'] = torch.from_numpy(poses_arr) 
        else:
            former_label['poses'] = torch.zeros((nl, self.pose_dim))
        return former_label
    
class YOLODataset(BaseDataset):
    """
    Dataset class for loading object detection and/or segmentation labels in YOLO format.

    Args:
        data (dict, optional): A dataset YAML dictionary. Defaults to None.
        task (str): An explicit arg to point current task, Defaults to 'detect'.

    Returns:
        (torch.utils.data.Dataset): A PyTorch dataset object that can be used for training an object detection model.
    """

    def __init__(self, *args, data=None, task='detect', **kwargs):
        """Initializes the YOLODataset with optional configurations for segments and keypoints."""
        self.use_segments = task == 'segment'
        self.use_keypoints = task == 'pose'
        self.use_obb = task == 'obb'
        self.data = {
            # 'train':'/data/shenfeihong/classification/image/train',
            # 'val':'/data/shenfeihong/classification/image/val',
            'train':'/home/vsfh/data/cls/image/train_sub',
            'val':'/home/vsfh/data/cls/image/train_sub', 
            'names':{2:'ceph',
                    8:'bite',
                    1:'pano',
                    3:'upper',
                    4:'lower',
                    5:'right',
                    6:'front',
                    7:'left',
                    9:'smile',
                    10:'face',
                    0:'small',
                    11:'f-ceph'}, 
                    'nc':12}
        assert not (self.use_segments and self.use_keypoints), 'Can not use both segments and keypoints.'
        super().__init__(*args, **kwargs)

    def verify_image_label(self, args):
        from ultralytics.data.utils import exif_size, ImageOps, segments2boxes
        """Verify one image-label pair."""
        pose_dim = 2
        im_file, lb_file, prefix, keypoint, num_cls, nkpt, ndim = args
        # Number (missing, found, empty, corrupt), message, segments, keypoints
        nm, nf, ne, nc, msg, segments, keypoints = 0, 0, 0, 0, '', [], None
        # Verify images
        im = Image.open(im_file)
        im.verify()  # PIL verify
        shape = exif_size(im)  # image size
        shape = (shape[1], shape[0])  # hw
        assert (shape[0] > 9) & (shape[1] > 9), f'image size {shape} <10 pixels'
        assert im.format.lower() in IMG_FORMATS, f'invalid image format {im.format}'
        if im.format.lower() in ('jpg', 'jpeg'):
            with open(im_file, 'rb') as f:
                f.seek(-2, 2)
                if f.read() != b'\xff\xd9':  # corrupt JPEG
                    ImageOps.exif_transpose(Image.open(im_file)).save(im_file, 'JPEG', subsampling=0, quality=100)
                    msg = f'{prefix}WARNING ⚠️ {im_file}: corrupt JPEG restored and saved'
        folder_name = lb_file.split('/')[-2]    
        # Verify labels
        if os.path.isfile(lb_file):
            nf = 1  # label found
            with open(lb_file) as f:
                # lb = [x.split() for x in f.read().strip().splitlines() if len(x)]
                if folder_name=='18':
                    lb = []
                else:
                    context = json.load(f)
                    bbox = context['xyxy']
                    euler = [0,0,0]
                    if folder_name in smile_cls:
                        cls = 9
                        a = bbox[0]-0.1*abs(bbox[2]-bbox[0])
                        c = bbox[2]+0.1*abs(bbox[2]-bbox[0])
                        b = bbox[1]-0.1*abs(bbox[3]-bbox[1])
                        d = bbox[3]+0.1*abs(bbox[3]-bbox[1])
                        euler = context['euler']
                        if len(euler) != 3:
                            euler = euler[0]
                    elif folder_name in face_cls:
                        cls = 10
                        a = bbox[0]-0.1*abs(bbox[2]-bbox[0])
                        c = bbox[2]+0.1*abs(bbox[2]-bbox[0])
                        b = bbox[1]-0.1*abs(bbox[3]-bbox[1])
                        d = bbox[3]+0.1*abs(bbox[3]-bbox[1])
                        euler = context['euler']
                        if len(euler) != 3:
                            euler = euler[0]
                    else:
                        cls = project[folder_name][1]
                        a,b,c,d = bbox
                    a = min(max(0.001,a/shape[1]),0.999)
                    b = min(max(0.001,b/shape[0]),0.999)
                    c = min(max(a,min(c/shape[1],0.999)),0.999)
                    d = min(max(b,min(d/shape[0],0.999)),0.999)
                    a,b,c,d = xyxy2xywh(np.array([a,b,c,d])).tolist()
                    lb = [[cls, a,b,c,d, euler[1]/90, 0]]
                    # if any(len(x) > 6 for x in lb) and (not keypoint):  # is segment
                    if False:  # is segment
                        classes = np.array([x[0] for x in lb], dtype=np.float32)
                        segments = [np.array(x[1:], dtype=np.float32).reshape(-1, 2) for x in lb]  # (cls, xy1...)
                        lb = np.concatenate((classes.reshape(-1, 1), segments2boxes(segments)), 1)  # (cls, xywh)
                    lb = np.array(lb, dtype=np.float32)
            nl = len(lb)
            if nl:
                if keypoint:
                    assert lb.shape[1] == (5 + nkpt * ndim), f'labels require {(5 + nkpt * ndim)} columns each'
                    points = lb[:, 5:].reshape(-1, ndim)[:, :2]
                else:
                    assert lb.shape[1] == 5 + pose_dim, f'labels require 5 columns, {lb.shape[1]} columns detected'
                    points = lb[:, 1:5]
                if points.max()>1:
                    a = 1
                assert points.max() <= 1, f'non-normalized or out of bounds coordinates {bbox} {shape}'
                assert lb[:,:5].min() >= 0, f'negative label values {lb[lb < 0]}'

                # All labels
                max_cls = lb[:, 0].max()  # max label count
                assert max_cls <= num_cls, \
                    f'Label class {int(max_cls)} exceeds dataset class count {num_cls}. ' \
                    f'Possible class labels are 0-{num_cls - 1}'
                _, i = np.unique(lb, axis=0, return_index=True)
                if len(i) < nl:  # duplicate row check
                    lb = lb[i]  # remove duplicates
                    if segments:
                        segments = [segments[x] for x in i]
                    msg = f'{prefix}WARNING ⚠️ {im_file}: {nl - len(i)} duplicate labels removed'
            else:
                ne = 1  # label empty
                lb = np.zeros((0, (5 + pose_dim + nkpt * ndim) if keypoint else 5+pose_dim), dtype=np.float32)
        else:
            nm = 1  # label missing
            lb = np.zeros((0, (5 + pose_dim + nkpt * ndim) if keypoints else 5+pose_dim), dtype=np.float32)
        if keypoint:
            keypoints = lb[:, 5:].reshape(-1, nkpt, ndim)
            if ndim == 2:
                kpt_mask = np.where((keypoints[..., 0] < 0) | (keypoints[..., 1] < 0), 0.0, 1.0).astype(np.float32)
                keypoints = np.concatenate([keypoints, kpt_mask[..., None]], axis=-1)  # (nl, nkpt, 3)
        # lb = lb[:, :5]
        return im_file, lb, shape, segments, keypoints, nm, nf, ne, nc, msg


    def get_img_files(self, img_path):
        """Read image files."""
        try:
            f = []  # image files
            for p in img_path if isinstance(img_path, list) else [img_path]:
                p = Path(p)  # os-agnostic
                if p.is_dir():  # dir
                    f += glob.glob(str(p / '**' / '*.*'), recursive=True)
                    # F = list(p.rglob('*.*'))  # pathlib
                elif p.is_file():  # file
                    with open(p) as t:
                        t = t.read().strip().splitlines()
                        parent = str(p.parent) + os.sep
                        f += [x.replace('./', parent) if x.startswith('./') else x for x in t]  # local to global path
                        # F += [p.parent / x.lstrip(os.sep) for x in t]  # local to global path (pathlib)
                else:
                    raise FileNotFoundError(f'{self.prefix}{p} does not exist')
            im_files = sorted(x.replace('/', os.sep) for x in f if x.split('.')[-1].lower() in IMG_FORMATS)
            # im_files = [im for im in im_files if im.split('/')[-2]!= '18']
            # self.img_files = sorted([x for x in f if x.suffix[1:].lower() in IMG_FORMATS])  # pathlib
            assert im_files, f'{self.prefix}No images found in {img_path}'
        except Exception as e:
            raise FileNotFoundError(f'{self.prefix}Error loading data from {img_path}\n{HELP_URL}') from e
        if self.fraction < 1:
            im_files = im_files[:round(len(im_files) * self.fraction)]
        return im_files
    
    def cache_labels(self, path=Path('./labels.cache')):
        """
        Cache dataset labels, check images and read shapes.

        Args:
            path (Path): path where to save the cache file (default: Path('./labels.cache')).
        Returns:
            (dict): labels.
        """
        x = {'labels': []}
        nm, nf, ne, nc, msgs = 0, 0, 0, 0, []  # number missing, found, empty, corrupt, messages
        desc = f'{self.prefix}Scanning {path.parent / path.stem}...'
        total = len(self.im_files)
        nkpt, ndim = (0, 0)
        if self.use_keypoints and (nkpt <= 0 or ndim not in (2, 3)):
            raise ValueError("'kpt_shape' in data.yaml missing or incorrect. Should be a list with [number of "
                             "keypoints, number of dims (2 for x,y or 3 for x,y,visible)], i.e. 'kpt_shape: [17, 3]'")
        with ThreadPool(NUM_THREADS) as pool:
            results = pool.imap(func=self.verify_image_label,
                                iterable=zip(self.im_files, self.label_files, repeat(self.prefix),
                                             repeat(self.use_keypoints), repeat(len(self.data['names'])), repeat(nkpt),
                                             repeat(ndim)))
            pbar = TQDM(results, desc=desc, total=total)
            for im_file, lb, shape, segments, keypoint, nm_f, nf_f, ne_f, nc_f, msg in pbar:
                nm += nm_f
                nf += nf_f
                ne += ne_f
                nc += nc_f
                if im_file:
                    x['labels'].append(
                        dict(
                            im_file=im_file,
                            shape=shape,
                            cls=lb[:, 0:1],  # n, 1
                            bboxes=lb[:, 1:5],  # n, 4
                            poses=lb[:,5:],
                            segments=segments,
                            keypoints=keypoint,
                            normalized=True,
                            bbox_format='xywh'))
                if msg:
                    msgs.append(msg)
                pbar.desc = f'{desc} {nf} images, {nm + ne} backgrounds, {nc} corrupt'
            pbar.close()

        if msgs:
            LOGGER.info('\n'.join(msgs))
        if nf == 0:
            LOGGER.warning(f'{self.prefix}WARNING ⚠️ No labels found in {path}. {HELP_URL}')
        x['hash'] = get_hash(self.label_files + self.im_files)
        x['results'] = nf, nm, ne, nc, len(self.im_files)
        x['msgs'] = msgs  # warnings
        save_dataset_cache_file(self.prefix, path, x)
        return x

    def img2label_paths(self, im_files):
        label_files = []
        # for path in im_files:
        #     label_files.append(path.replace('.jpg','.json').replace('img','label'))
        json_dir = '/data/shenfeihong/classification/network_res/'
        json_dir = '/mnt/hdd/data/cls/network_res/'
        for x in im_files:
            label_files.append(os.path.join(json_dir, x.split('/')[-2], os.path.basename(x).replace('jpg', 'json')))
        return label_files
    
    def get_labels(self):
        """Returns dictionary of labels for YOLO training."""
        self.label_files = self.img2label_paths(self.im_files)
        cache_path = Path(self.im_files[0]).parent.parent.with_suffix('.cache')
        try:
            cache, exists = load_dataset_cache_file(cache_path), True  # attempt to load a *.cache file
            assert cache['version'] == DATASET_CACHE_VERSION  # matches current version
            assert cache['hash'] == get_hash(self.label_files + self.im_files)  # identical hash
        except (FileNotFoundError, AssertionError, AttributeError):
            cache, exists = self.cache_labels(cache_path), False  # run cache ops
        # Display cache
        nf, nm, ne, nc, n = cache.pop('results')  # found, missing, empty, corrupt, total
        if exists and LOCAL_RANK in (-1, 0):
            d = f'Scanning {cache_path}... {nf} images, {nm + ne} backgrounds, {nc} corrupt'
            TQDM(None, desc=self.prefix + d, total=n, initial=n)  # display results
            if cache['msgs']:
                LOGGER.info('\n'.join(cache['msgs']))  # display warnings

        # Read cache
        [cache.pop(k) for k in ('hash', 'version', 'msgs')]  # remove items
        labels = cache['labels']
        if not labels:
            LOGGER.warning(f'WARNING ⚠️ No images found in {cache_path}, training may not work correctly. {HELP_URL}')
        self.im_files = [lb['im_file'] for lb in labels]  # update im_files

        # Check if the dataset is all boxes or all segments
        lengths = ((len(lb['cls']), len(lb['bboxes']), len(lb['segments'])) for lb in labels)
        len_cls, len_boxes, len_segments = (sum(x) for x in zip(*lengths))
        if len_segments and len_boxes != len_segments:
            LOGGER.warning(
                f'WARNING ⚠️ Box and segment counts should be equal, but got len(segments) = {len_segments}, '
                f'len(boxes) = {len_boxes}. To resolve this only boxes will be used and all segments will be removed. '
                'To avoid this please supply either a detect or segment dataset, not a detect-segment mixed dataset.')
            for lb in labels:
                lb['segments'] = []
        if len_cls == 0:
            LOGGER.warning(f'WARNING ⚠️ No labels found in {cache_path}, training may not work correctly. {HELP_URL}')
        return labels

    def build_transforms(self, hyp=None):
        """Builds and appends transforms to the list."""
        transforms = Compose([LetterBox(new_shape=(self.imgsz, self.imgsz), scaleup=False), RandomRot(),])

        transforms.append(
            FormatPose(bbox_format='xywh',
                   normalize=True,
                   return_mask=self.use_segments,
                   return_keypoint=self.use_keypoints,
                   return_obb=self.use_obb,
                   batch_idx=True,
                   mask_ratio=hyp.mask_ratio,
                   mask_overlap=hyp.overlap_mask))
        return transforms

    def close_mosaic(self, hyp):
        """Sets mosaic, copy_paste and mixup options to 0.0 and builds transformations."""
        hyp.mosaic = 0.0  # set mosaic ratio=0.0
        hyp.copy_paste = 0.0  # keep the same behavior as previous v8 close-mosaic
        hyp.mixup = 0.0  # keep the same behavior as previous v8 close-mosaic
        self.transforms = self.build_transforms(hyp)

    def update_labels_info(self, label):
        """Custom your label format here."""
        # NOTE: cls is not with bboxes now, classification and semantic segmentation need an independent cls label
        # We can make it also support classification and semantic segmentation by add or remove some dict keys there.
        bboxes = label.pop('bboxes')
        poses = label.pop('poses')
        segments = label.pop('segments', [])
        keypoints = label.pop('keypoints', None)
        bbox_format = label.pop('bbox_format')
        normalized = label.pop('normalized')

        # NOTE: do NOT resample oriented boxes
        segment_resamples = 100 if self.use_obb else 1000
        if len(segments) > 0:
            # list[np.array(1000, 2)] * num_samples
            # (N, 1000, 2)
            segments = np.stack(resample_segments(segments, n=segment_resamples), axis=0)
        else:
            segments = np.zeros((0, segment_resamples, 2), dtype=np.float32)
        label['instances'] = InstancesPose(bboxes, poses, segments, keypoints, bbox_format=bbox_format, normalized=normalized)
        return label

    @staticmethod
    def collate_fn(batch):
        """Collates data samples into batches."""
        new_batch = {}
        keys = batch[0].keys()
        values = list(zip(*[list(b.values()) for b in batch]))
        for i, k in enumerate(keys):
            value = values[i]
            if k == 'img':
                value = torch.stack(value, 0)
            if k in ['masks', 'keypoints', 'bboxes', 'cls', 'segments', 'obb']:
                value = torch.cat(value, 0)
            new_batch[k] = value
        new_batch['batch_idx'] = list(new_batch['batch_idx'])
        for i in range(len(new_batch['batch_idx'])):
            new_batch['batch_idx'][i] += i  # add target image index for build_targets()
        new_batch['batch_idx'] = torch.cat(new_batch['batch_idx'], 0)
        return new_batch

def load_dataset_cache_file(path):
    """Load an Ultralytics *.cache dictionary from path."""
    import gc
    gc.disable()  # reduce pickle load time https://github.com/ultralytics/ultralytics/pull/1585
    cache = np.load(str(path), allow_pickle=True).item()  # load dict
    gc.enable()
    return cache

def save_dataset_cache_file(prefix, path, x):
    """Save an Ultralytics dataset *.cache dictionary x to path."""
    x['version'] = DATASET_CACHE_VERSION  # add cache version
    if is_dir_writeable(path.parent):
        if path.exists():
            path.unlink()  # remove *.cache file if exists
        np.save(str(path), x)  # save cache for next time
        path.with_suffix('.cache.npy').rename(path)  # remove .npy suffix
        LOGGER.info(f'{prefix}New cache created: {path}')
    else:
        LOGGER.warning(f'{prefix}WARNING ⚠️ Cache directory {path.parent} is not writeable, cache not saved.')

if __name__=='__main__':
    ds = YOLODataset(
        img_path='/data/shenfeihong/classification/03/',
        data={'names':{0:'a'}})
    for i in ds:
        break
