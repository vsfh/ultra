
# Ultralytics Multi-GPU training temp file (should be automatically deleted after use)
overrides = {'task': 'detect', 'mode': 'train', 'model': './det.yaml', 'data': 'shenfeihong', 'epochs': 100, 'time': None, 'patience': 50, 'batch': 120, 'imgsz': 640, 'save': True, 'save_period': -1, 'cache': False, 'device': '1,2,3', 'workers': 8, 'project': None, 'name': 'train2', 'exist_ok': False, 'pretrained': True, 'optimizer': 'AdamW', 'verbose': True, 'seed': 0, 'deterministic': True, 'single_cls': False, 'rect': False, 'cos_lr': False, 'close_mosaic': 10, 'resume': False, 'amp': True, 'fraction': 1.0, 'profile': False, 'freeze': None, 'multi_scale': False, 'overlap_mask': True, 'mask_ratio': 4, 'dropout': 0.0, 'val': True, 'split': 'val', 'save_json': False, 'save_hybrid': False, 'conf': None, 'iou': 0.7, 'max_det': 300, 'half': False, 'dnn': False, 'plots': True, 'source': None, 'vid_stride': 1, 'stream_buffer': False, 'visualize': False, 'augment': False, 'agnostic_nms': False, 'classes': None, 'retina_masks': False, 'embed': None, 'show': False, 'save_frames': False, 'save_txt': False, 'save_conf': False, 'save_crop': False, 'show_labels': True, 'show_conf': True, 'show_boxes': True, 'line_width': None, 'format': 'onnx', 'keras': False, 'optimize': False, 'int8': False, 'dynamic': True, 'simplify': False, 'opset': None, 'workspace': 4, 'nms': False, 'lr0': 0.001, 'lrf': 0.01, 'momentum': 0.937, 'weight_decay': 0.0005, 'warmup_epochs': 3.0, 'warmup_momentum': 0.8, 'warmup_bias_lr': 0.1, 'box': 7.5, 'cls': 0.5, 'dfl': 1.5, 'pose': 12.0, 'kobj': 1.0, 'label_smoothing': 0.0, 'nbs': 64, 'hsv_h': 0.015, 'hsv_s': 0.7, 'hsv_v': 0.4, 'degrees': 0.0, 'translate': 0.1, 'scale': 0.5, 'shear': 0.0, 'perspective': 0.0, 'flipud': 0.0, 'fliplr': 0.5, 'mosaic': 1.0, 'mixup': 0.0, 'copy_paste': 0.0, 'auto_augment': 'randaugment', 'erasing': 0.4, 'crop_fraction': 1.0, 'cfg': None, 'tracker': 'botsort.yaml', 'save_dir': 'runs/detect/train2'}

if __name__ == "__main__":
    import sys
    sys.path.append('.')

    from detect.train import DetectionTrainer
    from ultralytics.utils import DEFAULT_CFG_DICT

    cfg = DEFAULT_CFG_DICT.copy()
    cfg.update(save_dir='')   # handle the extra key 'save_dir'
    trainer = DetectionTrainer(cfg=cfg, overrides=overrides)
    results = trainer.train()
