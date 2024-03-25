import sys
sys.path.append('.')
from yolo import YOLO


def train_new():
    model = YOLO('yolov8n.pt')
    # model.train(device='4,5,6,7')
    model.train(epochs=10)
    
def try_():
    import torch
    model = YOLO('/home/vsfh/code/ultra/runs/detect/train4/weights/best.pt')
    ckpt = torch.load('/home/vsfh/code/ultra/runs/detect/train4/weights/best.pt')
    # model.model.model.load_state_dict(ckpt)
    # model.model.model.eval()
    img_path = '/home/vsfh/data/cls/image/error/'
    model.predict(img_path)
def predict_new():
    model = YOLO('/home/vsfh/code/ultra/runs/detect/train2/weights/best.pt')
    model.predict('/home/vsfh/data/cls/image/error/58443579409700480_105149.jpg', device='cuda')
    
    
# def predict_new():
#     import os
#     import numpy as np
#     model = YOLO('/mnt/e/wsl/code/ultralytics/make_data_folder/runs/classify/train7/weights/best.pt')
#     path = '/mnt/e/data/classification/image_folder_04/val/03/'
#     for name in os.listdir(path):
#         res = model.predict(os.path.join(path, name))
#         cls = res[0].probs.top1
#         if cls:
#             img = cv2.imread(os.path.join(path, name))
#             img = np.rot90(img, k=cls)
#             cv2.imwrite(os.path.join(path, name), img)
#             print(name)
#         # break
    

def export_new():
    # model = YOLO('/home/gregory/code/ultralytics/make_data_folder/runs/detect/train2/weights/best.pt')
    model = YOLO('/home/gregory/code/ultra/runs/detect/train2/weights/best.pt')
    model.export(format="torchscript",dynamic=True,imgsz=640,device='cuda')

if __name__=='__main__':
    # try_()
    train_new()
    # predict_new()
