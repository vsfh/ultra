import sys
sys.path.append('.')
from yolo import YOLO

    
    
def predict_new():
    model = YOLO('/home/gregory/code/ultra/runs/detect/train/weights/best.pt')
    model.predict('/home/gregory/code/ultra/aaa.jpg', device='cuda')
    


if __name__=='__main__':
    predict_new()
