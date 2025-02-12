import sys
sys.path.append('.')
from yolo import YOLO


def train_new():
    model = YOLO('cls.yaml')
    # model.train(device='4,5,6,7',epochs=20,batch=64)
    model.train(batch=4)
    
if __name__=='__main__':
    train_new()
