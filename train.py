import sys
sys.path.append('.')
from yolo import YOLO


def train_new():
    model = YOLO('/home/gregory/code/ultra/runs/detect/150epoch_0.01_0.1_cosinlr_one_face_two_class_all_cut_face/weights/best.pt')
    # model.train(device='4,5,6,7',epochs=20,batch=64)
    model.train(batch=32)
    
if __name__=='__main__':
    train_new()
