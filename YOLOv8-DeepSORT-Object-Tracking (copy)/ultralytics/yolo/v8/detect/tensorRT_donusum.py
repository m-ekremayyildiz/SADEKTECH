from ultralytics import YOLO
import torch

# YOLO modelini oluştur
model = YOLO('/home/sami/YOLOv8-DeepSORT-Object-Tracking/runs/detect/train28/weights/best.pt')


# Şimdi model GPU üzerinde veya CPU üzerinde çalışacaktır
#results = model.export(format='engine', device=0, imgsz=256, dynamic=False)
results = model.export(format='engine', device=0,imgsz=(256,256), dynamic=True)





