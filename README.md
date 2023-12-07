# SADEKTECH
Kurallar:

## YOLOV8 ÖNEMLİ NOTLAR
### KURULUM:
* Conda env oluşturma python verisyonu önemli torch uyumlu olmalı:
  ```
  conda create -n yolov8 python=3.7.10
  ```

* Conda env aktivasyonu:
  ```
  conda activate yolov8
  ```

* Torch ve torchvision kurulumu:
  ```
  pip install torch==1.7.1 torchvision==0.8.2
  ```
  
* YoloV8 klasörümüzüzün içine gelmeliyiz:
  ```
  cd .. yolov8 klasörünün içine gel
  ```
  
* YoloV8 requirments kurulumları:
  ```
  pip install -e '.[dev]'
  ```D:\Sadektech\yolov8\YOLOv8-DeepSORT-Object-Tracking\ultralytics\yolo\v8\detect\predict.py
### TRAIN:
* Train.py dosyasının bulunduğu dizine gelelim ve kodu çalıştıralım:
  ```
  cd ...yolov8\YOLOv8-DeepSORT-Object-Tracking\ultralytics\yolo\v8\detect
  python3 train.py model=yolov8l.pt data=/home/sami/YOLOv8-DeepSORT-Object-Tracking/ultralytics/yolo/v8/detect/yolov8-1/data.yaml epochs=5 imgsz=225
  ```
### DETECT:
* Numpy versiyonumuz 1.20 bu yüzden detect işlemi için çalıştırdığımız predict.py dosyasında bulunan kodda numpy verisyonu ile ilgili bir hata aldık. Predict.py dosyası birçok class'tan işlemler gerçekleştiriyor.

* Predict.py dosyasının bulunduğu dizine gelelim ve kodu çalıştıralım:
  ```
  cd ...yolov8\YOLOv8-DeepSORT-Object-Tracking\ultralytics\yolo\v8\detect
  python3 predict.py model='/home/sami/YOLOv8-DeepSORT-Object-Tracking/runs/detect/train28/weights/best.pt' source="0"
  ```


