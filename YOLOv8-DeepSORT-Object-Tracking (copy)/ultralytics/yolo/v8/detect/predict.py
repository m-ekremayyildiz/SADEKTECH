# Ultralytics YOLO 🚀, GPL-3.0 license

import hydra # Hydra kütüphanesini içeriye alır. Hydra, yapılandırma yönetimi için kullanılan bir kütüphanedir. Bu, program parametrelerini ve yapılandırmalarını kolayca yönetmek için kullanılır.
import torch # PyTorch kütüphanesini içeriye alır. PyTorch, derin öğrenme modelleri oluşturmak ve eğitmek için kullanılan popüler bir makine öğrenimi kütüphanesidir.
import argparse # Komut satırı argümanlarını işlemek için kullanılan argparse kütüphanesini içeriye alır. Program çalıştırılırken kullanıcının belirlediği parametreleri kolayca almak için kullanılır.
import time # Zamanla ilgili işlemleri gerçekleştirmek için kullanılan bir modüldür.
from pathlib import Path # Dosya yollarını temsil etmek ve işlemek için kullanılan bir sınıftır. pathlib modülü, dosya sistemleri üzerinde işlem yapmayı kolaylaştıran bir araç sağlar.

import cv2 # OpenCV (Open Source Computer Vision Library) kütüphanesini içeriye alır. OpenCV, bilgisayar görüşü ve bilgisayarla görme görevleri için bir dizi araç sağlayan popüler bir kütüphanedir.
import torch 
import torch.backends.cudnn as cudnn # NVIDIA'nın Derin Sinir Ağı Kütüphanesi'nin (cuDNN) PyTorch bağlamını etkinleştirmek için kullanılır. cuDNN, NVIDIA GPU'lar üzerinde hızlandırılmış derin öğrenme işlemleri için optimize edilmiş bir kütüphanedir.
from numpy import random # Numpy kütüphanesinin bir alt modülü olan random modülünü içeriye alır. Bu, rastgele sayı üretimi ve rastgele öğelerle çalışma gibi rastgele işlemleri gerçekleştirmek için kullanılır.
from ultralytics.yolo.engine.predictor import BasePredictor # predicctor.py dosyasından BasePredictor fonksiyonunu getiriyoruz.
from ultralytics.yolo.utils import DEFAULT_CONFIG, ROOT, ops
"""
    DEFAULT_CONFIG: Bu, YOLO (You Only Look Once) algoritması için varsayılan yapılandırma ayarlarını içeren bir nesnedir. Bu ayarlar, modelin eğitimi, tahmini ve diğer işlemleri için kullanılan parametreleri içerir.

    ROOT: Bu, YOLO'nun kök dizinini temsil eden bir nesnedir. YOLO kütüphanesinin çalışma dizinini belirtir.

    ops: Bu, YOLO operasyonlarını (operations) içeren bir modüldür. YOLO'nun çeşitli işlevselliğini uygulayan temel operasyonları içerir.

Bu öğeler, YOLO modelini yapılandırmak, çalıştırmak ve sonuçları işlemek için kullanılır. DEFAULT_CONFIG öğesi, modelin varsayılan ayarlarını içerir ve 
bu ayarlar üzerinde özelleştirmeler yaparak YOLO modelini belirli bir görev veya veri kümesine uyarlamak mümkündür. 
ROOT ve ops öğeleri de ilgili kütüphanenin alt bileşenlerini kullanmak için gereklidir.
"""
from ultralytics.yolo.utils.checks import check_imgsz # checks modülünden check_imgsz fonksiyonunu içeri alır. Bu fonksiyon, resim boyutları gibi giriş parametrelerini kontrol etmek için kullanılabilir.
from ultralytics.yolo.utils.plotting import Annotator, colors, save_one_box # Bu, YOLO'nun sonuçlarını çizim ve görselleştirme işlemleri için gerekli olan plotting modülünden bazı sınıfları ve fonksiyonları içeri alır. Annotator sınıfı, görüntü üzerine nesne tespiti sonuçlarını eklemek için kullanılabilir. colors ve save_one_box ise çeşitli renk ve kutu kaydetme işlemlerini içerir.

import cv2
import time
from deep_sort_pytorch.utils.parser import get_config # DeepSORT (Deep Simple Online and Realtime Tracking) algoritması için yapılandırma ayarlarını okumak için kullanılan get_config fonksiyonunu içeri alır.
from deep_sort_pytorch.deep_sort import DeepSort # DeepSORT algoritmasını içeren deep_sort modülünden DeepSort sınıfını içeri alır. DeepSORT, nesne takibi için kullanılan bir derin öğrenme tabanlı algoritmadır.
from collections import deque # Python'ın collections modülünden deque sınıfını içeri alır. Bu, bir çift taraflı kuyruk (double-ended queue) veri yapısını temsil eder ve genellikle belirli bir kapasiteye sahip bir veri geçmişini tutmak için kullanılır.
import numpy as np # NumPy kütüphanesini içeri alır. NumPy, çok boyutlu dizilerle çalışmayı sağlayan ve bilimsel hesaplamalar için kullanılan güçlü bir Python kütüphanesidir.
palette = (2 ** 11 - 1, 2 ** 15 - 1, 2 ** 20 - 1)
data_deque = {}

deepsort = None

def init_tracker(): # Bu fonksiyon, DeepSORT algoritmasının başlatılmasını ve yapılandırılmasını sağlar. 
    global deepsort
    cfg_deep = get_config() # get_config fonksiyonunu kullanarak, DeepSORT algoritmasının yapılandırma ayarlarını içeren bir konfigürasyon objesi (cfg_deep) oluşturur.
    cfg_deep.merge_from_file("deep_sort_pytorch/configs/deep_sort.yaml") # YAML dosyasından gelen ek yapılandırma ayarlarını cfg_deep objesine birleştirir. Bu dosya, DeepSORT'un çeşitli parametrelerini içerir ve algoritmanın davranışını şekillendirmek için kullanılır.

    deepsort= DeepSort(cfg_deep.DEEPSORT.REID_CKPT, # Re-identification (ReID) modelinin checkpoint dosyasının yolunu belirtir.
                            max_dist=cfg_deep.DEEPSORT.MAX_DIST, min_confidence=cfg_deep.DEEPSORT.MIN_CONFIDENCE, # İki takip noktası arasındaki maksimum mesafeyi belirler. Minimum güvenilirlik değeri, bir nesnenin takip edilmesi için bu değeri aşmalıdır.
                            nms_max_overlap=cfg_deep.DEEPSORT.NMS_MAX_OVERLAP, max_iou_distance=cfg_deep.DEEPSORT.MAX_IOU_DISTANCE, # Non-maximum suppression (NMS) için maksimum örtüşme eşiği. İki nesnenin IOU (Intersection over Union) değerinin maksimum mesafesi.
                            max_age=cfg_deep.DEEPSORT.MAX_AGE, n_init=cfg_deep.DEEPSORT.N_INIT, nn_budget=cfg_deep.DEEPSORT.NN_BUDGET, # Bir nesnenin kaç çerçeve boyunca takip edileceğini belirler. Bir nesnenin takibi için kaç çerçeve boyunca algılama yapılacağını belirler. Kullanılacak en yakın komşu sayısını belirler.
                            use_cuda=True) # CUDA tabanlı GPU kullanımını belirler (eğer mevcutsa).
##########################################################################################
def xyxy_to_xywh(*xyxy):
    """ 
    Bu fonksiyonun parametresi, sınırlayıcı kutunun dört köşe noktasının koordinatlarını içeren bir demet (tuple) olarak kabul edilir.
    Fonksiyon, bu köşe noktalarını kullanarak sınırlayıcı kutunun sol üst ve sağ alt köşelerinin koordinatlarını belirler. 
    
    Bu dönüşüm, sınırlayıcı kutunun koordinatlarını merkez-gerçek genişlik (x_c, y_c, w, h) formatına dönüştürmek amacıyla kullanılır. 
    Bu tür bir format, özellikle nesne tespiti ve takip algoritmalarında sıklıkla kullanılır.
    """
    bbox_left = min([xyxy[0].item(), xyxy[2].item()])
    bbox_top = min([xyxy[1].item(), xyxy[3].item()])
    bbox_w = abs(xyxy[0].item() - xyxy[2].item())
    bbox_h = abs(xyxy[1].item() - xyxy[3].item())
    x_c = (bbox_left + bbox_w / 2)
    y_c = (bbox_top + bbox_h / 2)
    w = bbox_w
    h = bbox_h
    return x_c, y_c, w, h

def xyxy_to_tlwh(bbox_xyxy):
    """
    Bu fonksiyonun girdisi, sınırlayıcı kutuların koordinatlarını içeren bir liste olarak kabul edilir. 
    Her bir sınırlayıcı kutu, [x1, y1, x2, y2] formatında dört köşe noktasının koordinatlarını içerir.

    Bu tür bir dönüşüm, özellikle nesne takip (object tracking) uygulamalarında kullanılır, 
    çünkü bu tür koordinatlar nesnenin üst sol köşesinin konumunu ve genişlik-yükseklik bilgilerini içerir.
    """
    tlwh_bboxs = []
    for i, box in enumerate(bbox_xyxy):
        x1, y1, x2, y2 = [int(i) for i in box]
        top = x1
        left = y1
        w = int(x2 - x1)
        h = int(y2 - y1)
        tlwh_obj = [top, left, w, h]
        tlwh_bboxs.append(tlwh_obj)
    return tlwh_bboxs

def compute_color_for_labels(label):
    """
    Bu fonksiyon, bir etiketin (label) sınıfına bağlı olarak bir renk belirlemek için kullanılır. Etiketlere özel sabit renkler atanmıştır. 
    Eğer etiket, belirli bir sınıfa karşılık gelen sabit etiketlerden biriyle eşleşiyorsa, bu sınıfa özgü olan sabit rengi döndürür. 
    Eğer etiket bu sabit sınıflardan birine uymuyorsa, bir renk paletini kullanarak etikete özel dinamik bir renk üretilir.
    """
    if label == 0: #person
        color = (85,45,255)
    elif label == 2: # Car
        color = (222,82,175)
    elif label == 3:  # Motobike
        color = (0, 204, 255)
    elif label == 5:  # Bus
        color = (0, 149, 255)
    else:
        color = [int((p * (label ** 2 - label + 1)) % 255) for p in palette]
    return tuple(color)

def draw_border(img, pt1, pt2, color, thickness, r, d): 
    """
    Bu fonksiyon, verilen koordinatlar arasında bir dikdörtgen çizer ve bu dikdörtgenin köşelerine daireler ekler. 
    Ayrıca, dikdörtgenin köşelerinden çıkarak çerçeveyi süslemek için çizgiler de ekler. Son olarak, dikdörtgenin içini ve 
    çizgiler arasındaki alanı doldurarak çerçeveyi tamamlar. 
    Bu işlem, belirli bir nesnenin çevresini vurgulamak veya görsel olarak önemli bir alanı belirtmek için kullanılabilir.
    """
    x1,y1 = pt1
    x2,y2 = pt2
    # Top left
    cv2.line(img, (x1 + r, y1), (x1 + r + d, y1), color, thickness)
    cv2.line(img, (x1, y1 + r), (x1, y1 + r + d), color, thickness)
    cv2.ellipse(img, (x1 + r, y1 + r), (r, r), 180, 0, 90, color, thickness)
    # Top right
    cv2.line(img, (x2 - r, y1), (x2 - r - d, y1), color, thickness)
    cv2.line(img, (x2, y1 + r), (x2, y1 + r + d), color, thickness)
    cv2.ellipse(img, (x2 - r, y1 + r), (r, r), 270, 0, 90, color, thickness)
    # Bottom left
    cv2.line(img, (x1 + r, y2), (x1 + r + d, y2), color, thickness)
    cv2.line(img, (x1, y2 - r), (x1, y2 - r - d), color, thickness)
    cv2.ellipse(img, (x1 + r, y2 - r), (r, r), 90, 0, 90, color, thickness)
    # Bottom right
    cv2.line(img, (x2 - r, y2), (x2 - r - d, y2), color, thickness)
    cv2.line(img, (x2, y2 - r), (x2, y2 - r - d), color, thickness)
    cv2.ellipse(img, (x2 - r, y2 - r), (r, r), 0, 0, 90, color, thickness)

    cv2.rectangle(img, (x1 + r, y1), (x2 - r, y2), color, -1, cv2.LINE_AA)
    cv2.rectangle(img, (x1, y1 + r), (x2, y2 - r - d), color, -1, cv2.LINE_AA)
    
    cv2.circle(img, (x1 +r, y1+r), 2, color, 12)
    cv2.circle(img, (x2 -r, y1+r), 2, color, 12)
    cv2.circle(img, (x1 +r, y2-r), 2, color, 12)
    cv2.circle(img, (x2 -r, y2-r), 2, color, 12)
    
    return img

def UI_box(x, img, color=None, label=None, line_thickness=None):
    """
    Bu fonksiyon, bir nesnenin sınırlayıcı kutusunu (bounding box) bir görüntü üzerinde çizmek ve bu kutuya etiket (label) eklemek için kullanılır. Fonksiyonun parametreleri şu şekildedir:

    x: Bounding box koordinatları (sol üst ve sağ alt köşe koordinatları).
    img: Bounding box'un çizileceği görüntü.
    color: Bounding box ve etiketin renkleri. Varsayılan olarak rastgele bir renk seçilir.
    label: Bounding box üzerinde görüntülenecek etiket.
    line_thickness: Çizgi ve font kalınlığı. Varsayılan olarak görüntünün boyutlarına bağlı olarak dinamik bir değer atanır.

    Fonksiyon, cv2.rectangle işlevini kullanarak bounding box'u çizer. 
    Ardından, eğer bir etiket (label) belirtilmişse, etiketi çizilen bounding box'un sol üst köşesine ekler. 
    Etiketin üzerine çizilen çizgi ve elips detayları, draw_border adlı başka bir fonksiyon kullanılarak eklenir. 
    Bu şekilde, nesnelerin görüntü üzerinde belirgin bir şekilde işaretlenmesi ve izlenmesi sağlanır.
    """
    # Plots one bounding box on image img
    tl = line_thickness or round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1  # line/font thickness
    color = color or [random.randint(0, 255) for _ in range(3)]
    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
    cv2.rectangle(img, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
    if label:
        tf = max(tl - 1, 1)  # font thickness
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]

        img = draw_border(img, (c1[0], c1[1] - t_size[1] -3), (c1[0] + t_size[0], c1[1]+3), color, 1, 8, 2)

        cv2.putText(img, label, (c1[0], c1[1] - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)



def draw_boxes(img, bbox, names,object_id, identities=None, offset=(0, 0)):
    """
    Bu fonksiyon, nesnelerin (örnek olarak araç veya insan) takibini gerçekleştiren bir nesne takip algoritması sonuçlarını görüntü üzerinde görselleştirmek için kullanılır. Kodun ana amaçları şunlardır:

    draw_boxes fonksiyonu, nesne takip algoritmasının çıkışı olan bounding box'ları (sınırlayıcı kutuları) ve bunlara ait kimlik bilgilerini görüntü üzerinde çizer.
    Görüntü üzerinde her bir nesnenin etrafına bir sınırlayıcı kutu (bounding box) çizilir ve bu kutulara nesnenin kimliği ve adı eklenir.
    Her nesnenin takip edilen yolunu göstermek için geçmiş konumlarından oluşan bir izleme çizgisi çizilir. Bu izleme çizgileri, nesnenin geçmiş konumlarına dayanarak dinamik bir kalınlıkla çizilir, 
    yani nesne uzun süre takip edildiyse çizgi kalın, kısa süre takip edildiyse ince olur.
    Nesnelerin takip edildiği süre boyunca görüntü üzerindeki buffer'da (veri deposu) bu nesnelere ait konum bilgileri saklanır ve her bir frame'de bu konumlar güncellenir.

    Bu kod, nesne takibi sırasında nesnelerin hareketini ve takip edilen yollarını görselleştirerek, takip sürecini daha anlaşılır hale getirmeyi amaçlar.
    """
    #cv2.line(img, line[0], line[1], (46,162,112), 3)

    height, width, _ = img.shape
    # remove tracked point from buffer if object is lost
    for key in list(data_deque):
      if key not in identities:
        data_deque.pop(key)

    for i, box in enumerate(bbox):
        x1, y1, x2, y2 = [int(i) for i in box]
        x1 += offset[0]
        x2 += offset[0]
        y1 += offset[1]
        y2 += offset[1]

        # code to find center of bottom edge
        center = (int((x2+x1)/ 2), int((y2+y2)/2))

        # get ID of object
        id = int(identities[i]) if identities is not None else 0

        # create new buffer for new object
        if id not in data_deque:  
          data_deque[id] = deque(maxlen= 64)
        color = compute_color_for_labels(object_id[i])
        obj_name = names[object_id[i]]
        label = '{}{:d}'.format("", id) + ":"+ '%s' % (obj_name)

        # add center to buffer
        data_deque[id].appendleft(center)
        UI_box(box, img, label=label, color=color, line_thickness=2)
        # draw trail
        for i in range(1, len(data_deque[id])):
            # check if on buffer value is none
            if data_deque[id][i - 1] is None or data_deque[id][i] is None:
                continue
            # generate dynamic thickness of trails
            thickness = int(np.sqrt(64 / float(i + i)) * 1.5)
            # draw trails
            cv2.line(img, data_deque[id][i - 1], data_deque[id][i], color, thickness)
    return img


class DetectionPredictor(BasePredictor):
    """
    nesne tespiti (object detection) ve nesne takibi (object tracking) yapabilen bir modeli kullanarak, 
    girdi görüntüler üzerinde nesneleri tespit etmeyi ve takip etmeyi sağlar. 
    Kodun temel amacı şu adımları gerçekleştirmektir:
    """

    def get_annotator(self, img):
        
        return Annotator(img, line_width=self.args.line_thickness, example=str(self.model.names))

    def preprocess(self, img):
        """
        preprocess fonksiyonu, giriş görüntüsünü modelin girdi formatına uygun hale getirir. 
        Giriş görüntüsü, önce PyTorch tensor'una dönüştürülür, ardından eğer modelin işlemesi gereken formatta değilse uygun formata çevrilir. 
        Özellikle, görüntü pikselleri 0-255 aralığından 0.0-1.0 aralığına normalize edilir.
        """
        img = torch.from_numpy(img).to(self.model.device)
        img = img.half() if self.model.fp16 else img.float()  # uint8 to fp16/32
        img /= 255  # 0 - 255 to 0.0 - 1.0
        return img

    def postprocess(self, preds, img, orig_img):
        """
        postprocess fonksiyonu, modelin çıktıları olan tespit sonuçlarını alır. 
        Bu tespit sonuçları, non_max_suppression fonksiyonu ile aynı sınıfa ait ve birbirine yakın olan tespitler arasında bir eleme yaparak düzenlenir. 
        Ardından, tespit sonuçları, orijinal görüntü boyutlarına göre ölçeklenir ve yuvarlanır (rounding).
        """
        preds = ops.non_max_suppression(preds,
                                        self.args.conf,
                                        self.args.iou,
                                        agnostic=self.args.agnostic_nms,
                                        max_det=self.args.max_det)

        for i, pred in enumerate(preds):
            shape = orig_img[i].shape if self.webcam else orig_img.shape
            pred[:, :4] = ops.scale_boxes(img.shape[2:], pred[:, :4], shape).round()

        return preds

    def write_results(self, idx, preds, batch):
        start_time = time.time()
        """
        write_results fonksiyonu, nesne tespiti sonuçlarını yazmak ve görselleştirmek için kullanılır. 
        Tespit sonuçları, orijinal görüntü üzerine çizilir ve nesnelerin etrafına sınırlayıcı kutular (bounding box) eklenir.
        Ayrıca, her sınıftaki nesne sayısı ve isimleri yazılarak çıktı olarak döndürülür.
        """
        p, im, im0 = batch
        all_outputs = []
        log_string = ""
        if len(im.shape) == 3:
            im = im[None]  # expand for batch dim
        self.seen += 1
        im0 = im0.copy()
        if self.webcam:  # batch_size >= 1
            log_string += f'{idx}: ' # 0: kısmı
            frame = self.dataset.count
        else:
            frame = getattr(self.dataset, 'frame', 0)

        self.data_path = p
        save_path = str(self.save_dir / p.name)  # im.jpg
        self.txt_path = str(self.save_dir / 'labels' / p.stem) + ('' if self.dataset.mode == 'image' else f'_{frame}')
        log_string += '%gx%g ' % im.shape[2:]  # print string 480x640 gibi
        log_string += str(start_time)
        self.annotator = self.get_annotator(im0)
        det = preds[idx]
        all_outputs.append(det)
        if len(det) == 0:
            return log_string
       
        for c in det[:, 5].unique():
            n = (det[:, 5] == c).sum()  # detections per class
            #log_string += f"{n} {self.model.names[int(c)]}{'s' * (n > 1)}, "
        # write
        gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
        xywh_bboxs = []
        confs = []
        oids = []
        outputs = []
       
        for *xyxy, conf, cls in reversed(det):
            x_c, y_c, bbox_w, bbox_h = xyxy_to_xywh(*xyxy)
            xywh_obj = [x_c, y_c, bbox_w, bbox_h]
            xywh_bboxs.append(xywh_obj)
            confs.append([conf.item()])
            oids.append(int(cls))
        xywhs = torch.Tensor(xywh_bboxs)
        confss = torch.Tensor(confs)
          
        outputs = deepsort.update(xywhs, confss, oids, im0)
        if len(outputs) > 0:
            bbox_xyxy = outputs[:, :4]
            identities = outputs[:, -2]
            object_id = outputs[:, -1]
            
            draw_boxes(im0, bbox_xyxy, self.model.names, object_id,identities)
   
        return log_string



"""
@hydra.main decorator'ı, Hydra kütüphanesini kullanarak yapılandırma dosyalarını işlemek için kullanılır. 
Hydra, yapılandırma dosyaları aracılığıyla program parametrelerini yönetmeyi sağlar. 
Bu sayede kodu çalıştırırken farklı parametre setlerini kolayca belirleyebilirsiniz.
""" 
@hydra.main(version_base=None, config_path=str(DEFAULT_CONFIG.parent), config_name=DEFAULT_CONFIG.name) 
def predict(cfg): # predict fonksiyonu, nesne tespiti ve takibi işlemlerini gerçekleştirir.
    init_tracker() # init_tracker() fonksiyonu, nesne takibi için gerekli olan deepsort adlı bir takipçi (tracker) modelini başlatır.
    cfg.model = cfg.model or "yolov8n.pt" # cfg.model ifadesi, kullanılacak olan nesne tespiti modelinin dosya yolunu belirler. Eğer cfg.model değeri belirtilmemişse veya None ise, varsayılan bir model olan "yolov8n.pt" kullanılır.
    cfg.imgsz = check_imgsz(cfg.imgsz, min_dim=2)  # check image size, cfg.imgsz ifadesi, nesne tespiti modelinin işleyeceği görüntülerin boyutunu belirler. Bu değer, check_imgsz fonksiyonu aracılığıyla kontrol edilir ve minimum boyut 2 olarak ayarlanır.
    cfg.source = cfg.source if cfg.source is not None else ROOT / "assets" # cfg.source ifadesi, nesne tespiti ve takibi modeline veri sağlayacak olan kaynağı belirler. Eğer bu ifade belirtilmemişse veya None ise, varsayılan olarak ROOT / "assets" (muhtemelen proje kök dizini altındaki "assets" klasörü) kullanılır.
    predictor = DetectionPredictor(cfg) # predictor değişkeni, DetectionPredictor sınıfından bir örnektir. Bu örnek, nesne tespiti ve takibi modelini çalıştırmak için gerekli olan parametreleri alır.
    print(predictor)
    predictor() # predictor() ifadesi, DetectionPredictor sınıfındaki __call__ metodunu çağırarak nesne tespiti ve takibi işlemlerini başlatır. Bu metod, görüntülerin işlenmesi, nesne tespiti, takip ve sonuçların görselleştirilmesi gibi adımları içerir.


if __name__ == "__main__":
    predict()
    cv2.destroyAllWindows()  # Tüm OpenCV pencerelerini kapat


