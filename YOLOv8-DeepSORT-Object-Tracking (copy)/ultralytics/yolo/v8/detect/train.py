# Ultralytics YOLO 🚀, GPL-3.0 license

from copy import copy

import hydra
import torch
import torch.nn as nn

from ultralytics.nn.tasks import DetectionModel
from ultralytics.yolo import v8
from ultralytics.yolo.data import build_dataloader
from ultralytics.yolo.data.dataloaders.v5loader import create_dataloader
from ultralytics.yolo.engine.trainer import BaseTrainer
from ultralytics.yolo.utils import DEFAULT_CONFIG, colorstr
from ultralytics.yolo.utils.loss import BboxLoss
from ultralytics.yolo.utils.ops import xywh2xyxy
from ultralytics.yolo.utils.plotting import plot_images, plot_results
from ultralytics.yolo.utils.tal import TaskAlignedAssigner, dist2bbox, make_anchors
from ultralytics.yolo.utils.torch_utils import de_parallel


# BaseTrainer python usage
class DetectionTrainer(BaseTrainer):

    def get_dataloader(self, dataset_path, batch_size, mode="train", rank=0):
        """ 
            Bu Python kodu, bir nesne tespit modelinin eğitimini gerçekleştiren bir eğitim sınıfını tanımlamaktadır. Kod, genel bir eğitim altyapısı sağlamak üzere bir temel eğitim sınıfını genişletmektedir. Detayları anlamak için her bir metodun ne yaptığını inceleyelim:

            get_dataloader Metodu:
            Bu metod, veri kümesini yükleyen ve uygun bir şekilde yapılandıran bir veri yükleyici (dataloader) oluşturan bir yöntemdir.
            dataset_path: Veri kümesinin dosya yolu.
            batch_size: Bir eğitim iterasyonunda kullanılacak örnek sayısı.
            mode: Veri yükleyicinin çalışma modu, genellikle "train" veya "test".
            rank: İşlem sırası (MPI tabanlı çoklu işlemde kullanılır).
            gs: Stride değeri, modelin maksimum stride'ını kontrol eder.
            hyp: Hiperparametreler sözlüğü.
            augment: Veri artırma (data augmentation) yapılıp yapılmayacağını belirleyen bir bayrak.
            cache: Veri kümesinin önceden önbelleğe alınıp alınmayacağını belirleyen bir bayrak.
            pad: Veri artırma sırasında kullanılacak kenar dolgusu (padding) miktarı.
            rect: Dikdörtgen veri artırma modunu kontrol eden bir bayrak.
            workers: Veri yükleyici tarafından kullanılacak işçi sayısı.
            close_mosaic: Mozaik veri artırma modunu kontrol eden bir bayrak.
            prefix: Mesajların önüne eklenen bir ön ek.
            shuffle: Veri kümesinin karıştırılıp karıştırılmayacağını belirleyen bir bayrak.
            seed: Rastgele sayı üreteci için başlangıç ​​tohumu.
        """
        # TODO: manage splits differently
        # calculate stride - check if model is initialized
        gs = max(int(de_parallel(self.model).stride.max() if self.model else 0), 32)
        return create_dataloader(path=dataset_path,
                                 imgsz=self.args.imgsz,
                                 batch_size=batch_size,
                                 stride=gs,
                                 hyp=dict(self.args),
                                 augment=mode == "train",
                                 cache=self.args.cache,
                                 pad=0 if mode == "train" else 0.5,
                                 rect=self.args.rect,
                                 rank=rank,
                                 workers=self.args.workers,
                                 close_mosaic=self.args.close_mosaic != 0,
                                 prefix=colorstr(f'{mode}: '),
                                 shuffle=mode == "train",
                                 seed=self.args.seed)[0] if self.args.v5loader else \
            build_dataloader(self.args, batch_size, img_path=dataset_path, stride=gs, rank=rank, mode=mode)[0]
  

    def preprocess_batch(self, batch):
        """Batch verilerini ön işleme adımını gerçekleştiren bir metod. Görüntüleri cihaz hafızasına gönderir, veri türünü float'a dönüştürür ve 0 ile 255 arasındaki değerleri 0 ile 1 arasına ölçekler."""
        batch["img"] = batch["img"].to(self.device, non_blocking=True).float() / 255
        return batch

    def set_model_attributes(self):
        """Modelin özelliklerini ayarlayan bir metod. Bu metod, modelin kaç tane deteksiyon katmanına sahip olduğunu ve bu katman sayısına göre bazı hiperparametreleri ölçekler."""
        nl = de_parallel(self.model).model[-1].nl  # number of detection layers (to scale hyps)
        self.args.box *= 3 / nl  # scale to layers
        # self.args.cls *= self.data["nc"] / 80 * 3 / nl  # scale to classes and layers
        self.args.cls *= (self.args.imgsz / 640) ** 2 * 3 / nl  # scale to image size and layers
        self.model.nc = self.data["nc"]  # attach number of classes to model
        self.model.args = self.args  # attach hyperparameters to model
        # TODO: self.model.class_weights = labels_to_class_weights(dataset.labels, nc).to(device) * nc
        self.model.names = self.data["names"]

    def get_model(self, cfg=None, weights=None, verbose=True):
        """ Yeni bir nesne tespiti modeli oluşturan bir metod. Eğer belirtilmişse, ağırlıkları (weights) yükler. Model, DetectionModel adlı bir sınıftan türetilmiş gibi görünüyor."""
        model = DetectionModel(cfg, ch=3, nc=self.data["nc"], verbose=verbose)
        if weights:
            model.load(weights)

        return model

    def get_validator(self):
        """Modelin doğrulama (validation) işlemini gerçekleştirmek için bir doğrulama sınıfını döndüren bir metod. Bu metodun geri döndürdüğü sınıf, modelin doğrulama işleminde kullanılacak."""
        self.loss_names = 'box_loss', 'cls_loss', 'dfl_loss'
        return v8.detect.DetectionValidator(self.test_loader,
                                            save_dir=self.save_dir,
                                            logger=self.console,
                                            args=copy(self.args))

    def criterion(self, preds, batch):
        """Bu fonksiyon, modelin tahminlerini ve eğitim veri yükünü alır ve eğitim kaybını hesaplamak için kullanılır. compute_loss adında bir Loss sınıfının bir örneğini kullanıyor gibi görünüyor. Bu, özel bir kayıp fonksiyonunu içerebilir."""
        if not hasattr(self, 'compute_loss'):
            self.compute_loss = Loss(de_parallel(self.model))
        return self.compute_loss(preds, batch)

    def label_loss_items(self, loss_items=None, prefix="train"):
        """Bu fonksiyon, eğitim kaybını etiketleyerek (label) daha okunabilir ve takip edilebilir bir formatta döndürür. Özellikle, bu kaybın her bir bileşeninin isimlendirilmiş bir sözlük olarak döndüğü görünüyor."""
        """
        Returns a loss dict with labelled training loss items tensor
        """
        # Not needed for classification but necessary for segmentation & detection
        keys = [f"{prefix}/{x}" for x in self.loss_names]
        if loss_items is not None:
            loss_items = [round(float(x), 5) for x in loss_items]  # convert tensors to 5 decimal place floats
            return dict(zip(keys, loss_items))
        else:
            return keys

    def progress_string(self):
        """Bu fonksiyon, eğitim sırasında eğitim ilerlemesini ekrana basmak için kullanılan bir dizeyi oluşturur. Epoch sayısı, GPU belleği kullanımı, kayıp isimleri, örnek sayısı ve boyut gibi bilgiler içerir."""
        return ('\n' + '%11s' *
                (4 + len(self.loss_names))) % ('Epoch', 'GPU_mem', *self.loss_names, 'Instances', 'Size')

    def plot_training_samples(self, batch, ni):
        """Bu fonksiyon, eğitim örneklerinin bir kısmını görselleştirmek için kullanılır. Genellikle eğitim sırasında modelin nasıl performans gösterdiğini kontrol etmek için kullanışlıdır."""
        plot_images(images=batch["img"],
                    batch_idx=batch["batch_idx"],
                    cls=batch["cls"].squeeze(-1),
                    bboxes=batch["bboxes"],
                    paths=batch["im_file"],
                    fname=self.save_dir / f"train_batch{ni}.jpg")

    def plot_metrics(self):
        """Bu fonksiyon, eğitim sırasında elde edilen metrikleri (örneğin, doğruluk, kayıp gibi) görselleştirmek için kullanılır. plot_results fonksiyonunu çağırarak bu görselleştirmeyi sağlıyor olabilir."""
        plot_results(file=self.csv)  # save results.png


# Criterion class for computing training losses
class Loss:

    def __init__(self, model):  # model must be de-paralleled
        """
        Bu kod, YOLOv8 modelinin özel bir kayıp fonksiyonu (Loss sınıfı) için bir implementasyonu içerir. İşte bu sınıfın temel özelliklerinin adım adım açıklamaları:

        __init__ Metodu:
        device = next(model.parameters()).device: Modelin aygıtını alır. Yani, modelin parametrelerinden birinin aygıtını kullanır.
        h = model.args: Modelin hiperparametrelerini alır.
        m = model.model[-1]: Modelin sonundaki Detect() modülünü alır.
        self.bce = nn.BCEWithLogitsLoss(reduction='none'): Binary Cross Entropy (BCE) kaybını tanımlar. Bu kayıp, ikili sınıflandırma görevleri için kullanılır.
        self.hyp = h: Hiperparametreleri saklar.
        self.stride = m.stride: Modelin adım değerlerini saklar.
        self.nc = m.nc: Modelin çıkış sınıf sayısını saklar.
        self.no = m.no: Modelin çıkış hedef sayısını saklar.
        self.reg_max = m.reg_max: Modelin reg_max özelliğini saklar. Eğer bu değer 1'den büyükse, DFL (Dynamic Feature Learning) kullanılacak demektir.
        self.device = device: Modelin aygıtını saklar.
        self.use_dfl = m.reg_max > 1: DFL kullanılıp kullanılmayacağını belirler.
        self.assigner = TaskAlignedAssigner(topk=10, num_classes=self.nc, alpha=0.5, beta=6.0): Görev tabanlı bir atama yapıcı oluşturur. Bu, hedefleri tahminlere hizalamak için kullanılır.
        self.bbox_loss = BboxLoss(m.reg_max - 1, use_dfl=self.use_dfl).to(device): Hesaplanan kutu kaybını (BboxLoss) oluşturur. Bu, koordinatları ve boyutları düzeltmeye yönelik bir kayıp olabilir.
        self.proj = torch.arange(m.reg_max, dtype=torch.float, device=device): Modelin reg_max özelliğine göre bir dizi oluşturur.
        """

        device = next(model.parameters()).device  # get model device
        h = model.args  # hyperparameters

        m = model.model[-1]  # Detect() module
        self.bce = nn.BCEWithLogitsLoss(reduction='none')
        self.hyp = h
        self.stride = m.stride  # model strides
        self.nc = m.nc  # number of classes
        self.no = m.no
        self.reg_max = m.reg_max
        self.device = device

        self.use_dfl = m.reg_max > 1
        self.assigner = TaskAlignedAssigner(topk=10, num_classes=self.nc, alpha=0.5, beta=6.0)
        self.bbox_loss = BboxLoss(m.reg_max - 1, use_dfl=self.use_dfl).to(device)
        self.proj = torch.arange(m.reg_max, dtype=torch.float, device=device)

    def preprocess(self, targets, batch_size, scale_tensor):
        # Hedeflerin boyutu (targets.shape[0]) 0 ise (yani hiç hedef yoksa),
        # bir tensör oluşturulur, ancak bu tensörün içeriği sıfırdır.
        if targets.shape[0] == 0:
            out = torch.zeros(batch_size, 0, 5, device=self.device)
        else:
            # Hedeflerin ilk sütunu (targets[:, 0]) görsel indekslerini içerir.
            i = targets[:, 0]  # image index

            # Her görselde kaç hedef olduğunu ve her bir görseldeki hedef sayısını hesapla.
            _, counts = i.unique(return_counts=True)

            # Her bir görsel için maksimum hedef sayısına sahip bir tensör oluştur.
            out = torch.zeros(batch_size, counts.max(), 5, device=self.device)

            # Her bir görsel için döngü.
            for j in range(batch_size):
                # İlgili görseldeki hedefleri seç.
                matches = i == j
                n = matches.sum()

                # Eğer görselde en az bir hedef varsa, bu hedefleri çıkartılan tensöre kopyala.
                if n:
                    out[j, :n] = targets[matches, 1:]

            # Dönüştürülen hedeflerin ikinci sütunundan itibaren (1:5 arası) koordinatları xyxy biçimine çevir.
            out[..., 1:5] = xywh2xyxy(out[..., 1:5].mul_(scale_tensor))

        # Hazırlanan hedef tensörünü döndür.
        return out


    def bbox_decode(self, anchor_points, pred_dist):
        if self.use_dfl:
            # pred_dist'in boyutları: (batch, anchors, channels)
            b, a, c = pred_dist.shape  # batch, anchors, channels
            
            # DFL kullanılıyorsa, softmax aktivasyonu uygula ve projeksiyon matrisi ile çarp.
            pred_dist = pred_dist.view(b, a, 4, c // 4).softmax(3).matmul(self.proj.type(pred_dist.dtype))

            # pred_dist = pred_dist.view(b, a, c // 4, 4).transpose(2,3).softmax(3).matmul(self.proj.type(pred_dist.dtype))
            # pred_dist = (pred_dist.view(b, a, c // 4, 4).softmax(2) * self.proj.type(pred_dist.dtype).view(1, 1, -1, 1)).sum(2)
        """Bu kod, özellikle use_dfl bayrağı kullanıldığında geçerli olan bir özel durumu ele alır. Eğer use_dfl True ise, pred_dist üzerinde softmax aktivasyonunu uygular ve projeksiyon matrisi ile çarpar. Ardından, dist2bbox fonksiyonunu kullanarak bu tahmin dağılımlarını bbox (bounding box) koordinatlarına çevirir.

        dist2bbox fonksiyonu, bir mesafe tahmini ve referans noktaları (anchor noktalar) kullanarak bbox koordinatlarını hesaplamak için genellikle kullanılan bir yöntemdir. Bu fonksiyon, verilen mesafe tahmini ve anchor noktalarıyla ilişkilendirilmiş bbox'ları döndürür. Bu, genellikle nesne algılama modellerinin tahminlerini yorumlamak için kullanılan bir adımdır."""
        # dist2bbox fonksiyonunu kullanarak pred_dist'ten bbox koordinatlarını çıkar.
        return dist2bbox(pred_dist, anchor_points, xywh=False)

    
    def __call__(self, preds, batch):
        """Bu fonksiyon, detaylı bir nesne algılama modelinin eğitimi sırasında kullanılan kayıp fonksiyonunu uygular. Çeşitli kayıp bileşenlerini (bbox loss, cls loss, dfl loss) hesaplamak ve bu kayıpları hiperparametrelerle çarparak toplam kaybı döndürmektedir."""
        loss = torch.zeros(3, device=self.device)  # box, cls, dfl

        # Eldeki tahminlerin yapılarına göre uygun şekilde ayırma işlemi
        feats = preds[1] if isinstance(preds, tuple) else preds
        pred_distri, pred_scores = torch.cat([xi.view(feats[0].shape[0], self.no, -1) for xi in feats], 2).split(
            (self.reg_max * 4, self.nc), 1)

        # Pred_scores ve pred_distri tensorlerini uygun şekilde düzenleme
        pred_scores = pred_scores.permute(0, 2, 1).contiguous()
        pred_distri = pred_distri.permute(0, 2, 1).contiguous()

        dtype = pred_scores.dtype
        batch_size = pred_scores.shape[0]

        # Giriş resminin boyutları
        imgsz = torch.tensor(feats[0].shape[2:], device=self.device, dtype=dtype) * self.stride[0]  # image size (h,w)

        # Anchor noktalarını ve stride tensorünü oluşturma
        anchor_points, stride_tensor = make_anchors(feats, self.stride, 0.5)

        # Hedefleri uygun şekilde işleme
        targets = torch.cat((batch["batch_idx"].view(-1, 1), batch["cls"].view(-1, 1), batch["bboxes"]), 1)
        targets = self.preprocess(targets.to(self.device), batch_size, scale_tensor=imgsz[[1, 0, 1, 0]])
        gt_labels, gt_bboxes = targets.split((1, 4), 2)  # cls, xyxy
        mask_gt = gt_bboxes.sum(2, keepdim=True).gt_(0)

        # Tahmini bbox'ları çıkarma
        pred_bboxes = self.bbox_decode(anchor_points, pred_distri)  # xyxy, (b, h*w, 4)

        # Tahmini ve gerçek hedef bbox'larını ve skorlarını atama
        _, target_bboxes, target_scores, fg_mask, _ = self.assigner(
            pred_scores.detach().sigmoid(), (pred_bboxes.detach() * stride_tensor).type(gt_bboxes.dtype),
            anchor_points * stride_tensor, gt_labels, gt_bboxes, mask_gt)

        target_bboxes /= stride_tensor
        target_scores_sum = target_scores.sum()

        # Sınıf kaybı (cls loss)
        loss[1] = self.bce(pred_scores, target_scores.to(dtype)).sum() / target_scores_sum  # BCE

        # Bbox kaybı ve DFL kaybı (bbox loss, dfl loss)
        if fg_mask.sum():
            loss[0], loss[2] = self.bbox_loss(pred_distri, pred_bboxes, anchor_points, target_bboxes, target_scores,
                                            target_scores_sum, fg_mask)

        # Hiperparametreler ile çarparak toplam kaybı hesaplama
        loss[0] *= self.hyp.box  # box gain
        loss[1] *= self.hyp.cls  # cls gain
        loss[2] *= self.hyp.dfl  # dfl gain

        return loss.sum() * batch_size, loss.detach()  #loss(box, cls, dfl) Toplam kaybı ve ayrık kayıpları döndürme



# Hydra kütüphanesinden @hydra.main decorator'ını ekleyerek Hydra'yı kullanmaya başlıyoruz.
# Hydra, yapılandırma yönetimi için kullanılır.
@hydra.main(version_base=None, config_path=str(DEFAULT_CONFIG.parent), config_name=DEFAULT_CONFIG.name)
def train(cfg):
    # Config dosyalarında belirtilmeyen varsayılan değerleri atıyoruz.
    cfg.model = cfg.model or "yolov8n.yaml"
    cfg.data = cfg.data or "coco128.yaml"

    # Ultralytics kütüphanesinin YOLO sınıfını kullanarak YOLOv8 modelini oluşturuyoruz.
    from ultralytics import YOLO
    model = YOLO(cfg.model)

    # YOLO modelini eğitiyoruz, bu adım Ultralytics kütüphanesinin özel bir fonksiyonu olan train() ile yapılıyor.
    model.train(**cfg)


if __name__ == "__main__":
    """
    CLI usage:
    python ultralytics/yolo/v8/detect/train.py model=yolov8n.yaml data=coco128 epochs=100 imgsz=640

    TODO:
    yolo task=detect mode=train model=yolov8n.yaml data=coco128.yaml epochs=100
    """
    # Eğer bu script doğrudan çalıştırılıyorsa, train() fonksiyonunu çağırarak eğitimi başlatıyoruz.
    train()

