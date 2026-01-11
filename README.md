# Traffic Violation System (Computer Vision)

<div align="center">
  <video src="https://github.com/user-attachments/assets/ff5d2a10-ce2e-40f2-a753-9e867bd0bd02" width="100%" controls autoplay muted loop>
    Tarayıcınız video etiketini desteklemiyor.
  </video>
</div>




Python tabanlı, video üzerinden trafik ihlali tespiti yapan bir bilgisayarlı görü projesi. Araç tespiti + takip ve trafik ışığı durumunu birleştirerek ihlalleri yakalar; her ihlal için “kanıt paketi” (kısa klip + `meta.json`) üretir.

> English summary: A Python computer-vision project that detects traffic violations from video by combining vehicle detection/tracking with traffic-light state, and exports evidence clips + metadata.

## Özellikler

- Araç tespiti ve takip (YOLO/Ultralytics + takip mantığı)
- Trafik ışığı ROI üzerinden durum tespiti (kırmızı/yeşil) (ROI başlangıçta seçilir)
- İhlal anında kanıt üretimi: `outputs/evidence/violation_XXXX/clip.mp4` + `meta.json`
- `outputs/violations.csv` ile ihlal log’u

## Teknolojiler

- Python 3
- OpenCV (`opencv-python`)
- Ultralytics (`ultralytics`) (YOLO tabanlı)
- NumPy, Pandas, PyYAML, LAP

## Proje Yapısı

- `src/`: ana uygulama kodu (`main.py`, tespit/takip/kurallar)
- `models/`: model ağırlıkları (ör. `vehicle_model.pt`)
- `data/`: örnek video/harici veri dizini (repoya dahil edilmez, detay: `data/README.md`)
- `outputs/`: üretilen çıktılar (repoya dahil edilmez)

## Kurulum

```bash
python -m venv venv
# Windows
venv\Scripts\activate
pip install -r requirements.txt
```

## Kullanım

1) Video dosyanızı `data/videos/` altına koyun (örn. `data/videos/input.mp4`).

2) Uygulamayı çalıştırın (başlangıçta trafik ışığı ROI ve stop-line interaktif seçilir):

```bash
python src/main.py
```

## Çıktılar

- `outputs/annotated_video*.mp4`: çizimler eklenmiş çıktı videosu
- `outputs/violations.csv`: ihlal kayıtları
- `outputs/evidence/violation_XXXX/clip.mp4`: ihlal klibi
- `outputs/evidence/violation_XXXX/meta.json`: ihlal metadatası

## Notlar

- `data/` ve `outputs/` varsayılan olarak `.gitignore` ile repoya dahil edilmez (büyük/üretilen dosyalar).
- Model ağırlıkları `models/` altında tutulur. Daha büyük dosyalar için Git LFS önerilir.

## Katkı

Katkı yapmak isterseniz `CONTRIBUTING.md` dosyasına göz atın.


