# Traffic Violation System (Computer Vision)

Python tabanlı, video üzerinden trafik ihlali tespiti yapan bir bilgisayarlı görü projesi. Araç tespiti + takip ve trafik ışığı durumunu birleştirerek ihlalleri yakalar; her ihlal için “kanıt paketi” (kısa klip + `meta.json`) üretir.

> English summary: A Python computer-vision project that detects traffic violations from video by combining vehicle detection/tracking with traffic-light state, and exports evidence clips + metadata.

## Özellikler

- Araç tespiti ve takip (YOLO/Ultralytics + takip mantığı)
- Trafik ışığı ROI üzerinden durum tespiti (kırmızı/yeşil)
- İhlal anında kanıt üretimi: `outputs/evidence/violation_XXXX/clip.mp4` + `meta.json`
- `outputs/violations.csv` ile ihlal log’u
- ROI seçimi için yardımcı araç: `toolsselect_roi.py`

## Teknolojiler

- Python 3
- OpenCV (`opencv-python`)
- Ultralytics (`ultralytics`) (YOLO tabanlı)
- NumPy, Pandas, PyYAML, LAP

## Proje Yapısı

- `src/`: ana uygulama kodu (`main.py`, tespit/takip/kurallar)
- `configs/`: konfigürasyonlar (ör. `roi.yaml`)
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

2) Trafik ışığı için ROI seçin:

```bash
python toolsselect_roi.py
```

Çıktıdaki `x1, y1, x2, y2` değerlerini `configs/roi.yaml` içine girin.

3) Uygulamayı çalıştırın:

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

## Lisans

MIT License — detaylar için `LICENSE` dosyasına bakın.
