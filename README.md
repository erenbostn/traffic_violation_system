# traffic_violation_system

Trafik ihlali tespit sistemi için çalışma alanı.

## Kurulum

```bash
python -m venv venv
# Windows
venv\Scripts\activate
pip install -r requirements.txt
```

## ROI (Traffic Light) seçimi

Video dosyası: `data/videos/input.mp4`

```bash
python toolsselect_roi.py
```

Çıktıdaki `x1, y1, x2, y2` değerlerini `configs/roi.yaml` içine girin.

> Not: `data/` ve `outputs/` klasörleri varsayılan olarak `.gitignore` ile repoya dahil edilmez.