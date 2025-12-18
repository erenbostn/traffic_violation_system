import cv2

VIDEO_PATH = "data/videos/input.mp4"

cap = cv2.VideoCapture(VIDEO_PATH)
ret, frame = cap.read()
cap.release()

if not ret:
    raise RuntimeError("Video okunamadı")

# ROI seç
roi = cv2.selectROI(
    "Select Traffic Light ROI",
    frame,
    showCrosshair=True,
    fromCenter=False
)

x, y, w, h = roi
print(f"x1: {x}, y1: {y}, x2: {x+w}, y2: {y+h}")

cv2.destroyAllWindows()
