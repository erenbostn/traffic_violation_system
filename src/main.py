import cv2
import time
from detector import VehicleDetector

VIDEO_PATH = "data/videos/input.mp4"
MODEL_PATH = "models/vehicle_model.pt"  # BURAYA KENDİ MODELİNİ KOY
OUTPUT_PATH = "outputs/annotated_video.mp4"


def main():
    cap = cv2.VideoCapture(VIDEO_PATH)
    assert cap.isOpened(), "Video açılamadı"

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    writer = cv2.VideoWriter(
        OUTPUT_PATH, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
    )

    detector = VehicleDetector(MODEL_PATH)

    prev_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        tracked_objects = detector.detect_and_track(frame)

        for obj in tracked_objects:
            x1, y1, x2, y2 = obj["bbox"]
            track_id = obj["track_id"]
            cls_id = obj["class_id"]

            label = f"ID {track_id} | CLS {cls_id}"

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(
                frame,
                label,
                (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                2,
            )

        curr_time = time.time()
        fps_text = f"FPS: {1 / (curr_time - prev_time):.2f}"
        prev_time = curr_time

        cv2.putText(
            frame, fps_text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2
        )

        writer.write(frame)
        cv2.imshow("Detection", frame)

        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    writer.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
