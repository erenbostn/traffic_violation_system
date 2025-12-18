import cv2
import time
from detector import VehicleDetector
from traffic_light import TrafficLightDetector

VIDEO_PATH = "data/videos/input3.mp4"
MODEL_PATH = "models/vehicle_model.pt"
OUTPUT_PATH = "outputs/annotated_video.mp4"

CLASS_NAMES = {0: "bus", 1: "car", 2: "motorcycle", 3: "truck"}


def main():
    cap = cv2.VideoCapture(VIDEO_PATH)
    assert cap.isOpened(), "Video açılamadı"

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    writer = cv2.VideoWriter(
        OUTPUT_PATH, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
    )

    # ===============================
    # INTERACTIVE ROI SELECTION
    # ===============================
    ret, first_frame = cap.read()
    if not ret:
        raise RuntimeError("İlk frame okunamadı")

    cv2.putText(
        first_frame,
        "Select TRAFFIC LIGHT ROI and press ENTER",
        (40, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 255),
        2,
    )

    roi = cv2.selectROI(
        "ROI Selection", first_frame, fromCenter=False, showCrosshair=True
    )
    cv2.destroyWindow("ROI Selection")

    traffic_light = TrafficLightDetector(roi)

    # Videoyu tekrar başa al
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    # ===============================
    detector = VehicleDetector(MODEL_PATH)
    prev_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # -------- Traffic light state --------
        light_state = traffic_light.get_light_state(frame)

        # -------- Vehicle tracking --------
        tracked_objects = detector.detect_and_track(frame)

        for obj in tracked_objects:
            x1, y1, x2, y2 = obj["bbox"]
            track_id = obj["track_id"]
            cls_id = obj["class_id"]

            class_name = CLASS_NAMES.get(cls_id, "unknown")
            label = f"ID {track_id} | {class_name}"

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

        # -------- Traffic light overlay --------
        color = (0, 0, 255) if light_state == "RED" else (0, 255, 0)

        cv2.putText(
            frame,
            f"Traffic Light: {light_state}",
            (20, 80),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            color,
            2,
        )

        # ROI kutusunu çiz (debug + güven)
        cv2.rectangle(
            frame,
            (traffic_light.x1, traffic_light.y1),
            (traffic_light.x2, traffic_light.y2),
            (255, 0, 0),
            2,
        )

        # -------- FPS --------
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
