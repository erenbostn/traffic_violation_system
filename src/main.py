import cv2
import time
import csv
from detector import VehicleDetector
from traffic_light import TrafficLightDetector

VIDEO_PATH = "data/videos/input3.mp4"
MODEL_PATH = "models/vehicle_model.pt"
OUTPUT_PATH = "outputs/annotated_video3.mp4"

CLASS_NAMES = {0: "bus", 1: "car", 2: "motorcycle", 3: "truck"}


# -------------------------------------------------
# Geometry: which side of a line is a point on?
# -------------------------------------------------
def point_side(p, a, b):
    return (p[0] - a[0]) * (b[1] - a[1]) - (p[1] - a[1]) * (b[0] - a[0])


def main():
    cap = cv2.VideoCapture(VIDEO_PATH)
    assert cap.isOpened(), "Video açılamadı"

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    writer = cv2.VideoWriter(
        OUTPUT_PATH,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )

    # ==================================================
    # INTERACTIVE TRAFFIC LIGHT ROI
    # ==================================================
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
        "Traffic Light ROI", first_frame, fromCenter=False, showCrosshair=True
    )
    cv2.destroyWindow("Traffic Light ROI")

    traffic_light = TrafficLightDetector(roi)

    # ==================================================
    # INTERACTIVE STOP LINE (2 POINT LINE)
    # ==================================================
    stop_line_points = []

    def select_stop_line(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and len(stop_line_points) < 2:
            stop_line_points.append((x, y))

    ret, stop_frame = cap.read()
    if not ret:
        raise RuntimeError("Stop-line frame okunamadı")

    cv2.putText(
        stop_frame,
        "Click 2 points to define STOP LINE",
        (40, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 255),
        2,
    )

    cv2.imshow("Select STOP LINE", stop_frame)
    cv2.setMouseCallback("Select STOP LINE", select_stop_line)

    while len(stop_line_points) < 2:
        cv2.waitKey(1)

    cv2.destroyWindow("Select STOP LINE")
    line_p1, line_p2 = stop_line_points

    # Videoyu başa sar
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    # ==================================================
    detector = VehicleDetector(MODEL_PATH)

    # track_id -> previous line side
    vehicle_states = {}
    violations = set()

    csv_file = open("outputs/violations.csv", "w", newline="")
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(["track_id", "class", "violation"])

    prev_time = time.time()

    # ==================================================
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        light_state = traffic_light.get_light_state(frame)
        tracked_objects = detector.detect_and_track(frame)

        for obj in tracked_objects:
            x1, y1, x2, y2 = obj["bbox"]
            track_id = obj["track_id"]
            cls_id = obj["class_id"]

            class_name = CLASS_NAMES.get(cls_id, "unknown")

            # ------------------------------------------
            # VEHICLE FRONT POINT (ALT ORTA NOKTA)
            # ------------------------------------------
            front_x = (x1 + x2) // 2
            front_y = y2  # araç önü

            current_side = point_side(
                (front_x, front_y),
                line_p1,
                line_p2,
            )

            prev_side = vehicle_states.get(track_id)

            # ------------------------------------------
            # RED LIGHT VIOLATION RULE
            # ------------------------------------------
            if (
                light_state == "RED"
                and prev_side is not None
                and prev_side * current_side < 0
                and track_id not in violations
            ):
                violations.add(track_id)
                csv_writer.writerow([track_id, class_name, "RED_LIGHT"])

            vehicle_states[track_id] = current_side

            # ------------------------------------------
            # DRAW VEHICLE
            # ------------------------------------------
            if track_id in violations:
                label = f"ID {track_id} | {class_name} | RED LIGHT VIOLATION"
                color = (0, 0, 255)
            else:
                label = f"ID {track_id} | {class_name}"
                color = (0, 255, 0)

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.circle(frame, (front_x, front_y), 5, (0, 255, 255), -1)
            cv2.putText(
                frame,
                label,
                (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                2,
            )

        # ------------------------------------------
        # OVERLAYS
        # ------------------------------------------
        tl_color = (0, 0, 255) if light_state == "RED" else (0, 255, 0)
        cv2.putText(
            frame,
            f"Traffic Light: {light_state}",
            (20, 80),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            tl_color,
            2,
        )

        # Traffic light ROI
        cv2.rectangle(
            frame,
            (traffic_light.x1, traffic_light.y1),
            (traffic_light.x2, traffic_light.y2),
            (255, 0, 0),
            2,
        )

        # Stop line
        cv2.line(frame, line_p1, line_p2, (0, 0, 255), 3)

        # FPS
        curr_time = time.time()
        fps_text = f"FPS: {1 / (curr_time - prev_time):.2f}"
        prev_time = curr_time

        cv2.putText(
            frame,
            fps_text,
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 255),
            2,
        )

        writer.write(frame)
        cv2.imshow("Traffic Violation Detection", frame)

        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    writer.release()
    csv_file.close()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
