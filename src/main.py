import cv2
import time
import csv
from collections import deque
from pathlib import Path
import json
from detector import VehicleDetector
from traffic_light import TrafficLightDetector

VIDEO_PATH = "data/videos/input2.mp4"
MODEL_PATH = "models/vehicle_model.pt"
OUTPUT_PATH = "outputs/annotated_video3.mp4"

CLASS_NAMES = {0: "bus", 1: "car", 2: "motorcycle", 3: "truck"}
VALID_VEHICLE_CLASSES = ["car", "truck", "bus", "motorcycle"]
TRACK_ID_TO_VEHICLE_TYPE = {}
CLIP_PRE_FRAMES = 10
CLIP_POST_FRAMES = 10
FRAME_BUFFER_SIZE = CLIP_PRE_FRAMES
FRAME_BUFFER = deque(maxlen=FRAME_BUFFER_SIZE)
SOURCE_FPS = None
EVIDENCE_DIR = Path("outputs") / "evidence"


# -------------------------------------------------
# Geometry: which side of a line is a point on?
# -------------------------------------------------
def point_side(p, a, b):
    return (p[0] - a[0]) * (b[1] - a[1]) - (p[1] - a[1]) * (b[0] - a[0])


def write_clip(frames, output_path, fps, frame_size):
    output_path.parent.mkdir(parents=True, exist_ok=True)

    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        frame_size,
    )
    if not writer.isOpened():
        return False

    for img in frames:
        writer.write(img)

    writer.release()
    return True


def clone_frame_item(frame_item):
    return {
        "image": frame_item["image"].copy(),
        "frame_index": frame_item["frame_index"],
        "timestamp": frame_item["timestamp"],
        "tracks": dict(frame_item.get("tracks", {})),
    }


def write_meta_json(meta_path, meta):
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    with meta_path.open("w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)


def next_violation_id(evidence_dir):
    evidence_dir.mkdir(parents=True, exist_ok=True)

    max_id = 0
    for p in evidence_dir.glob("violation_*"):
        if not p.is_dir():
            continue
        suffix = p.name.replace("violation_", "", 1)
        if not suffix.isdigit():
            continue
        max_id = max(max_id, int(suffix))

    return max_id + 1


def draw_violation_overlay(
    image,
    *,
    bbox,
    track_id,
    vehicle_type,
    violation_type,
    timestamp,
    stop_line,
):
    (line_p1, line_p2) = stop_line

    cv2.line(image, line_p1, line_p2, (0, 0, 255), 3)

    if bbox is not None:
        x1, y1, x2, y2 = bbox
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 255), 2)

    ts_text = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(timestamp))
    violation_text = f"{violation_type.replace('_', ' ')} VIOLATION"
    label_main = f"ID {track_id} | {vehicle_type} | {violation_text}"

    cv2.putText(
        image,
        label_main,
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 0, 255),
        2,
    )
    cv2.putText(
        image,
        f"Timestamp: {ts_text}",
        (20, 75),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 0, 255),
        2,
    )


def main():
    global SOURCE_FPS

    cap = cv2.VideoCapture(VIDEO_PATH)
    assert cap.isOpened(), "Video açılamadı"

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    SOURCE_FPS = fps

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
    violation_snapshots = {}
    pending_clips = {}
    violation_id_counter = next_violation_id(EVIDENCE_DIR)

    csv_file = open("outputs/violations.csv", "w", newline="")
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(
        [
            "track_id",
            "class",
            "violation",
            "vehicle_type",
            "violation_type",
            "evidence_path",
        ]
    )

    prev_time = time.time()
    frame_index = 0

    # ==================================================
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        current_frame_index = frame_index
        timestamp = time.time()
        frame_for_evidence = frame.copy()

        light_state = traffic_light.get_light_state(frame)
        tracked_objects = detector.detect_and_track(frame)
        frame_tracks = {}
        violations_triggered = []

        for obj in tracked_objects:
            x1, y1, x2, y2 = obj["bbox"]
            track_id = obj["track_id"]
            cls_id = obj["class_id"]

            yolo_class_name = CLASS_NAMES.get(cls_id)

            vehicle_type = TRACK_ID_TO_VEHICLE_TYPE.get(track_id)
            if vehicle_type is None:
                if (
                    yolo_class_name is None
                    or yolo_class_name not in VALID_VEHICLE_CLASSES
                ):
                    continue
                TRACK_ID_TO_VEHICLE_TYPE[track_id] = yolo_class_name
                vehicle_type = yolo_class_name

            frame_tracks[track_id] = (x1, y1, x2, y2)

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
                if track_id not in violation_snapshots:
                    violation_snapshots[track_id] = {
                        "track_id": track_id,
                        "vehicle_type": vehicle_type,
                        "violation_type": "RED_LIGHT",
                        "frame_index": current_frame_index,
                        "timestamp": timestamp,
                    }

                violations_triggered.append(
                    {
                        "track_id": track_id,
                        "vehicle_type": vehicle_type,
                        "violation_type": "RED_LIGHT",
                        "bbox": (x1, y1, x2, y2),
                        "frame_index": current_frame_index,
                        "timestamp": timestamp,
                    }
                )

            vehicle_states[track_id] = current_side

            # ------------------------------------------
            # DRAW VEHICLE
            # ------------------------------------------
            if track_id in violations:
                label = f"ID {track_id} | {vehicle_type} | RED LIGHT VIOLATION"
                color = (0, 0, 255)
            else:
                label = f"ID {track_id} | {vehicle_type}"
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

        current_frame_item = {
            "image": frame_for_evidence,
            "frame_index": current_frame_index,
            "timestamp": timestamp,
            "tracks": frame_tracks,
        }

        for trigger in violations_triggered:
            track_id = trigger["track_id"]
            violation_id = violation_id_counter
            violation_id_counter += 1

            violation_dir = EVIDENCE_DIR / f"violation_{violation_id:04d}"
            clip_path = violation_dir / "clip.mp4"
            meta_path = violation_dir / "meta.json"
            pre_frames = [clone_frame_item(item) for item in FRAME_BUFFER] + [
                clone_frame_item(current_frame_item)
            ]
            pending_clips[track_id] = {
                "violation_id": violation_id,
                "vehicle_type": trigger["vehicle_type"],
                "violation_type": trigger["violation_type"],
                "violation_frame_index": trigger["frame_index"],
                "violation_timestamp": trigger["timestamp"],
                "clip_path": clip_path,
                "meta_path": meta_path,
                "pre_frames": pre_frames,
                "post_frames": [],
                "remaining_post_frames": CLIP_POST_FRAMES,
            }

        for pending_track_id, state in list(pending_clips.items()):
            if current_frame_index <= state["violation_frame_index"]:
                continue

            if state["remaining_post_frames"] > 0:
                state["post_frames"].append(clone_frame_item(current_frame_item))
                state["remaining_post_frames"] -= 1

            if state["remaining_post_frames"] == 0:
                clip_items = state["pre_frames"] + state["post_frames"]
                for item in clip_items:
                    draw_violation_overlay(
                        item["image"],
                        bbox=item["tracks"].get(pending_track_id),
                        track_id=pending_track_id,
                        vehicle_type=state["vehicle_type"],
                        violation_type=state["violation_type"],
                        timestamp=item["timestamp"],
                        stop_line=(line_p1, line_p2),
                    )

                clip_path = state["clip_path"]
                meta_path = state["meta_path"]
                clip_written = write_clip(
                    [item["image"] for item in clip_items],
                    clip_path,
                    SOURCE_FPS,
                    (width, height),
                )

                evidence_path = str(clip_path) if clip_written else ""
                write_meta_json(
                    meta_path,
                    {
                        "violation_id": state.get("violation_id"),
                        "track_id": pending_track_id,
                        "vehicle_type": state["vehicle_type"],
                        "violation_type": state["violation_type"],
                        "frame_index": state["violation_frame_index"],
                        "timestamp": state["violation_timestamp"],
                        "source_fps": SOURCE_FPS,
                        "clip_pre_frames": CLIP_PRE_FRAMES,
                        "clip_post_frames": CLIP_POST_FRAMES,
                        "clip_total_frames": len(clip_items),
                        "clip_path": evidence_path,
                    },
                )
                csv_writer.writerow(
                    [
                        pending_track_id,
                        state["vehicle_type"],
                        state["violation_type"],
                        state["vehicle_type"],
                        state["violation_type"],
                        evidence_path,
                    ]
                )
                del pending_clips[pending_track_id]

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

        FRAME_BUFFER.append(current_frame_item)
        frame_index += 1

    for pending_track_id, state in list(pending_clips.items()):
        clip_items = state["pre_frames"] + state["post_frames"]
        for item in clip_items:
            draw_violation_overlay(
                item["image"],
                bbox=item["tracks"].get(pending_track_id),
                track_id=pending_track_id,
                vehicle_type=state["vehicle_type"],
                violation_type=state["violation_type"],
                timestamp=item["timestamp"],
                stop_line=(line_p1, line_p2),
            )

        clip_path = state["clip_path"]
        meta_path = state["meta_path"]
        clip_written = write_clip(
            [item["image"] for item in clip_items],
            clip_path,
            SOURCE_FPS,
            (width, height),
        )

        evidence_path = str(clip_path) if clip_written else ""
        write_meta_json(
            meta_path,
            {
                "violation_id": state.get("violation_id"),
                "track_id": pending_track_id,
                "vehicle_type": state["vehicle_type"],
                "violation_type": state["violation_type"],
                "frame_index": state["violation_frame_index"],
                "timestamp": state["violation_timestamp"],
                "source_fps": SOURCE_FPS,
                "clip_pre_frames": CLIP_PRE_FRAMES,
                "clip_post_frames": CLIP_POST_FRAMES,
                "clip_total_frames": len(clip_items),
                "clip_path": evidence_path,
            },
        )
        csv_writer.writerow(
            [
                pending_track_id,
                state["vehicle_type"],
                state["violation_type"],
                state["vehicle_type"],
                state["violation_type"],
                evidence_path,
            ]
        )
        del pending_clips[pending_track_id]

    cap.release()
    writer.release()
    csv_file.close()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
