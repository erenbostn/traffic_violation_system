from ultralytics import YOLO


class VehicleDetector:
    def __init__(self, model_path):
        self.model = YOLO(model_path)

    def detect_and_track(self, frame):
        results = self.model.track(
            frame,
            persist=True,
            conf=0.4,
            iou=0.5,
            tracker="bytetrack.yaml",
            verbose=False,
        )

        tracked_objects = []

        for r in results:
            if r.boxes.id is None:
                continue

            boxes = r.boxes.xyxy.cpu().numpy()
            class_ids = r.boxes.cls.cpu().numpy()
            track_ids = r.boxes.id.cpu().numpy()

            for box, cls_id, track_id in zip(boxes, class_ids, track_ids):
                x1, y1, x2, y2 = map(int, box)

                tracked_objects.append(
                    {
                        "track_id": int(track_id),
                        "class_id": int(cls_id),
                        "bbox": (x1, y1, x2, y2),
                    }
                )

        return tracked_objects
