# Copilot instructions for traffic_violation_system

Purpose: Make AI coding agents productive quickly in this repository (video-based traffic-violation detection).

- **Big picture:** This project processes video files in `data/videos/` and produces violation records in `outputs/violations.csv`.
  - Pipeline stages (separate responsibilities in `src/`): object detection (`detector.py`), tracking (`tracker.py`), lane logic (`lane.py`), speed estimation (`speed.py`), rule evaluation (`rules.py`), and traffic-signal handling (`traffic_light.py`). The intended entry point is `src/main.py` (currently empty).

- **Key files to inspect/modify:**
  - `toolsselect_roi.py`: interactive helper that opens a video frame, lets a developer select an ROI and prints `x1,y1,x2,y2`. Use this to populate `configs/roi.yaml`.
  - `configs/roi.yaml`: contains `traffic_light_roi` coordinates. Example keys: `x1`, `y1`, `x2`, `y2`.
  - `requirements.txt`: external deps (notably `ultralytics`, `opencv-python`, `numpy`, `pandas`, `pyyaml`).
  - `outputs/violations.csv`: expected CSV sink for violation records.

- **Observable conventions and patterns:**
  - Configs are simple YAML files in `configs/` (read with `pyyaml`). Keep keys flat and explicit (see `configs/roi.yaml`).
  - Video input lives under `data/videos/`. Use OpenCV `cv2.VideoCapture` for frame access (pattern shown in `toolsselect_roi.py`).
  - Detection is expected to use the `ultralytics` model API (installed via `requirements.txt`). If you add detection code, follow the simple pattern: load model once, call model on frames, then convert model outputs to bounding boxes and classes for downstream tracking/rules.
  - Outputs are written as CSV; prefer `pandas` to collate and persist violation rows.

- **Integration points and external expectations:**
  - `ultralytics` (YOLO) for object detection — ensure models are downloaded/available in the environment.
  - OpenCV for I/O and ROI selection (`toolsselect_roi.py`).
  - `pyyaml` for reading/writing `configs/*`.

- **Developer workflows (discoverable from repo):**
  - Install dependencies: `pip install -r requirements.txt` (create and activate a virtualenv first on Windows with `python -m venv .venv` then `.venv\Scripts\activate`).
  - Select traffic-light ROI interactively: `python toolsselect_roi.py` — copy printed `x1,y1,x2,y2` into `configs/roi.yaml` under `traffic_light_roi`.
  - There is no runnable `src/main.py` yet; inspect and compose pipeline using `src/*` modules. Prefer adding a single `process_video(video_path, cfg)` function in `src/main.py` that orchestrates: open video, run detector -> tracker -> per-frame rule checks -> append violations to `outputs/violations.csv`.

- **For AI agents editing this repo — concrete tips:**
  - Preserve `configs/roi.yaml` structure. Example from repo:

    ```yaml
    traffic_light_roi:
      x1: 1500
      y1: 91
      x2: 1518
      y2: 149
    ```

  - When adding detection code, follow `toolsselect_roi.py` patterns for frame acquisition and use `ultralytics` model inference in a per-frame loop.
  - Write violations with columns: timestamp/frame_id, object_id (from tracker), class, rule_id, description. Append to `outputs/violations.csv` via `pandas.DataFrame`.
  - Keep modules small and single-purpose (the repo already splits responsibilities by file name). If you add public functions, prefer names like `detect_frame()`, `update_tracker()`, `evaluate_rules()` to match the conceptual pipeline.

- **What not to assume:**
  - `src/*.py` files are mostly placeholders; do not assume existing helper functions or class names. Inspect each file before making cross-file calls.
  - There are no tests or CI config present — include small runnable examples when adding features.

If anything in this summary is unclear or you want the instructions tailored to a specific task (e.g., implement `src/main.py`, add detection using YOLOv8, or add a CI job), tell me which task and I'll iterate.
