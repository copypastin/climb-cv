from pathlib import Path
import time

import cv2
import numpy as np

from mediapipe.tasks.python.core.base_options import BaseOptions
from mediapipe.tasks.python.vision.core.image import Image, ImageFormat
from mediapipe.tasks.python.vision.core.vision_task_running_mode import (
    VisionTaskRunningMode,
)
from mediapipe.tasks.python.vision.pose_landmarker import (
    PoseLandmarker,
    PoseLandmarkerOptions,
    PoseLandmarksConnections,
)
from mediapipe.tasks.python.vision import drawing_utils as mp_drawing

import matplotlib.pyplot as plt


MODEL_DIR = Path("models")
MODEL_PATH = MODEL_DIR / "pose_landmarker_heavy.task"
MODELS_PATHS = {
    "heavy": MODEL_DIR / "pose_landmarker_heavy.task",
    "full": MODEL_DIR / "pose_landmarker_full.task",
    "regular": MODEL_DIR / "pose_landmarker.task"
}

CURRENT_MODEL = MODELS_PATHS["full"]
CAPTURE_WIDTH = 640
CAPTURE_HEIGHT = 480
INFER_SCALE = 0.5


def open_camera() -> cv2.VideoCapture | None:
    for index in (0, 1, 2):
        cap = cv2.VideoCapture(index, cv2.CAP_AVFOUNDATION)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAPTURE_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAPTURE_HEIGHT)
        if cap.isOpened():
            print(f"Using camera index {index} (AVFoundation)")
            return cap
        cap.release()

    for index in (0, 1, 2):
        cap = cv2.VideoCapture(index)
        if cap.isOpened():
            print(f"Using camera index {index} (default backend)")
            return cap
        cap.release()

    return None


def main():

    options = PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(CURRENT_MODEL)),
        running_mode=VisionTaskRunningMode.VIDEO,
        min_pose_detection_confidence=0.2,
        min_pose_presence_confidence=0.2,
        min_tracking_confidence=0.2,
    )

    cap = open_camera()
    if cap is None:
        print("No camera could be opened.")
        print("On macOS, allow camera access for VS Code and Python in:")
        print("System Settings -> Privacy & Security -> Camera")
        return

    prev_time = 0.0

    with PoseLandmarker.create_from_options(options) as landmarker:
        while cap.isOpened():
            success, frame = cap.read()

            if INFER_SCALE != 1.0:
                infer_frame = cv2.resize(
                    frame,
                    (0, 0),
                    fx=INFER_SCALE,
                    fy=INFER_SCALE,
                    interpolation=cv2.INTER_AREA,
                )
            else:
                infer_frame = frame

            rgb_frame = cv2.cvtColor(infer_frame, cv2.COLOR_BGR2RGB)
            mp_image = Image(ImageFormat.SRGB, np.asarray(rgb_frame))
            timestamp_ms = int(time.monotonic() * 1000)
            result = landmarker.detect_for_video(mp_image, timestamp_ms)

            if result.pose_landmarks:
                mp_drawing.draw_landmarks(
                    frame,
                    result.pose_landmarks[0],
                    PoseLandmarksConnections.POSE_LANDMARKS,
                )

            curr_time = time.time()
            fps = 1 / (curr_time - prev_time) if prev_time else 0
            prev_time = curr_time
            cv2.putText(
                frame,
                f"FPS: {int(fps)}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255, 0, 0),
                2,
            )

            cv2.imshow("MediaPipe Pose Landmarker", frame)
            if cv2.waitKey(5) & 0xFF == 27:
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()