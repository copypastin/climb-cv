import time

from mediapipe.tasks.python.vision.pose_landmarker import PoseLandmarksConnections
from mediapipe.tasks.python.vision import drawing_utils as mp_drawing
import utils.config as config

INFER_SCALE = 0.5


def exo_live(cv, frame, result) -> None:

    if result.pose_landmarks:
        mp_drawing.draw_landmarks(
            frame,
            result.pose_landmarks[0],
            PoseLandmarksConnections.POSE_LANDMARKS,
        )

    curr_time = time.time()
    fps = 1 / (curr_time - config.prev_time) if config.prev_time else 0
    config.prev_time = curr_time
    cv.putText(frame, f"FPS: {int(fps)}", (10, 30), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    cv.imshow("MediaPipe Pose Landmarker", frame)

