import time

from mediapipe.tasks.python.vision.pose_landmarker import PoseLandmarksConnections
from mediapipe.tasks.python.vision import drawing_utils as mp_drawing
from .. import config

def exo_live(cv, frame, result, lid_angle, lid_timestamp) -> None:

    curr_time, latency, fps = None, None, None

    if result.pose_landmarks:
        mp_drawing.draw_landmarks(
            frame,
            result.pose_landmarks[0],
            PoseLandmarksConnections.POSE_LANDMARKS,
        )

    curr_time = time.time()
    fps: float = (1 / (curr_time - config.prev_time) if config.prev_time else 0)

    config.prev_time = curr_time
    cv.putText(frame, f"FPS: {int(fps)}", (10, 30), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)


    if lid_angle.value is not None and lid_timestamp.value is not None:
        latency = curr_time - lid_timestamp.value
        cv.putText(frame, f"Mac Camera Angle: {lid_angle.value} ({latency:.2f}ms ago)", (10, 60), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
    else:
        cv.putText(frame, "Mac Camera Angle: n/a", (10, 60), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
    cv.imshow("MediaPipe Pose Landmarker", frame)

