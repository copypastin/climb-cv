from pathlib import Path
import time
import cv2
import numpy as np
from multiprocessing import Process, Manager

from mediapipe.tasks.python.core.base_options import BaseOptions
from mediapipe.tasks.python.vision.core.image import Image, ImageFormat
from mediapipe.tasks.python.vision.core.vision_task_running_mode import VisionTaskRunningMode

from mediapipe.tasks.python.vision.pose_landmarker import PoseLandmarker, PoseLandmarkerOptions

import matplotlib.pyplot as plt
from utils.rendering.plot_pose_live import plot_world_landmarks
from utils.rendering.exo_live import exo_live
from utils.angles.read_swift_lid import read_swift_lid


MODEL_DIR: Path = Path("./models")
MODEL_PATH: Path = MODEL_DIR / "pose_landmarker_heavy.task"
MODELS_PATHS: dict[str, Path] = {
    "heavy": MODEL_DIR / "pose_landmarker_heavy.task",
    "full": MODEL_DIR / "pose_landmarker_full.task",
    "regular": MODEL_DIR / "pose_landmarker.task"
}

# configurables
CURRENT_MODEL: Path = MODELS_PATHS["full"]
CAPTURE_WIDTH, CAPTURE_HEIGHT = 320, 240

def open_camera() -> cv2.VideoCapture | None:
    for index in np.arange(0, 4):
        cap = cv2.VideoCapture(index, cv2.CAP_AVFOUNDATION)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAPTURE_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAPTURE_HEIGHT)
        
        if cap.isOpened():
            print(f"Using camera index {index} (AVFoundation)")
            return cap
        cap.release()

    for index in np.arange(0, 4):
        cap = cv2.VideoCapture(index)
        if cap.isOpened():
            print(f"Using camera index {index} (default backend)")
            return cap
        cap.release()

    return None


def main():
    
    # SETUP

    # Pose Landmarker options
    options: PoseLandmarkerOptions = PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(CURRENT_MODEL)),
        running_mode=VisionTaskRunningMode.VIDEO,
        min_pose_detection_confidence=0.2,
        min_pose_presence_confidence=0.2,
        min_tracking_confidence=0.2,
    )

    # Configuration for matplotlib live plotting
    plt.ion()
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    # Camera setup
    cap = open_camera()

    if cap is None:
        print("No camera could be opened.")
        return
    
    # Multiprocessing manager for angle
    manager = Manager()
    thread_lid: Process | None = None
    lid_angle_value = manager.Value('d', 0.0)
    lid_timestamp = manager.Value('f', 0.0) 


    with PoseLandmarker.create_from_options(options) as landmarker:
        while cap.isOpened():
            success, frame = cap.read()

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = Image(ImageFormat.SRGB, np.asarray(rgb_frame))
            timestamp_ms = int(time.monotonic() * 1000)
            result = landmarker.detect_for_video(mp_image, timestamp_ms)


            if not result.pose_landmarks:
                continue

            # matlib section - dyanmicly plot the 3D pose landmarks
            plot_world_landmarks(plt, ax,result.pose_world_landmarks[0])

            # exo_live - draw pose landmarks and display the frame
            exo_live(cv2, frame, result, lid_angle_value, lid_timestamp)

            # spawn a separate process to read the mac camera angle if the previous process is not alive
            if thread_lid is None or not thread_lid.is_alive():
                thread_lid = Process(target=read_swift_lid, args=(lid_angle_value, lid_timestamp))
                thread_lid.start()

            if cv2.waitKey(5) & 0xFF == 27:
                break

    cap.release()
    cv2.destroyAllWindows()
    if thread_lid is not None and thread_lid.is_alive():
        thread_lid.terminate()
        thread_lid.join(timeout=1)
    manager.shutdown()


if __name__ == "__main__":
    main()
    