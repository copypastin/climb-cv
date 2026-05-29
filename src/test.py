import time
from pathlib import Path

import numpy as np

from climbcv.climbcv import climbcv



latest_landmarks = None
frames = []
root_path = Path(__file__).resolve().parents[1]
data_path = root_path / "data"
filename = None


"""
demo code for wokrking with landmarks using climbcv
hopefully this will be useful for testing building more complex features on top of climbcv-
without needing to worry about the camera and mediapipe setup

- aaron
"""


def main():
    global filename
    data_path.mkdir(parents=True, exist_ok=True)
    filename = data_path / f"landmarks_{int(time.time())}.npy"
    print(f"Writing landmarks to: {filename}")

    ccv = climbcv(enable_plotting=True)
    ccv.start(on_landmarks=on_landmarks)
    save_landmarks_to_file(frames)


def on_landmarks(landmarks):
    global latest_landmarks
    latest_landmarks = landmarks
    frames.append(np.asarray(landmarks, dtype=np.float32))

def save_landmarks_to_file(landmarks_batch):
    if not landmarks_batch:
        print("No landmarks to save")
        return

    np.save(filename, np.stack(landmarks_batch, axis=0))

if __name__ == "__main__":
    main()