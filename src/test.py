from climbcv.climbcv import climbcv

latest_landmarks = None

"""
demo code for wokrking with landmarks using climbcv
hopefully this will be useful for testing building more complex features on top of climbcv-
without needing to worry about the camera and mediapipe setup

- aaron
"""


def main():
    ccv = climbcv()
    ccv.start(on_landmarks=on_landmarks)

def on_landmarks(landmarks):
    global latest_landmarks
    latest_landmarks = landmarks
    print("Received landmarks:", landmarks)


if __name__ == "__main__":
    main()