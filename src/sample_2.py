from climbcv.climbcv import climbcv


"""
demo code for loading replays

- aaron
"""

def main():
    ccv = climbcv(enable_plotting=True)
    ccv.replay("sample-data/sample_replay.npy")

def on_landmarks(landmarks):
    global latest_landmarks
    latest_landmarks = landmarks

if __name__ == "__main__":
    main()