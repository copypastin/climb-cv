import time

from mediapipe.tasks.python.vision.pose_landmarker import PoseLandmarksConnections
from mediapipe.tasks.python.vision import drawing_utils as mp_drawing
from .. import config
import numpy as np


def exo_live(cv, frame, result, lid_angle, lid_timestamp) -> None:

    curr_time, latency, fps = None, None, None
    body_angle = None

    if result.pose_landmarks:

        landmarks = result.pose_landmarks[0]

        # FPS calculation
        curr_time = time.time()
        fps: float = (1 / (curr_time - config.prev_time) if config.prev_time else 0)

        config.prev_time = curr_time
        cv.putText(frame, f"FPS: {int(fps)}", (10, 30), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

        # Drawing skeleton overlay
        mp_drawing.draw_landmarks(
            frame,
            landmarks,
            PoseLandmarksConnections.POSE_LANDMARKS,
            mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=1, circle_radius=1),
            mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=1)
        )

        if lid_angle.value is not None and lid_timestamp.value is not None:
            latency_ms = (curr_time - lid_timestamp.value) * 1000.0
            cv.putText(frame, f"Mac Camera Angle: {lid_angle.value} ({latency_ms:.2f}ms ago)", (10, 60), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

        # Calculating body tilt (requires mac camera angle)
            def xyz_helper(index):
                return np.array([landmarks[index].x, landmarks[index].y, landmarks[index].z])
            
            l_shoulder = xyz_helper(11)
            r_shoulder = xyz_helper(12)
            l_hip = xyz_helper(23)
            r_hip = xyz_helper(24)

            camera_theta = np.radians(lid_angle.value-90)  # Convert to radians and adjust for coordinate system
            
            # Original 'up' vector in MediaPipe (Y points down, so up is -1)
            original_y = -1.0
            original_z = 0.0
            
            # Apply X-axis rotation matrix
            new_x = 0.0
            new_y = (original_y * np.cos(camera_theta)) - (original_z * np.sin(camera_theta))
            new_z = (original_y * np.sin(camera_theta)) + (original_z * np.cos(camera_theta))
            
            reference_vector = np.array([new_x, new_y, new_z])
    
            mid_shoulder = (l_shoulder + r_shoulder) / 2.0
            mid_hip = (l_hip + r_hip) / 2.0
    
            body_vector = mid_shoulder - mid_hip
            body_dot = np.dot(body_vector, reference_vector)
            magnitude = np.linalg.norm(body_vector)
            magnitude_ref = np.linalg.norm(reference_vector)
    
            if magnitude == 0:
                body_angle = 0.0
            else:
                cos_angle = np.clip((body_dot / (magnitude * magnitude_ref)), -1.0, 1.0)
                body_angle = np.degrees(np.arccos(cos_angle))
    
            cv.line(frame, (int(mid_shoulder[0] * frame.shape[1]), int(mid_shoulder[1] * frame.shape[0])),
                    (int(mid_hip[0] * frame.shape[1]), int(mid_hip[1] * frame.shape[0])), (0, 255, 0), 2)
            cv.putText(frame, f"Body Tilt: {int(body_angle)} deg", (10, 90), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        else:
            cv.putText(frame, "Mac Camera Angle: n/a", (10, 60), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

        


        



    

    cv.imshow("MediaPipe Pose Landmarker", frame)


