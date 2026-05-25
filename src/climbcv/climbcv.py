from pathlib import Path
from typing import Callable
import time
import cv2
import numpy as np
from multiprocessing import Process, Manager
from threading import Thread, current_thread

from mediapipe.tasks.python.core.base_options import BaseOptions
from mediapipe.tasks.python.vision.core.image import Image, ImageFormat
from mediapipe.tasks.python.vision.core.vision_task_running_mode import VisionTaskRunningMode

from mediapipe.tasks.python.vision.pose_landmarker import PoseLandmarker, PoseLandmarkerOptions

from .utils.rendering.plot_pose_live import plotting_process
from .utils.rendering.exo_live import exo_live
from .utils.angles.read_swift_lid import read_swift_lid

class climbcv:
        
    MODEL_DIR: Path = Path("./models")
    MODEL_PATH: Path = MODEL_DIR / "pose_landmarker_heavy.task"
    MODELS_PATHS: dict[str, Path] = {
        "heavy": MODEL_DIR / "pose_landmarker_heavy.task",
        "full": MODEL_DIR / "pose_landmarker_full.task",
        "regular": MODEL_DIR / "pose_landmarker.task"
    }

    # configurables
    CURRENT_MODEL: Path = MODELS_PATHS["heavy"]
    CAPTURE_WIDTH, CAPTURE_HEIGHT = 320, 240


    def __init__(self, model: str = "heavy", capture_width: int = 320, capture_height: int = 240, 
                delegate: BaseOptions.Delegate = BaseOptions.Delegate.GPU, 
                 enable_exo_live: bool = True, enable_plotting: bool = False, 
                 enable_mac_lid: bool = True, mac_lid_backend: str = "process"):
        
        
        self.model = model
        self.capture_width = capture_width
        self.capture_height = capture_height
        self.delegate = delegate
        self.enable_exo_live = enable_exo_live
        self.enable_plotting = enable_plotting
        self.enable_mac_lid = enable_mac_lid
        self.mac_lid_backend = mac_lid_backend

        self.cap: cv2.VideoCapture | None = None
        self.manager = None
        self.stop_event = None
        self.thread_lid: Process | Thread | None = None
        self.plot_proc: Process | None = None
        self.plot_queue = None
        self.lid_angle_value = None
        self.lid_timestamp = None
        self.raw_landmarks = None
        self._run_thread: Thread | None = None

        self.options: PoseLandmarkerOptions = PoseLandmarkerOptions(
            base_options=BaseOptions(
                model_asset_path=str(self.CURRENT_MODEL),
                delegate= self.delegate,
            ),
            running_mode=VisionTaskRunningMode.VIDEO,
            min_pose_detection_confidence=0.5,
            min_pose_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )

        print(f"Initialized climbcv with model: {self.model}, capture size: ({self.capture_width}x{self.capture_height}), delegate: {self.delegate}, enable_exo_live: {self.enable_exo_live}, enable_plotting: {self.enable_plotting}, enable_mac_lid: {self.enable_mac_lid}, mac_lid_backend: {self.mac_lid_backend}")


    def _cleanup(self) -> None:
        cap = getattr(self, "cap", None)
        if cap is not None:
            cap.release()

        try:
            cv2.destroyAllWindows()
        except Exception:
            pass

        worker = getattr(self, "thread_lid", None)
        stop_event = getattr(self, "stop_event", None)
        if stop_event is not None:
            try:
                stop_event.set()
            except Exception:
                pass
        if worker is not None and worker.is_alive():
            if isinstance(worker, Process):
                worker.terminate()
                worker.join(timeout=1)
            else:
                worker.join(timeout=1)

        plot_proc = getattr(self, "plot_proc", None)
        plot_queue = getattr(self, "plot_queue", None)
        if plot_proc is not None and plot_proc.is_alive():
            try:
                if plot_queue is not None:
                    plot_queue.put_nowait(None)
            except Exception:
                plot_proc.terminate()
            plot_proc.join(timeout=1)

        manager = getattr(self, "manager", None)
        if manager is not None:
            manager.shutdown()


    def __del__(self):
        self._cleanup()



    def __open_camera(self) -> cv2.VideoCapture | None:
        for index in np.arange(0, 4):
            cap = cv2.VideoCapture(index, cv2.CAP_AVFOUNDATION)
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.CAPTURE_WIDTH)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.CAPTURE_HEIGHT)
            
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

    


    def start(
        self,
        blocking: bool = True,
        on_landmarks: Callable[[list], None] | None = None,
    ) -> None:
        if not blocking:
            if self._run_thread is not None and self._run_thread.is_alive():
                return
            self._run_thread = Thread(
                target=self.start,
                kwargs={"blocking": True, "on_landmarks": on_landmarks},
                daemon=True,
            )
            self._run_thread.start()
            return
        if self._run_thread is not None and self._run_thread.is_alive():
            if current_thread() is not self._run_thread:
                raise RuntimeError("climbcv is already running")

        self.cap = self.__open_camera()

        if self.cap is None:
            raise RuntimeError("No camera could be opened.")
        
        # Multiprocessing manager for angle and plotting queue
        self.manager = Manager()
        self.stop_event = self.manager.Event()
        self.thread_lid = None
        self.lid_angle_value = self.manager.Value('d', 0.0)
        self.lid_timestamp = self.manager.Value('f', 0.0)

        # Plotting process and queue (not running on the main thread to avoid blocking)
        if self.enable_plotting:
            self.plot_queue = self.manager.Queue(maxsize=2)
            self.plot_proc = Process(target=plotting_process, args=(self.plot_queue,))
            self.plot_proc.start()

        with PoseLandmarker.create_from_options(self.options) as landmarker:
            while self.cap.isOpened():
                if self.stop_event is not None and self.stop_event.is_set():
                    break

                success, frame = self.cap.read()
                if not success:
                    continue

                frame = cv2.flip(frame, 1)
                rgba_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
                mp_image = Image(ImageFormat.SRGBA, np.asarray(rgba_frame))
                timestamp_ms = int(time.monotonic() * 1000)
                result = landmarker.detect_for_video(mp_image, timestamp_ms)

                has_pose = bool(result.pose_landmarks)

                # extract raw landmark data for external access and optional plotting
                if has_pose and result.pose_world_landmarks:
                    try:
                        landmarks_obj = result.pose_world_landmarks[0]
                        landmarks_iter = landmarks_obj.landmark if hasattr(landmarks_obj, "landmark") else landmarks_obj
                        self.raw_landmarks = [
                            (getattr(l, "visibility", 1.0), float(l.x), float(l.y), float(l.z))
                            for l in landmarks_iter
                        ]

                        self.last_three_landmarks = getattr(self, "last_three_landmarks", [])
                        self.last_three_landmarks.append(self.raw_landmarks)
                        if len(self.last_three_landmarks) > 3:
                            self.last_three_landmarks.pop(0)
                            self.average_landmarks = np.mean(self.last_three_landmarks)
                        else:
                            continue
                        
                        
                        if on_landmarks is not None:
                            try:
                                on_landmarks(self.average_landmarks)
                            except Exception:
                                pass
                    except Exception:
                        pass

                if self.enable_plotting and self.plot_queue is not None and self.raw_landmarks is not None:
                    try:
                        if self.plot_queue.full():
                            _ = self.plot_queue.get_nowait()
                    except Exception:
                        pass
                    try:
                        self.plot_queue.put_nowait(self.raw_landmarks)
                    except Exception:
                        pass


                # exo_live - draw pose landmarks and display the frame
                if self.enable_exo_live:
                    exo_live(cv2, frame, result, self.lid_angle_value, self.lid_timestamp)


                # spawn a separate worker to read the mac camera angle if the previous worker is not alive
                if self.enable_mac_lid and (self.thread_lid is None or not self.thread_lid.is_alive()):
                    if self.mac_lid_backend == "thread":
                        self.thread_lid = Thread(target=read_swift_lid, args=(self.lid_angle_value, self.lid_timestamp, self.stop_event), daemon=True)
                    else:
                        self.thread_lid = Process(target=read_swift_lid, args=(self.lid_angle_value, self.lid_timestamp, self.stop_event))
                    self.thread_lid.start()

                if cv2.waitKey(1) & 0xFF == 27:
                    break

        self._cleanup()


    def stop(self, timeout: float = 2.0) -> None:
        if self.stop_event is not None:
            self.stop_event.set()

        if self._run_thread is not None and self._run_thread.is_alive():
            self._run_thread.join(timeout=timeout)

        if self._run_thread is None or not self._run_thread.is_alive():
            self._cleanup()

