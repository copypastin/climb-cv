from pathlib import Path
from typing import Callable
import sys
import time
import cv2
import numpy as np
from multiprocessing import Process, Manager, Queue
from queue import Empty, Full
from threading import Thread, current_thread

from mediapipe.tasks.python.core.base_options import BaseOptions
from mediapipe.tasks.python.vision.core.image import Image, ImageFormat
from mediapipe.tasks.python.vision.core.vision_task_running_mode import VisionTaskRunningMode
from mediapipe.tasks.python.vision.pose_landmarker import PoseLandmarker, PoseLandmarkerOptions

import matplotlib.pyplot as plt

from .utils.yolo import yolo_boxes_worker
from .utils.rendering.plot_pose_live import plotting_process, plot_world_landmarks_points
from .utils.rendering.exo_live import exo_live
from .utils.angles.read_swift_lid import read_swift_lid
from .utils.smoothing import OneEuroFilter


saved_frames = []

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
    ENUM_MODELS = {"heavy", "full", "regular"}        


    def __init__(
        self,
        feed: str = "live",
        model: str = "heavy",
        capture_width: int = 320,
        capture_height: int = 240,
        delegate: BaseOptions.Delegate = BaseOptions.Delegate.GPU,
        enable_exo_live: bool = True,
        exo_live_yolo_every_n_frames: int = 4,
        enable_plotting: bool = False,
        enable_mac_lid: bool = True,
        smoothing_enabled: bool = True,
        smoothing_min_cutoff: float = 1.0,
        smoothing_beta: float = 0.4,
        smoothing_d_cutoff: float = 1.0,
        smoothing_visibility_threshold: float = 0.2,

    ):
        """Configure the video pipeline, pose model, and optional worker processes."""

        if model not in self.ENUM_MODELS:
            raise ValueError(f"Invalid model '{model}'. Valid options are: {self.ENUM_MODELS}")
        

        self.model = model
        self.feed = feed
        self.capture_width = capture_width
        self.capture_height = capture_height
        self.delegate = delegate
        self.enable_exo_live = enable_exo_live
        if exo_live_yolo_every_n_frames < 1:
            raise ValueError("exo_live_yolo_every_n_frames must be at least 1")
        self.exo_live_yolo_every_n_frames = exo_live_yolo_every_n_frames
        self.enable_plotting = enable_plotting
        self.enable_mac_lid = enable_mac_lid
        # The lid angle sensor relies on the Swift compiler and Apple hardware,
        # so it is only available on macOS. On other platforms it would spawn a
        # failing `swiftc` subprocess every frame, so disable it up front.
        if self.enable_mac_lid and sys.platform != "darwin":
            print("enable_mac_lid is only supported on macOS; disabling lid angle sensor.")
            self.enable_mac_lid = False
        self.smoothing_enabled = smoothing_enabled
        self.smoothing_visibility_threshold = smoothing_visibility_threshold

        self.cap: cv2.VideoCapture | None = None
        self.manager = None
        self.stop_event = None
        self.thread_lid: Process | Thread | None = None
        self.plot_proc: Process | None = None
        self.plot_queue = None
        self.lid_angle_value = None
        self.lid_timestamp = None
        self._yolo_proc: Process | None = None
        self._yolo_input_queue: Queue | None = None
        self._yolo_output_queue: Queue | None = None
        self._latest_yolo_boxes: list[tuple[int, int, int, int, str, float]] = []
        self.raw_landmarks = None
        self.smoothed_landmarks = None
        self._exo_live_frame_index = 0
        self._run_thread: Thread | None = None
        self._landmark_filter = OneEuroFilter(
            min_cutoff=smoothing_min_cutoff,
            beta=smoothing_beta,
            d_cutoff=smoothing_d_cutoff,
        )

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

        self.yolo_model_path = "models/hold_detection.pt"
        self.yolo_imgsz = 256
        self.yolo_input_width = 192
        
        print(
            "Initialized climbcv with the configs: "
            f"\n\tmodel={self.model} "
            f"\n\tfeed={self.feed}"
            f"\n\tcapture=({self.capture_width}x{self.capture_height})"
            f"\n\tdelegate={self.delegate}"
            f"\n\texo_live={self.enable_exo_live}"
            f"\n\texo_live_yolo_every_n_frames={self.exo_live_yolo_every_n_frames}"
            f"\n\tyolo_imgsz={self.yolo_imgsz}"
            f"\n\tyolo_input_width={self.yolo_input_width}"
            f"\n\tplotting={self.enable_plotting}"
            f"\n\tmac_lid={self.enable_mac_lid}"
        )


    def _draw_yolo_boxes(self, frame: np.ndarray) -> np.ndarray:
        if not self._latest_yolo_boxes:
            return frame

        for x1, y1, x2, y2, label, conf in self._latest_yolo_boxes:
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 1)
            cv2.putText(
                frame,
                f"{label} {conf:.2f}",
                (x1, max(20, y1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                1,
            )

        return frame


    def _initialize_runtime_workers(self) -> None:
        """Create shared state and start optional YOLO and plotting workers."""
        self.manager = Manager()
        self.stop_event = self.manager.Event()
        self.thread_lid = None
        self.lid_angle_value = self.manager.Value('d', 0.0)
        self.lid_timestamp = self.manager.Value('f', 0.0)
        self._latest_yolo_boxes = []

        if self.enable_exo_live:
            self._yolo_input_queue = Queue(maxsize=1)
            self._yolo_output_queue = Queue(maxsize=1)
            self._yolo_proc = Process(
                target=yolo_boxes_worker,
                args=(
                    self.yolo_model_path,
                    self.yolo_imgsz,
                    self._yolo_input_queue,
                    self._yolo_output_queue,
                    self.stop_event,
                ),
            )
            self._yolo_proc.start()

        if self.enable_plotting:
            self.plot_queue = self.manager.Queue(maxsize=2)
            self.plot_proc = Process(target=plotting_process, args=(self.plot_queue,))
            self.plot_proc.start()


    def _update_landmarks(
        self,
        result,
        timestamp_ms: int,
        on_landmarks: Callable[[np.ndarray], None] | None,
    ) -> None:
        """Extract world landmarks, smooth them, and notify the callback."""
        has_pose = bool(result.pose_landmarks)
        if not (has_pose and result.pose_world_landmarks):
            return

        try:
            landmarks_obj = result.pose_world_landmarks[0]
            landmarks_iter = landmarks_obj.landmark if hasattr(landmarks_obj, "landmark") else landmarks_obj
            self.raw_landmarks = [
                (getattr(l, "visibility", 1.0), float(l.x), float(l.y), float(l.z))
                for l in landmarks_iter
            ]

            raw_array = np.asarray(self.raw_landmarks, dtype=np.float32)
            if self.smoothing_enabled:
                if self.smoothed_landmarks is not None and raw_array.shape != self.smoothed_landmarks.shape:
                    self._landmark_filter.reset()
                    self.smoothed_landmarks = None

                if self.smoothing_visibility_threshold > 0.0 and self.smoothed_landmarks is not None:
                    low_vis = raw_array[:, 0] < self.smoothing_visibility_threshold
                    if np.any(low_vis):
                        raw_array[low_vis, 1:] = self.smoothed_landmarks[low_vis, 1:]

                self.smoothed_landmarks = self._landmark_filter.apply(raw_array, timestamp_ms / 1000.0)
            else:
                self.smoothed_landmarks = raw_array

            self.average_landmarks = self.smoothed_landmarks

            if on_landmarks is not None:
                try:
                    on_landmarks(self.smoothed_landmarks)
                except Exception:
                    pass
        except Exception:
            pass


    def _queue_plot_landmarks(self) -> None:
        """Send the latest landmarks to the plotting queue and persist them."""
        if self.plot_queue is None:
            return

        landmarks = self.smoothed_landmarks if self.smoothed_landmarks is not None else self.raw_landmarks
        if landmarks is None:
            return

        if self.smoothed_landmarks is not None:
            saved_frames.append(self.smoothed_landmarks)

        try:
            if self.plot_queue.full():
                _ = self.plot_queue.get_nowait()
        except Exception:
            pass

        try:
            self.plot_queue.put_nowait(landmarks)
        except Exception:
            pass


    def _queue_yolo_frame(self, frame: np.ndarray) -> None:
        """Resize and enqueue a frame for YOLO inference when due."""
        if self._yolo_input_queue is None:
            return

        self._exo_live_frame_index += 1
        if self._exo_live_frame_index % self.exo_live_yolo_every_n_frames != 0:
            return

        try:
            if self._yolo_input_queue.full():
                _ = self._yolo_input_queue.get_nowait()
        except Empty:
            pass
        except Exception:
            pass

        try:
            height, width = frame.shape[:2]
            resized_height = max(1, int(height * (self.yolo_input_width / width)))
            small_frame = cv2.resize(
                frame,
                (self.yolo_input_width, resized_height),
                interpolation=cv2.INTER_LINEAR,
            )
            scale_x = width / float(self.yolo_input_width)
            scale_y = height / float(resized_height)
            self._yolo_input_queue.put_nowait((small_frame, scale_x, scale_y))
        except Full:
            pass
        except Exception:
            pass


    def _drain_yolo_boxes(self) -> None:
        """Keep only the newest YOLO detections from the output queue."""
        if self._yolo_output_queue is None:
            return

        try:
            while True:
                self._latest_yolo_boxes = self._yolo_output_queue.get_nowait()
        except Empty:
            pass
        except Exception:
            pass


    def _render_exo_live(self, frame: np.ndarray, result) -> None:
        """Draw YOLO boxes and render the live exo view."""
        frame = self._draw_yolo_boxes(frame)
        exo_live(cv2, frame, result, self.lid_angle_value, self.lid_timestamp)


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

        yolo_proc = getattr(self, "_yolo_proc", None)
        yolo_input_queue = getattr(self, "_yolo_input_queue", None)
        if yolo_proc is not None and yolo_proc.is_alive():
            try:
                if yolo_input_queue is not None:
                    yolo_input_queue.put_nowait(None)
            except Exception:
                pass
            yolo_proc.join(timeout=1)
            if yolo_proc.is_alive():
                yolo_proc.terminate()
                yolo_proc.join(timeout=1)

        manager = getattr(self, "manager", None)
        if manager is not None:
            manager.shutdown()

        data_path = Path(__file__).resolve().parents[2] / "data"
        data_path.mkdir(parents=True, exist_ok=True)
        filename = data_path / f"landmarks_{int(time.time())}.npy"

        if len(saved_frames) > 0:
            np.save(filename, np.stack(saved_frames, axis=0))
            print(f"Saved {len(saved_frames)} frames to {filename}")



    def __del__(self):
        self._cleanup()



    def __open_camera(self) -> cv2.VideoCapture | None:
        if self.feed == "live":
            for index in np.arange(0, 4):
                cap = cv2.VideoCapture(index, cv2.CAP_AVFOUNDATION)
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.CAPTURE_WIDTH)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.CAPTURE_HEIGHT)

                if cap.isOpened():
                    print(f"Using camera index {index} (AVFoundation)")
                    return cap

            for index in np.arange(0, 4):
                cap = cv2.VideoCapture(index)
                if cap.isOpened():
                    print(f"Using camera index {index} (default backend)")
                    return cap
        else:
            cap = cv2.VideoCapture(self.feed)
            if cap.isOpened():
                print(f"Using video file {self.feed}")
                return cap
            
        cap.release()

        return None

    


    def start(
        self,
        blocking: bool = True,
        on_landmarks: Callable[[np.ndarray], None] | None = None,
    ) -> None:
        """Start capture and processing, optionally on a background thread."""
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

        self._initialize_runtime_workers()

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

                self._update_landmarks(result, timestamp_ms, on_landmarks)

                if self.enable_plotting:
                    self._queue_plot_landmarks()

                if self.enable_exo_live:
                    self._queue_yolo_frame(frame)
                    self._drain_yolo_boxes()
                    self._render_exo_live(frame, result)


                # spawn a separate worker to read the mac camera angle if the previous worker is not alive
                if self.enable_mac_lid and (self.thread_lid is None or not self.thread_lid.is_alive()):
                    self.thread_lid = Process(target=read_swift_lid, args=(self.lid_angle_value, self.lid_timestamp, self.stop_event))
                    self.thread_lid.start()

                if cv2.waitKey(1) & 0xFF == 27:
                    break

        self._cleanup()


    def stop(self, timeout: float = 2.0) -> None:
        """Request shutdown and wait for the background thread to finish."""
        if self.stop_event is not None:
            self.stop_event.set()

        if self._run_thread is not None and self._run_thread.is_alive():
            self._run_thread.join(timeout=timeout)

        if self._run_thread is None or not self._run_thread.is_alive():
            self._cleanup()

    def replay(self, filename: str) -> None:
        """Replay a saved landmark file as an animated 3D plot."""

        print(f"Replaying landmarks from file: {filename}")

        plt.ion()
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")

        try:
            data = np.load(filename)
            print("File loaded successfully.\n Total Frames:", len(data))

        except Exception as e:
            print(f"Error loading landmarks from file: {filename}")
            print(e)
            return

        plot_world_landmarks_points(ax,data)

        for frame in data:
            frame = np.asarray(frame, dtype=np.float32)
            plot_world_landmarks_points(ax, frame)
            fig.canvas.draw_idle()
            plt.pause(0.01)

