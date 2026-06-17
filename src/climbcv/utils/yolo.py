from queue import Empty, Full

import numpy as np
from multiprocessing import Queue
from ultralytics import YOLO


def yolo_boxes_worker(
    model_path: str,
    imgsz: int,
    input_queue: Queue,
    output_queue: Queue,
    stop_event,
) -> None:
    """Read frames from the input queue, run YOLO, and publish scaled boxes."""
    model = YOLO(model_path)

    while not stop_event.is_set():
        try:
            payload = input_queue.get(timeout=0.1)
        except Empty:
            continue

        if payload is None:
            break

        frame, scale_x, scale_y = payload

        boxes_payload = []
        try:
            results = model.predict(source=frame, imgsz=imgsz, verbose=False)
            if results and len(results) > 0 and getattr(results[0], "boxes", None) is not None:
                det = results[0]
                boxes = det.boxes
                xyxy = boxes.xyxy.cpu().numpy() if hasattr(boxes.xyxy, "cpu") else np.asarray(boxes.xyxy)
                confs = boxes.conf.cpu().numpy() if getattr(boxes, "conf", None) is not None and hasattr(boxes.conf, "cpu") else np.asarray(getattr(boxes, "conf", []))
                classes = boxes.cls.cpu().numpy().astype(int) if getattr(boxes, "cls", None) is not None and hasattr(boxes.cls, "cpu") else np.asarray(getattr(boxes, "cls", []), dtype=int)
                names = det.names if isinstance(getattr(det, "names", None), dict) else {}

                for index, box in enumerate(xyxy):
                    x1, y1, x2, y2 = box.astype(int)
                    class_id = int(classes[index]) if index < len(classes) else -1
                    conf = float(confs[index]) if index < len(confs) else 0.0
                    label = names.get(class_id, str(class_id))
                    boxes_payload.append(
                        (
                            int(x1 * scale_x),
                            int(y1 * scale_y),
                            int(x2 * scale_x),
                            int(y2 * scale_y),
                            label,
                            conf,
                        )
                    )
        except Exception:
            boxes_payload = []

        try:
            if output_queue.full():
                _ = output_queue.get_nowait()
        except Empty:
            pass
        except Exception:
            pass

        try:
            output_queue.put_nowait(boxes_payload)
        except Full:
            pass
        except Exception:
            pass