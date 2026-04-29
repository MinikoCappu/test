import os
import cv2
import numpy as np
import threading
import queue
import sys

CAMERA_INDEX = 0

VIT_RKNN_PATH = "vit_quant.rknn"
COATNET_RKNN_PATH = "coatnet_clear.rknn"

FACE_PROTO = "deploy.prototxt"
FACE_MODEL = "res10_300x300_ssd_iter_140000.caffemodel"

CONF_THRESHOLD = 0.5
IMAGE_SIZE = 224

DROWSY_THRESHOLD = 0.5
MIN_EVENT_SEC = 3.63

QUEUE_MAXSIZE = 2
DEBUG_MODE = True
USE_FLOAT_NORMALIZATION = False


def validate_face_detector():
    if not os.path.exists(FACE_PROTO):
        raise FileNotFoundError(f"Не найден файл: {FACE_PROTO}")

    if not os.path.exists(FACE_MODEL):
        raise FileNotFoundError(f"Не найден файл: {FACE_MODEL}")

    model_size = os.path.getsize(FACE_MODEL)

    if model_size < 5_000_000:
        raise ValueError(
            f"Файл {FACE_MODEL} слишком маленький: {model_size:,} bytes"
        )


def load_face_detector():
    validate_face_detector()

    net = cv2.dnn.readNetFromCaffe(FACE_PROTO, FACE_MODEL)

    if net.empty():
        raise RuntimeError("OpenCV не смог загрузить face detector")

    dummy_frame = np.zeros((300, 300, 3), dtype=np.uint8)

    dummy_blob = cv2.dnn.blobFromImage(
        dummy_frame,
        scalefactor=1.0,
        size=(300, 300),
        mean=(104.0, 177.0, 123.0),
        swapRB=False,
        crop=False
    )

    net.setInput(dummy_blob)

    try:
        _ = net.forward()
    except Exception as e:
        raise RuntimeError(f"Ошибка face detector: {e}") from e

    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

    return net


class RKNNClassifier:
    def __init__(self, model_path, name="RKNNModel", core_mask=1):
        try:
            from rknnlite.api import RKNNLite
        except ImportError:
            print("rknn-lite-runtime не установлен")
            sys.exit(1)

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Не найден RKNN-файл: {model_path}")

        self.name = name
        self.rknn = RKNNLite()

        ret = self.rknn.load_rknn(model_path)
        if ret != 0:
            raise RuntimeError(f"Ошибка загрузки {self.name}, code={ret}")

        ret = self.rknn.init_runtime(core_mask=core_mask)
        if ret != 0:
            raise RuntimeError(f"Ошибка init_runtime для {self.name}, code={ret}")

    def preprocess(self, face_bgr):
        img = cv2.resize(face_bgr, (IMAGE_SIZE, IMAGE_SIZE))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        if USE_FLOAT_NORMALIZATION:
            img = img.astype(np.float32) / 255.0

            mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
            std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

            img = (img - mean) / std
            input_data = np.expand_dims(img.transpose(2, 0, 1), axis=0)
            input_data = np.ascontiguousarray(input_data, dtype=np.float32)
        else:
            input_data = np.expand_dims(img.transpose(2, 0, 1), axis=0)
            input_data = np.ascontiguousarray(input_data, dtype=np.uint8)

        return input_data

    def predict(self, face_bgr, debug=False):
        if face_bgr is None or face_bgr.size == 0:
            return 0.0

        input_data = self.preprocess(face_bgr)
        outputs = self.rknn.inference(inputs=[input_data])

        if outputs is None or len(outputs) == 0:
            raise RuntimeError(f"{self.name}: пустой output")

        out_tensor = outputs[0].copy()
        raw_value = float(out_tensor.flatten()[0])

        if raw_value < 0.0 or raw_value > 1.0:
            prob = 1.0 / (1.0 + np.exp(-raw_value))
        else:
            prob = raw_value

        return float(prob)

    def release(self):
        if hasattr(self, "rknn"):
            self.rknn.release()


def detect_face(net, frame):
    h, w = frame.shape[:2]

    blob = cv2.dnn.blobFromImage(
        frame,
        scalefactor=1.0,
        size=(300, 300),
        mean=(104.0, 177.0, 123.0),
        swapRB=False,
        crop=False
    )

    net.setInput(blob)
    detections = net.forward()

    best_face = None
    best_area = 0

    for i in range(detections.shape[2]):
        conf = float(detections[0, 0, i, 2])

        if conf < CONF_THRESHOLD:
            continue

        box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
        x1, y1, x2, y2 = box.astype(int)

        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(w, x2)
        y2 = min(h, y2)

        if x2 <= x1 or y2 <= y1:
            continue

        area = (x2 - x1) * (y2 - y1)

        if area > best_area:
            best_area = area
            best_face = (x1, y1, x2, y2)

    return best_face


class SmoothPredictor:
    def __init__(self, window=5, threshold=DROWSY_THRESHOLD):
        self.history = []
        self.lock = threading.Lock()
        self.window = window
        self.threshold = threshold

    def add(self, prob):
        with self.lock:
            self.history.append(prob)

            if len(self.history) > self.window:
                self.history.pop(0)

            avg_prob = sum(self.history) / len(self.history)
            cls = 1 if avg_prob >= self.threshold else 0

            return cls


class EventTracker:
    def __init__(self, fps, min_sec):
        self.fps = fps if fps and fps > 1 else 30.0
        self.min_sec = min_sec
        self.start_idx = None
        self.idx = 0
        self.lock = threading.Lock()

    def update(self, is_drowsy):
        with self.lock:
            self.idx += 1

            if is_drowsy and self.start_idx is None:
                self.start_idx = self.idx

            elif not is_drowsy and self.start_idx is not None:
                self._finalize(self.idx - 1)
                self.start_idx = None

    def force_end(self):
        with self.lock:
            if self.start_idx is not None:
                self._finalize(self.idx)
                self.start_idx = None

    def _finalize(self, end_idx):
        duration_sec = (end_idx - self.start_idx + 1) / self.fps

        if duration_sec >= self.min_sec:
            print(f"СОНЛИВОСТЬ ОБНАРУЖЕНА: длительность {duration_sec:.2f} сек")


class CameraThread:
    def __init__(self, idx=0):
        self.cap = cv2.VideoCapture(idx)

        if not self.cap.isOpened():
            raise RuntimeError(f"Не удалось открыть камеру {idx}")

        self.q = queue.Queue(maxsize=QUEUE_MAXSIZE)
        self.stop = False

        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def _run(self):
        while not self.stop:
            ret, frame = self.cap.read()

            if not ret:
                self.stop = True
                break

            if self.q.full():
                try:
                    self.q.get_nowait()
                except queue.Empty:
                    pass

            self.q.put((ret, frame))

    def read(self):
        return self.q.get()

    def close(self):
        self.stop = True
        self.cap.release()


class ProcessingThread:
    def __init__(
        self,
        cam,
        vit_model,
        coatnet_model,
        face_net,
        smoother,
        tracker
    ):
        self.cam = cam
        self.vit = vit_model
        self.coatnet = coatnet_model
        self.face_net = face_net
        self.smoother = smoother
        self.tracker = tracker

        self.stop = False
        self.display_frame = None
        self.error = None
        self.lock = threading.Lock()

        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def _run(self):
        try:
            while not self.stop:
                ret, frame = self.cam.read()

                if not ret:
                    break

                draw = frame.copy()

                status = "NO FACE"
                color = (255, 255, 255)
                is_drowsy = False

                box = detect_face(self.face_net, frame)

                if box is not None:
                    x1, y1, x2, y2 = box
                    roi = frame[y1:y2, x1:x2]

                    vit_prob = self.vit.predict(roi)
                    coat_prob = self.coatnet.predict(roi)

                    prob = vit_prob * 0.8 + coat_prob * 0.2

                    cls = self.smoother.add(prob)

                    if cls == 1:
                        is_drowsy = True
                        status = "DROWSY!"
                        color = (0, 0, 255)
                    else:
                        status = "AWAKE"
                        color = (0, 255, 0)

                    cv2.rectangle(draw, (x1, y1), (x2, y2), color, 2)

                self.tracker.update(is_drowsy)

                cv2.putText(
                    draw,
                    status,
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.0,
                    color,
                    2
                )

                with self.lock:
                    self.display_frame = draw

        except Exception as e:
            self.error = e
            print(f"Processing error: {e}")

    def get_frame(self):
        with self.lock:
            return self.display_frame

    def has_error(self):
        return self.error is not None

    def close(self):
        self.stop = True


def main():
    vit_model = None
    coatnet_model = None
    face_net = None
    cam = None
    proc = None
    tracker = None

    try:
        vit_model = RKNNClassifier(VIT_RKNN_PATH, name="ViT", core_mask=1)
        coatnet_model = RKNNClassifier(COATNET_RKNN_PATH, name="CoAtNet", core_mask=1)
        face_net = load_face_detector()
        cam = CameraThread(CAMERA_INDEX)

    except Exception as e:
        print(f"INITIALIZATION FAILED: {e}")
        sys.exit(1)

    fps = cam.cap.get(cv2.CAP_PROP_FPS)

    if fps is None or fps <= 1:
        fps = 30.0

    smoother = SmoothPredictor(window=5, threshold=DROWSY_THRESHOLD)

    tracker = EventTracker(fps=fps, min_sec=MIN_EVENT_SEC)

    proc = ProcessingThread(
        cam=cam,
        vit_model=vit_model,
        coatnet_model=coatnet_model,
        face_net=face_net,
        smoother=smoother,
        tracker=tracker
    )

    try:
        while True:
            if proc.has_error():
                print(f"Pipeline halted: {proc.error}")
                break

            frame = proc.get_frame()

            if frame is not None:
                cv2.imshow("Drowsiness Detection", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    except KeyboardInterrupt:
        pass

    finally:
        if proc:
            proc.close()

        if cam:
            cam.close()

        if tracker:
            tracker.force_end()

        if vit_model:
            vit_model.release()

        if coatnet_model:
            coatnet_model.release()

        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()