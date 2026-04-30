import os
import cv2
import numpy as np
import threading
import queue
import sys
import sqlite3
import uuid
import time
from datetime import datetime


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

DEBUG_MODE = False
USE_FLOAT_NORMALIZATION = False

DB_PATH = "drowsiness_events.db"
VIDEO_DIR = "drowsy_videos"
VIDEO_CODEC = "mp4v"


def validate_face_detector():
    if not os.path.exists(FACE_PROTO):
        raise FileNotFoundError(f"Не найден файл: {FACE_PROTO}")

    if not os.path.exists(FACE_MODEL):
        raise FileNotFoundError(f"Не найден файл: {FACE_MODEL}")

    model_size = os.path.getsize(FACE_MODEL)

    if model_size < 5_000_000:
        raise ValueError(
            f"Файл {FACE_MODEL} слишком маленький: {model_size:,} bytes. "
            f"Скорее всего, модель повреждена."
        )


def load_face_detector():
    validate_face_detector()

    print("[INIT] loading_face_detector", flush=True)

    net = cv2.dnn.readNetFromCaffe(FACE_PROTO, FACE_MODEL)

    if net.empty():
        raise RuntimeError("OpenCV не смог загрузить Caffe face detector")

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
        raise RuntimeError(f"Face detector не прошёл dummy inference: {e}") from e

    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

    print("[INIT] face_detector_ready", flush=True)

    return net


class RKNNClassifier:
    def __init__(self, model_path, name="RKNNModel", core_mask=1):
        try:
            from rknnlite.api import RKNNLite
        except ImportError:
            print("[ERROR] rknn_lite_runtime_not_installed", flush=True)
            sys.exit(1)

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Не найден RKNN-файл: {model_path}")

        self.name = name
        self.rknn = RKNNLite()

        print(f"[INIT] loading_model name={self.name} path={model_path}", flush=True)

        ret = self.rknn.load_rknn(model_path)

        if ret != 0:
            raise RuntimeError(f"Ошибка загрузки {self.name}, code={ret}")

        print(f"[INIT] init_npu name={self.name}", flush=True)

        ret = self.rknn.init_runtime(core_mask=core_mask)

        if ret != 0:
            raise RuntimeError(f"Ошибка init_runtime для {self.name}, code={ret}")

        print(f"[INIT] model_ready name={self.name}", flush=True)

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

        if debug:
            print(
                f"[DEBUG] model={self.name} "
                f"input_shape={input_data.shape} "
                f"dtype={input_data.dtype} "
                f"min={input_data.min()} "
                f"max={input_data.max()}",
                flush=True
            )

        outputs = self.rknn.inference(inputs=[input_data])

        if outputs is None or len(outputs) == 0:
            raise RuntimeError(f"{self.name}: пустой output от RKNN")

        out_tensor = outputs[0].copy()
        raw_value = float(out_tensor.flatten()[0])

        if raw_value < 0.0 or raw_value > 1.0:
            prob = 1.0 / (1.0 + np.exp(-raw_value))
        else:
            prob = raw_value

        if debug:
            print(
                f"[DEBUG] model={self.name} raw={raw_value:.4f} prob={prob:.4f}",
                flush=True
            )

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
    best_conf = 0.0

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
            best_conf = conf

    return best_face, best_conf


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

            return cls, avg_prob

    def reset(self):
        with self.lock:
            self.history.clear()


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
        start_sec = self.start_idx / self.fps
        end_sec = end_idx / self.fps
        duration_sec = (end_idx - self.start_idx + 1) / self.fps

        if duration_sec >= self.min_sec:
            print(
                f"[DROWSY_EVENT] "
                f"start={start_sec:.2f}s "
                f"end={end_sec:.2f}s "
                f"duration={duration_sec:.2f}s",
                flush=True
            )


class LocalEventDB:
    def __init__(self, db_path=DB_PATH):
        self.db_path = db_path
        self.lock = threading.Lock()
        self._init_db()

    def _connect(self):
        return sqlite3.connect(self.db_path, check_same_thread=False)

    def _init_db(self):
        with self._connect() as conn:
            cur = conn.cursor()

            cur.execute("""
                CREATE TABLE IF NOT EXISTS events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,

                    event_uid TEXT UNIQUE NOT NULL,

                    start_time_local TEXT NOT NULL,
                    confirmed_time_local TEXT,
                    end_time_local TEXT NOT NULL,

                    duration_sec REAL NOT NULL,
                    confirmation_sec REAL NOT NULL,

                    frame_count INTEGER NOT NULL,
                    fps REAL NOT NULL,

                    video_path TEXT NOT NULL,

                    avg_prob REAL,
                    max_prob REAL,
                    min_prob REAL,

                    avg_smooth_prob REAL,
                    max_smooth_prob REAL,
                    min_smooth_prob REAL,

                    avg_vit_prob REAL,
                    max_vit_prob REAL,
                    min_vit_prob REAL,

                    avg_coatnet_prob REAL,
                    max_coatnet_prob REAL,
                    min_coatnet_prob REAL,

                    avg_face_conf REAL,
                    max_face_conf REAL,
                    min_face_conf REAL,

                    created_at TEXT NOT NULL
                )
            """)

            cur.execute("""
                CREATE TABLE IF NOT EXISTS frame_predictions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,

                    event_uid TEXT NOT NULL,

                    local_timestamp TEXT NOT NULL,
                    elapsed_sec REAL NOT NULL,

                    frame_index INTEGER NOT NULL,

                    status TEXT NOT NULL,

                    prob REAL,
                    smooth_prob REAL,

                    vit_prob REAL,
                    coatnet_prob REAL,
                    face_conf REAL,

                    is_drowsy INTEGER NOT NULL,

                    FOREIGN KEY(event_uid) REFERENCES events(event_uid)
                )
            """)

            conn.commit()

    def save_event(self, event_data, frame_predictions):
        with self.lock:
            with self._connect() as conn:
                cur = conn.cursor()

                cur.execute("""
                    INSERT INTO events (
                        event_uid,

                        start_time_local,
                        confirmed_time_local,
                        end_time_local,

                        duration_sec,
                        confirmation_sec,

                        frame_count,
                        fps,

                        video_path,

                        avg_prob,
                        max_prob,
                        min_prob,

                        avg_smooth_prob,
                        max_smooth_prob,
                        min_smooth_prob,

                        avg_vit_prob,
                        max_vit_prob,
                        min_vit_prob,

                        avg_coatnet_prob,
                        max_coatnet_prob,
                        min_coatnet_prob,

                        avg_face_conf,
                        max_face_conf,
                        min_face_conf,

                        created_at
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    event_data["event_uid"],

                    event_data["start_time_local"],
                    event_data["confirmed_time_local"],
                    event_data["end_time_local"],

                    event_data["duration_sec"],
                    event_data["confirmation_sec"],

                    event_data["frame_count"],
                    event_data["fps"],

                    event_data["video_path"],

                    event_data["avg_prob"],
                    event_data["max_prob"],
                    event_data["min_prob"],

                    event_data["avg_smooth_prob"],
                    event_data["max_smooth_prob"],
                    event_data["min_smooth_prob"],

                    event_data["avg_vit_prob"],
                    event_data["max_vit_prob"],
                    event_data["min_vit_prob"],

                    event_data["avg_coatnet_prob"],
                    event_data["max_coatnet_prob"],
                    event_data["min_coatnet_prob"],

                    event_data["avg_face_conf"],
                    event_data["max_face_conf"],
                    event_data["min_face_conf"],

                    event_data["created_at"]
                ))

                cur.executemany("""
                    INSERT INTO frame_predictions (
                        event_uid,
                        local_timestamp,
                        elapsed_sec,

                        frame_index,
                        status,

                        prob,
                        smooth_prob,

                        vit_prob,
                        coatnet_prob,
                        face_conf,

                        is_drowsy
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, [
                    (
                        p["event_uid"],
                        p["local_timestamp"],
                        p["elapsed_sec"],

                        p["frame_index"],
                        p["status"],

                        p["prob"],
                        p["smooth_prob"],

                        p["vit_prob"],
                        p["coatnet_prob"],
                        p["face_conf"],

                        p["is_drowsy"]
                    )
                    for p in frame_predictions
                ])

                conn.commit()


class ContinuousDrowsyEventRecorder:
    def __init__(
        self,
        db,
        fps,
        min_confirm_sec=MIN_EVENT_SEC,
        video_dir=VIDEO_DIR
    ):
        self.db = db
        self.fps = fps if fps and fps > 1 else 30.0
        self.min_confirm_sec = float(min_confirm_sec)
        self.video_dir = video_dir

        os.makedirs(self.video_dir, exist_ok=True)

        self.lock = threading.Lock()

        self.active = False
        self.confirmed = False

        self.writer = None

        self.event_uid = None
        self.video_path = None

        self.start_time_perf = None
        self.confirmed_time_perf = None
        self.end_time_perf = None

        self.start_time_local = None
        self.confirmed_time_local = None
        self.end_time_local = None

        self.frame_count = 0

        self.frame_predictions = []

        self.probs = []
        self.smooth_probs = []
        self.vit_probs = []
        self.coatnet_probs = []
        self.face_confs = []

    def _now_local(self):
        return datetime.now().astimezone().isoformat(timespec="milliseconds")

    def _new_video_path(self):
        timestamp_for_file = datetime.now().astimezone().strftime("%Y%m%d_%H%M%S_%f")

        return os.path.join(
            self.video_dir,
            f"drowsy_{timestamp_for_file}_{self.event_uid[:8]}.mp4"
        )

    def _start_candidate_event(self, frame):
        self.active = True
        self.confirmed = False

        self.event_uid = str(uuid.uuid4())
        self.video_path = self._new_video_path()

        self.start_time_perf = time.perf_counter()
        self.confirmed_time_perf = None
        self.end_time_perf = None

        self.start_time_local = self._now_local()
        self.confirmed_time_local = None
        self.end_time_local = None

        self.frame_count = 0

        self.frame_predictions = []

        self.probs = []
        self.smooth_probs = []
        self.vit_probs = []
        self.coatnet_probs = []
        self.face_confs = []

        h, w = frame.shape[:2]

        fourcc = cv2.VideoWriter_fourcc(*VIDEO_CODEC)

        self.writer = cv2.VideoWriter(
            self.video_path,
            fourcc,
            self.fps,
            (w, h)
        )

        if not self.writer.isOpened():
            raise RuntimeError(f"Не удалось открыть VideoWriter: {self.video_path}")

        print(
            f"[EVENT] candidate_started "
            f"uid={self.event_uid} "
            f"video={self.video_path}",
            flush=True
        )

    def update(
        self,
        frame,
        is_drowsy,
        status,
        prob=None,
        smooth_prob=None,
        vit_prob=None,
        coatnet_prob=None,
        face_conf=None
    ):
        with self.lock:
            if is_drowsy:
                if not self.active:
                    self._start_candidate_event(frame)

                self._write_drowsy_frame(
                    frame=frame,
                    status=status,
                    prob=prob,
                    smooth_prob=smooth_prob,
                    vit_prob=vit_prob,
                    coatnet_prob=coatnet_prob,
                    face_conf=face_conf
                )

                self._check_confirmation()

            else:
                if self.active:
                    if self.confirmed:
                        self._finish_confirmed_event()
                    else:
                        self._discard_candidate_event(
                            reason=f"{status}_before_confirmation"
                        )

    def _write_drowsy_frame(
        self,
        frame,
        status,
        prob=None,
        smooth_prob=None,
        vit_prob=None,
        coatnet_prob=None,
        face_conf=None
    ):
        if self.writer is not None:
            self.writer.write(frame)

        self.frame_count += 1

        now_perf = time.perf_counter()
        elapsed_sec = now_perf - self.start_time_perf

        self.frame_predictions.append({
            "event_uid": self.event_uid,
            "local_timestamp": self._now_local(),
            "elapsed_sec": float(elapsed_sec),

            "frame_index": int(self.frame_count),
            "status": status,

            "prob": float(prob) if prob is not None else None,
            "smooth_prob": float(smooth_prob) if smooth_prob is not None else None,

            "vit_prob": float(vit_prob) if vit_prob is not None else None,
            "coatnet_prob": float(coatnet_prob) if coatnet_prob is not None else None,
            "face_conf": float(face_conf) if face_conf is not None else None,

            "is_drowsy": 1
        })

        if prob is not None:
            self.probs.append(float(prob))

        if smooth_prob is not None:
            self.smooth_probs.append(float(smooth_prob))

        if vit_prob is not None:
            self.vit_probs.append(float(vit_prob))

        if coatnet_prob is not None:
            self.coatnet_probs.append(float(coatnet_prob))

        if face_conf is not None:
            self.face_confs.append(float(face_conf))

    def _check_confirmation(self):
        if not self.active or self.confirmed:
            return

        elapsed_sec = time.perf_counter() - self.start_time_perf

        if elapsed_sec >= self.min_confirm_sec:
            self.confirmed = True
            self.confirmed_time_perf = time.perf_counter()
            self.confirmed_time_local = self._now_local()

            print(
                f"[EVENT] confirmed "
                f"uid={self.event_uid} "
                f"confirmation_sec={elapsed_sec:.2f} "
                f"frames={self.frame_count}",
                flush=True
            )

    def _finish_confirmed_event(self):
        self.end_time_perf = time.perf_counter()
        self.end_time_local = self._now_local()

        if self.writer is not None:
            self.writer.release()
            self.writer = None

        duration_sec = self.end_time_perf - self.start_time_perf

        event_data = {
            "event_uid": self.event_uid,

            "start_time_local": self.start_time_local,
            "confirmed_time_local": self.confirmed_time_local,
            "end_time_local": self.end_time_local,

            "duration_sec": float(duration_sec),
            "confirmation_sec": float(self.min_confirm_sec),

            "frame_count": int(self.frame_count),
            "fps": float(self.fps),

            "video_path": self.video_path,

            "avg_prob": self._safe_avg(self.probs),
            "max_prob": self._safe_max(self.probs),
            "min_prob": self._safe_min(self.probs),

            "avg_smooth_prob": self._safe_avg(self.smooth_probs),
            "max_smooth_prob": self._safe_max(self.smooth_probs),
            "min_smooth_prob": self._safe_min(self.smooth_probs),

            "avg_vit_prob": self._safe_avg(self.vit_probs),
            "max_vit_prob": self._safe_max(self.vit_probs),
            "min_vit_prob": self._safe_min(self.vit_probs),

            "avg_coatnet_prob": self._safe_avg(self.coatnet_probs),
            "max_coatnet_prob": self._safe_max(self.coatnet_probs),
            "min_coatnet_prob": self._safe_min(self.coatnet_probs),

            "avg_face_conf": self._safe_avg(self.face_confs),
            "max_face_conf": self._safe_max(self.face_confs),
            "min_face_conf": self._safe_min(self.face_confs),

            "created_at": self._now_local()
        }

        self.db.save_event(event_data, self.frame_predictions)

        print(
            f"[EVENT] saved "
            f"uid={self.event_uid} "
            f"duration={duration_sec:.2f}s "
            f"frames={self.frame_count} "
            f"video={self.video_path}",
            flush=True
        )

        self._reset()

    def _discard_candidate_event(self, reason):
        if self.writer is not None:
            self.writer.release()
            self.writer = None

        duration_sec = 0.0

        if self.start_time_perf is not None:
            duration_sec = time.perf_counter() - self.start_time_perf

        print(
            f"[EVENT] discarded "
            f"uid={self.event_uid} "
            f"reason={reason} "
            f"duration={duration_sec:.2f}s "
            f"frames={self.frame_count}",
            flush=True
        )

        if self.video_path and os.path.exists(self.video_path):
            try:
                os.remove(self.video_path)
            except OSError as e:
                print(
                    f"[WARN] cannot_remove_candidate_video "
                    f"path={self.video_path} "
                    f"error={e}",
                    flush=True
                )

        self._reset()

    def force_end(self):
        with self.lock:
            if not self.active:
                return

            if self.confirmed:
                self._finish_confirmed_event()
            else:
                self._discard_candidate_event(
                    reason="program_stopped_before_confirmation"
                )

    def _safe_avg(self, values):
        if not values:
            return None

        return float(sum(values) / len(values))

    def _safe_max(self, values):
        if not values:
            return None

        return float(max(values))

    def _safe_min(self, values):
        if not values:
            return None

        return float(min(values))

    def _reset(self):
        self.active = False
        self.confirmed = False

        self.writer = None

        self.event_uid = None
        self.video_path = None

        self.start_time_perf = None
        self.confirmed_time_perf = None
        self.end_time_perf = None

        self.start_time_local = None
        self.confirmed_time_local = None
        self.end_time_local = None

        self.frame_count = 0

        self.frame_predictions = []

        self.probs = []
        self.smooth_probs = []
        self.vit_probs = []
        self.coatnet_probs = []
        self.face_confs = []


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
        tracker,
        event_recorder
    ):
        self.cam = cam
        self.vit = vit_model
        self.coatnet = coatnet_model
        self.face_net = face_net
        self.smoother = smoother
        self.tracker = tracker
        self.event_recorder = event_recorder

        self.stop = False
        self.display_frame = None
        self.error = None
        self.lock = threading.Lock()

        self.prev_state = None

        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def log_state(
        self,
        state,
        prob=None,
        vit_prob=None,
        coat_prob=None,
        face_conf=None
    ):
        if state == self.prev_state:
            return

        if prob is None:
            print(f"[STATE] {state}", flush=True)
        else:
            print(
                f"[STATE] {state} "
                f"prob={prob:.3f} "
                f"vit={vit_prob:.3f} "
                f"coatnet={coat_prob:.3f} "
                f"face_conf={face_conf:.3f}",
                flush=True
            )

        self.prev_state = state

    def _run(self):
        try:
            while not self.stop:
                ret, frame = self.cam.read()

                if not ret:
                    break

                draw = frame.copy()

                status = "NO_FACE"
                color = (255, 255, 255)
                is_drowsy = False

                current_prob = None
                current_smooth_prob = None
                current_vit_prob = None
                current_coat_prob = None
                current_face_conf = None

                box, face_conf = detect_face(self.face_net, frame)

                if box is not None:
                    x1, y1, x2, y2 = box
                    roi = frame[y1:y2, x1:x2]

                    vit_prob = self.vit.predict(roi, debug=DEBUG_MODE)
                    coat_prob = self.coatnet.predict(roi, debug=DEBUG_MODE)

                    prob = vit_prob * 0.8 + coat_prob * 0.2

                    cls, smooth_prob = self.smoother.add(prob)

                    current_prob = prob
                    current_smooth_prob = smooth_prob
                    current_vit_prob = vit_prob
                    current_coat_prob = coat_prob
                    current_face_conf = face_conf

                    if cls == 1:
                        is_drowsy = True
                        status = "DROWSY"
                        color = (0, 0, 255)

                        self.log_state(
                            state="DROWSY",
                            prob=smooth_prob,
                            vit_prob=vit_prob,
                            coat_prob=coat_prob,
                            face_conf=face_conf
                        )
                    else:
                        is_drowsy = False
                        status = "NON_DROWSY"
                        color = (0, 255, 0)

                        self.log_state(
                            state="NON_DROWSY",
                            prob=smooth_prob,
                            vit_prob=vit_prob,
                            coat_prob=coat_prob,
                            face_conf=face_conf
                        )

                    cv2.rectangle(draw, (x1, y1), (x2, y2), color, 2)

                    cv2.putText(
                        draw,
                        f"face={face_conf:.2f}",
                        (x1, max(20, y1 - 45)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        color,
                        1
                    )

                    cv2.putText(
                        draw,
                        f"vit={vit_prob:.2f} coat={coat_prob:.2f}",
                        (x1, max(20, y1 - 28)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        color,
                        1
                    )

                    cv2.putText(
                        draw,
                        f"ens={prob:.2f} smooth={smooth_prob:.2f}",
                        (x1, max(20, y1 - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        color,
                        1
                    )

                else:
                    self.smoother.reset()
                    self.log_state("NO_FACE")

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

                self.event_recorder.update(
                    frame=draw,
                    is_drowsy=is_drowsy,
                    status=status,
                    prob=current_prob,
                    smooth_prob=current_smooth_prob,
                    vit_prob=current_vit_prob,
                    coatnet_prob=current_coat_prob,
                    face_conf=current_face_conf
                )

                with self.lock:
                    self.display_frame = draw

        except Exception as e:
            self.error = e
            print(f"[ERROR] processing_error={e}", flush=True)

    def get_frame(self):
        with self.lock:
            return self.display_frame

    def has_error(self):
        return self.error is not None

    def close(self):
        self.stop = True


def main():
    print("[INIT] starting_vit_coatnet_hybrid", flush=True)

    vit_model = None
    coatnet_model = None
    face_net = None
    cam = None
    proc = None
    tracker = None
    db = None
    event_recorder = None

    try:
        vit_model = RKNNClassifier(
            VIT_RKNN_PATH,
            name="ViT",
            core_mask=1
        )

        coatnet_model = RKNNClassifier(
            COATNET_RKNN_PATH,
            name="CoAtNet",
            core_mask=1
        )

        face_net = load_face_detector()
        cam = CameraThread(CAMERA_INDEX)

    except Exception as e:
        print(f"[ERROR] initialization_failed={e}", flush=True)
        sys.exit(1)

    fps = cam.cap.get(cv2.CAP_PROP_FPS)

    if fps is None or fps <= 1:
        fps = 30.0

    print(f"[INIT] camera_fps={fps:.2f}", flush=True)

    smoother = SmoothPredictor(
        window=5,
        threshold=DROWSY_THRESHOLD
    )

    tracker = EventTracker(
        fps=fps,
        min_sec=MIN_EVENT_SEC
    )

    db = LocalEventDB(DB_PATH)

    event_recorder = ContinuousDrowsyEventRecorder(
        db=db,
        fps=fps,
        min_confirm_sec=MIN_EVENT_SEC,
        video_dir=VIDEO_DIR
    )

    proc = ProcessingThread(
        cam=cam,
        vit_model=vit_model,
        coatnet_model=coatnet_model,
        face_net=face_net,
        smoother=smoother,
        tracker=tracker,
        event_recorder=event_recorder
    )

    print("[INIT] pipeline_started", flush=True)

    try:
        while True:
            if proc.has_error():
                print(f"[ERROR] pipeline_halted={proc.error}", flush=True)
                break

            frame = proc.get_frame()

            if frame is not None:
                cv2.imshow("Drowsiness Detection", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    except KeyboardInterrupt:
        print("[INFO] stopped_by_user", flush=True)

    finally:
        print("[INFO] shutting_down", flush=True)

        if proc:
            proc.close()

        if cam:
            cam.close()

        if tracker:
            tracker.force_end()

        if event_recorder:
            event_recorder.force_end()

        if vit_model:
            vit_model.release()

        if coatnet_model:
            coatnet_model.release()

        cv2.destroyAllWindows()

        print("[INFO] done", flush=True)


if __name__ == "__main__":
    main()