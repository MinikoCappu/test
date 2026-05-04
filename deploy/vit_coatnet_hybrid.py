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
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


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

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DB_PATH = os.path.join(BASE_DIR, "drowsiness_events.db")
VIDEO_DIR = os.path.join(BASE_DIR, "drowsy_videos")
LATEST_FRAME_PATH = os.path.join(BASE_DIR, "latest_frame.jpg")

VIDEO_CODEC = "MJPG"
VIDEO_EXT = ".avi"
VIDEO_FPS = 8.0
SNAPSHOT_INTERVAL_SEC = 0.5

LIVE_STREAM_ENABLED = os.environ.get("LIVE_STREAM_ENABLED", "1") == "1"
LIVE_STREAM_HOST = os.environ.get("LIVE_STREAM_HOST", "0.0.0.0")
LIVE_STREAM_PORT = int(os.environ.get("LIVE_STREAM_PORT", "8080"))
LIVE_STREAM_MAX_FPS = float(os.environ.get("LIVE_STREAM_MAX_FPS", "8.0"))
LIVE_STREAM_JPEG_QUALITY = int(os.environ.get("LIVE_STREAM_JPEG_QUALITY", "80"))

SHOW_WINDOW = os.environ.get("SHOW_WINDOW", "0") == "1"


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

                print(
                    f"[DB] inserted_event "
                    f"uid={event_data['event_uid']} "
                    f"frames={len(frame_predictions)}",
                    flush=True
                )


class ContinuousDrowsyEventRecorder:
    def __init__(
        self,
        db,
        fps,
        min_confirm_sec=MIN_EVENT_SEC,
        video_dir=VIDEO_DIR
    ):
        self.db = db
        self.fps = fps if fps and fps > 1 else VIDEO_FPS
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
            f"drowsy_{timestamp_for_file}_{self.event_uid[:8]}{VIDEO_EXT}"
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

        os.makedirs(self.video_dir, exist_ok=True)

        if frame is None or frame.size == 0:
            print("[ERROR] empty_frame_for_video_writer", flush=True)
            self._reset()
            return False

        h, w = frame.shape[:2]

        if w <= 0 or h <= 0:
            print(f"[ERROR] invalid_frame_size w={w} h={h}", flush=True)
            self._reset()
            return False

        if not os.path.isdir(self.video_dir):
            print(f"[ERROR] video_dir_not_exists path={self.video_dir}", flush=True)
            self._reset()
            return False

        if not os.access(self.video_dir, os.W_OK):
            print(f"[ERROR] video_dir_not_writable path={self.video_dir}", flush=True)
            self._reset()
            return False

        fourcc = cv2.VideoWriter_fourcc(*VIDEO_CODEC)

        print(
            f"[VIDEO] opening_writer "
            f"path={self.video_path} "
            f"codec={VIDEO_CODEC} "
            f"fps={self.fps:.2f} "
            f"size=({w},{h}) "
            f"dir_exists={os.path.isdir(self.video_dir)} "
            f"dir_writable={os.access(self.video_dir, os.W_OK)}",
            flush=True
        )

        self.writer = cv2.VideoWriter(
            self.video_path,
            fourcc,
            self.fps,
            (w, h)
        )

        if not self.writer.isOpened():
            print(
                f"[ERROR] cannot_open_video_writer "
                f"path={self.video_path} "
                f"codec={VIDEO_CODEC} "
                f"fps={self.fps:.2f} "
                f"size=({w},{h})",
                flush=True
            )

            self.writer = None
            self._reset()
            return False

        print(
            f"[EVENT] candidate_started "
            f"uid={self.event_uid} "
            f"video={self.video_path}",
            flush=True
        )

        return True

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
                    started = self._start_candidate_event(frame)

                    if not started:
                        return

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
        video_duration_sec = self.frame_count / self.fps if self.fps > 0 else 0.0

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

        print(
            f"[DB] saving_event "
            f"uid={self.event_uid} "
            f"frames={len(self.frame_predictions)} "
            f"video={self.video_path}",
            flush=True
        )

        self.db.save_event(event_data, self.frame_predictions)

        print(
            f"[EVENT] saved "
            f"uid={self.event_uid} "
            f"event_duration={duration_sec:.2f}s "
            f"video_duration={video_duration_sec:.2f}s "
            f"frames={self.frame_count} "
            f"fps={self.fps:.2f} "
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


class LiveFrameBuffer:
    def __init__(self):
        self.condition = threading.Condition()
        self.jpeg_bytes = None
        self.frame_id = 0
        self.updated_at = None

    def update(self, frame):
        if frame is None or frame.size == 0:
            return

        ok, encoded = cv2.imencode(
            ".jpg",
            frame,
            [int(cv2.IMWRITE_JPEG_QUALITY), LIVE_STREAM_JPEG_QUALITY]
        )

        if not ok:
            return

        with self.condition:
            self.jpeg_bytes = encoded.tobytes()
            self.frame_id += 1
            self.updated_at = time.time()
            self.condition.notify_all()

    def wait_for_frame(self, last_frame_id, timeout=2.0):
        with self.condition:
            if self.frame_id == last_frame_id:
                self.condition.wait(timeout=timeout)

            return self.frame_id, self.jpeg_bytes, self.updated_at

    def snapshot(self):
        with self.condition:
            return self.frame_id, self.jpeg_bytes, self.updated_at


class LiveStreamHandler(BaseHTTPRequestHandler):
    frame_buffer = None
    protocol_version = "HTTP/1.0"

    def do_GET(self):
        if self.path in ("/", "/health"):
            self._handle_health()
            return

        if self.path.startswith("/video"):
            self._handle_video()
            return

        self.send_error(404, "Not found")

    def _handle_health(self):
        frame_id, jpeg_bytes, updated_at = self.frame_buffer.snapshot()
        status = "online" if jpeg_bytes is not None else "waiting_for_frame"
        age = 0.0 if updated_at is None else max(0.0, time.time() - updated_at)
        body = (
            f"status={status}\n"
            f"frame_id={frame_id}\n"
            f"age_sec={age:.3f}\n"
        ).encode("utf-8")

        self.send_response(200)
        self.send_header("Content-Type", "text/plain; charset=utf-8")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _handle_video(self):
        self.send_response(200)
        self.send_header("Age", "0")
        self.send_header("Cache-Control", "no-cache, private")
        self.send_header("Pragma", "no-cache")
        self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
        self.end_headers()

        last_frame_id = 0

        try:
            while True:
                frame_id, jpeg_bytes, _ = self.frame_buffer.wait_for_frame(last_frame_id)

                if jpeg_bytes is None or frame_id == last_frame_id:
                    continue

                last_frame_id = frame_id

                self.wfile.write(b"--frame\r\n")
                self.wfile.write(b"Content-Type: image/jpeg\r\n")
                self.wfile.write(f"Content-Length: {len(jpeg_bytes)}\r\n\r\n".encode("ascii"))
                self.wfile.write(jpeg_bytes)
                self.wfile.write(b"\r\n")
                self.wfile.flush()

        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
            return

    def log_message(self, _format, *args):
        return


class MJPEGLiveServer:
    def __init__(self, host, port, frame_buffer):
        handler_class = type(
            "ConfiguredLiveStreamHandler",
            (LiveStreamHandler,),
            {"frame_buffer": frame_buffer}
        )

        self.server = ThreadingHTTPServer((host, port), handler_class)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.host = host
        self.port = port

    def start(self):
        print(
            f"[LIVE] mjpeg_server_started "
            f"host={self.host} port={self.port} "
            f"max_fps={LIVE_STREAM_MAX_FPS:.2f}",
            flush=True
        )
        self.thread.start()

    def close(self):
        print("[LIVE] mjpeg_server_stopping", flush=True)
        self.server.shutdown()
        self.server.server_close()


class CameraThread:
    def __init__(self, idx=0):
        self.idx = idx
        self.cap = cv2.VideoCapture(idx)

        if not self.cap.isOpened():
            raise RuntimeError(f"Не удалось открыть камеру {idx}")

        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        self.q = queue.Queue(maxsize=QUEUE_MAXSIZE)
        self.stop = False

        self.frame_counter = 0
        self.failed_reads = 0
        self.last_log_time = time.perf_counter()

        print(f"[CAMERA] opened index={idx}", flush=True)

        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def _run(self):
        while not self.stop:
            ret, frame = self.cap.read()

            if not ret or frame is None:
                self.failed_reads += 1

                print(
                    f"[CAMERA] read_failed count={self.failed_reads}",
                    flush=True
                )

                time.sleep(0.1)

                if self.failed_reads % 30 == 0:
                    print("[CAMERA] trying_reopen", flush=True)
                    self._reopen_camera()

                continue

            self.failed_reads = 0
            self.frame_counter += 1

            now = time.perf_counter()

            if now - self.last_log_time >= 5.0:
                print(
                    f"[CAMERA] active "
                    f"frames={self.frame_counter} "
                    f"queue_size={self.q.qsize()}",
                    flush=True
                )
                self.last_log_time = now

            if self.q.full():
                try:
                    self.q.get_nowait()
                except queue.Empty:
                    pass

            self.q.put((ret, frame))

        print("[CAMERA] thread_stopped", flush=True)

    def _reopen_camera(self):
        try:
            self.cap.release()
        except Exception:
            pass

        time.sleep(0.5)

        self.cap = cv2.VideoCapture(self.idx)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        if self.cap.isOpened():
            print("[CAMERA] reopened_successfully", flush=True)
        else:
            print("[CAMERA] reopen_failed", flush=True)

    def read(self):
        return self.q.get()

    def close(self):
        self.stop = True

        if self.thread.is_alive():
            self.thread.join(timeout=2.0)

        self.cap.release()

        print("[CAMERA] released", flush=True)


class ProcessingThread:
    def __init__(
        self,
        cam,
        vit_model,
        coatnet_model,
        face_net,
        smoother,
        tracker,
        event_recorder,
        live_frame_buffer=None
    ):
        self.cam = cam
        self.vit = vit_model
        self.coatnet = coatnet_model
        self.face_net = face_net
        self.smoother = smoother
        self.tracker = tracker
        self.event_recorder = event_recorder
        self.live_frame_buffer = live_frame_buffer

        self.stop = False
        self.display_frame = None
        self.error = None
        self.lock = threading.Lock()

        self.prev_state = None

        self.processed_frames = 0
        self.last_processing_log_time = time.perf_counter()
        self.last_snapshot_time = 0.0
        self.last_live_frame_time = 0.0

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

    def _write_latest_snapshot(self, frame):
        now = time.perf_counter()

        if now - self.last_snapshot_time < SNAPSHOT_INTERVAL_SEC:
            return

        tmp_path = f"{LATEST_FRAME_PATH}.tmp"

        try:
            ok = cv2.imwrite(tmp_path, frame)

            if ok:
                os.replace(tmp_path, LATEST_FRAME_PATH)
                self.last_snapshot_time = now
        except Exception as e:
            print(f"[WARN] cannot_write_latest_frame error={e}", flush=True)

    def _update_live_stream(self, frame):
        if self.live_frame_buffer is None:
            return

        now = time.perf_counter()
        min_interval = 1.0 / LIVE_STREAM_MAX_FPS if LIVE_STREAM_MAX_FPS > 0 else 0.0

        if now - self.last_live_frame_time < min_interval:
            return

        self.live_frame_buffer.update(frame)
        self.last_live_frame_time = now

    def _run(self):
        try:
            while not self.stop:
                ret, frame = self.cam.read()

                if not ret:
                    break

                self.processed_frames += 1

                now = time.perf_counter()

                if now - self.last_processing_log_time >= 5.0:
                    print(
                        f"[PROCESS] active "
                        f"frames={self.processed_frames}",
                        flush=True
                    )
                    self.last_processing_log_time = now

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

                self._write_latest_snapshot(draw)
                self._update_live_stream(draw)

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
    print(f"[INIT] base_dir={BASE_DIR}", flush=True)
    print(f"[INIT] db_path={DB_PATH}", flush=True)
    print(f"[INIT] video_dir={VIDEO_DIR}", flush=True)
    print(f"[INIT] video_codec={VIDEO_CODEC}", flush=True)
    print(f"[INIT] video_ext={VIDEO_EXT}", flush=True)
    print(f"[INIT] video_fps={VIDEO_FPS:.2f}", flush=True)
    print(f"[INIT] live_stream_enabled={LIVE_STREAM_ENABLED}", flush=True)
    print(f"[INIT] live_stream_host={LIVE_STREAM_HOST}", flush=True)
    print(f"[INIT] live_stream_port={LIVE_STREAM_PORT}", flush=True)
    print(f"[INIT] show_window={SHOW_WINDOW}", flush=True)

    vit_model = None
    coatnet_model = None
    face_net = None
    cam = None
    proc = None
    tracker = None
    db = None
    event_recorder = None
    live_frame_buffer = None
    live_stream_server = None

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

    camera_fps = cam.cap.get(cv2.CAP_PROP_FPS)

    if camera_fps is None or camera_fps <= 1:
        camera_fps = 30.0

    record_fps = VIDEO_FPS

    print(f"[INIT] camera_fps={camera_fps:.2f}", flush=True)
    print(f"[INIT] record_fps={record_fps:.2f}", flush=True)

    smoother = SmoothPredictor(
        window=5,
        threshold=DROWSY_THRESHOLD
    )

    tracker = EventTracker(
        fps=record_fps,
        min_sec=MIN_EVENT_SEC
    )

    db = LocalEventDB(DB_PATH)

    event_recorder = ContinuousDrowsyEventRecorder(
        db=db,
        fps=record_fps,
        min_confirm_sec=MIN_EVENT_SEC,
        video_dir=VIDEO_DIR
    )

    if LIVE_STREAM_ENABLED:
        try:
            live_frame_buffer = LiveFrameBuffer()
            live_stream_server = MJPEGLiveServer(
                LIVE_STREAM_HOST,
                LIVE_STREAM_PORT,
                live_frame_buffer
            )
            live_stream_server.start()
        except Exception as e:
            print(f"[WARN] mjpeg_server_not_started error={e}", flush=True)

    proc = ProcessingThread(
        cam=cam,
        vit_model=vit_model,
        coatnet_model=coatnet_model,
        face_net=face_net,
        smoother=smoother,
        tracker=tracker,
        event_recorder=event_recorder,
        live_frame_buffer=live_frame_buffer
    )

    print("[INIT] pipeline_started", flush=True)

    try:
        while True:
            if proc.has_error():
                print(f"[ERROR] pipeline_halted={proc.error}", flush=True)
                break

            frame = proc.get_frame()

            if SHOW_WINDOW:
                if frame is not None:
                    cv2.imshow("Drowsiness Detection", frame)

                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
            else:
                time.sleep(0.03)

    except KeyboardInterrupt:
        print("[INFO] stopped_by_user", flush=True)

    finally:
        print("[INFO] shutting_down", flush=True)

        if proc:
            proc.close()

        if live_stream_server:
            live_stream_server.close()

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

        if SHOW_WINDOW:
            cv2.destroyAllWindows()

        print("[INFO] done", flush=True)


if __name__ == "__main__":
    main()
