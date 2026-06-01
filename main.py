
"""
main.py — Attention Engine (Backend)
======================================
This module is the SINGLE source of truth for all ML / CV logic.
It is imported by Front_1.py.  It can also be run standalone for a
raw OpenCV debug window (python main.py).

Responsibilities:
  • Load gaze_model.keras + pose_model.keras (trained by Test_1.py)
  • Open webcam via OpenCV
  • Run MediaPipe FaceMesh every frame
  • Predict gaze + pose (Focused / Distracted)
  • Run DeepFace emotion every DEEPFACE_EVERY_N frames
  • Estimate WHERE on screen the user is looking  (gaze_nx, gaze_ny)
  • Expose AttentionEngine class for Front_1.py to import

Architecture:
  Test_1.py  →  trains models  →  saves .keras / .pkl files
  main.py    →  loads models, runs inference, exposes AttentionEngine
  Front_1.py →  imports AttentionEngine, runs it in a QThread, draws UI
"""

import ssl, certifi
ssl._create_default_https_context = lambda: ssl.create_default_context(
    cafile=certifi.where()
)

import cv2
import math
import time
import threading
import numpy as np
import tensorflow as tf
import joblib
import mediapipe as mp
from deepface import DeepFace

# ══════════════════════════════════════════════════════════════════════════════
#  Constants  (shared with Front_1.py via import)
# ══════════════════════════════════════════════════════════════════════════════
FOCAL_LENGTH       = 600
KNOWN_FACE_WIDTH   = 14.0
MAX_DISTANCE_CM    = 90
DEEPFACE_EVERY_N   = 15        # run DeepFace every N frames
WARNING_INTERVAL_S = 15        # distraction warning interval in seconds
GAZE_FEATURES      = 43        # fixed feature width expected by gaze_model

GAZE_MODEL_PATH  = "gaze_model.keras"
GAZE_SCALER_PATH = "gaze_scaler.pkl"
GAZE_FEAT_PATH   = "gaze_feature_cols.pkl"
POSE_MODEL_PATH  = "pose_model.keras"
POSE_SCALER_PATH = "pose_scaler.pkl"

# MediaPipe landmark indices
LEFT_IRIS         = [474, 475, 476, 477]
RIGHT_IRIS        = [469, 470, 471, 472]
LEFT_EYE_CORNERS  = [33,  133]
RIGHT_EYE_CORNERS = [362, 263]
LEFT_EYE_TOP      = 159
LEFT_EYE_BOT      = 145
RIGHT_EYE_TOP     = 386
RIGHT_EYE_BOT     = 374


# ══════════════════════════════════════════════════════════════════════════════
#  Pure helper functions  (stateless — safe to call from any thread)
# ══════════════════════════════════════════════════════════════════════════════

def estimate_distance(face_width_px: float) -> float:
    """Pinhole camera distance estimate in cm."""
    if face_width_px == 0:
        return float("inf")
    return (KNOWN_FACE_WIDTH * FOCAL_LENGTH) / face_width_px


def get_gaze_vector(landmarks, w: int, h: int) -> np.ndarray:
    """
    Returns a 4-element raw gaze vector [lx, ly, rx, ry].
    Each component is the iris centre offset from eye corner,
    normalised by eye width, then centred at 0.
    """
    def iris_center(ids):
        pts = np.array([[landmarks[i].x * w, landmarks[i].y * h] for i in ids])
        return pts.mean(axis=0)

    def eye_width(ids):
        l = np.array([landmarks[ids[0]].x * w, landmarks[ids[0]].y * h])
        r = np.array([landmarks[ids[1]].x * w, landmarks[ids[1]].y * h])
        return np.linalg.norm(r - l) + 1e-6

    l_iris   = iris_center(LEFT_IRIS)
    r_iris   = iris_center(RIGHT_IRIS)
    l_corner = np.array([landmarks[LEFT_EYE_CORNERS[0]].x * w,
                         landmarks[LEFT_EYE_CORNERS[0]].y * h])
    r_corner = np.array([landmarks[RIGHT_EYE_CORNERS[0]].x * w,
                         landmarks[RIGHT_EYE_CORNERS[0]].y * h])

    l_gaze = (l_iris - l_corner) / eye_width(LEFT_EYE_CORNERS) - 0.5
    r_gaze = (r_iris - r_corner) / eye_width(RIGHT_EYE_CORNERS) - 0.5

    return np.array([l_gaze[0], l_gaze[1],
                     r_gaze[0], r_gaze[1]], dtype=np.float32)


def get_head_pose_vector(landmarks, w: int, h: int) -> np.ndarray:
    """
    Returns [yaw, pitch, roll, magnitude] from facial landmarks.
    Nose tip + chin + eye corners used as reference geometry.
    """
    def lm(idx):
        return np.array([landmarks[idx].x * w, landmarks[idx].y * h])

    nose  = lm(1); chin = lm(152)
    l_eye = lm(33); r_eye = lm(263)

    eye_mid  = (l_eye + r_eye) / 2
    eye_span = np.linalg.norm(r_eye - l_eye) + 1e-6
    face_h   = np.linalg.norm(chin - eye_mid) + 1e-6

    yaw   = (nose[0] - eye_mid[0]) / eye_span
    pitch = (nose[1] - eye_mid[1]) / face_h
    roll  = np.arctan2((r_eye - l_eye)[1], (r_eye - l_eye)[0])
    mag   = math.sqrt(yaw**2 + pitch**2 + roll**2)

    return np.array([yaw, pitch, roll, mag], dtype=np.float32)


def estimate_gaze_screen_point(landmarks, w: int, h: int):
    """
    Returns (gaze_x_norm, gaze_y_norm) both in [0.0, 1.0]:
      • 0, 0  →  top-left of screen
      • 1, 1  →  bottom-right of screen

    Uses iris centre offsets (horizontal + vertical), averaged across
    both eyes, then blended 70/30 with head yaw/pitch direction to
    compensate for head rotation.
    """
    def pt(idx):
        return np.array([landmarks[idx].x * w, landmarks[idx].y * h])

    # Iris centres
    l_iris = np.mean([pt(i) for i in LEFT_IRIS],  axis=0)
    r_iris = np.mean([pt(i) for i in RIGHT_IRIS], axis=0)

    # Eye corner / eyelid spans
    l_inner = pt(LEFT_EYE_CORNERS[0]);  l_outer = pt(LEFT_EYE_CORNERS[1])
    r_inner = pt(RIGHT_EYE_CORNERS[0]); r_outer = pt(RIGHT_EYE_CORNERS[1])
    l_top   = pt(LEFT_EYE_TOP);         l_bot   = pt(LEFT_EYE_BOT)
    r_top   = pt(RIGHT_EYE_TOP);        r_bot   = pt(RIGHT_EYE_BOT)

    l_ew = np.linalg.norm(l_outer - l_inner) + 1e-6
    r_ew = np.linalg.norm(r_outer - r_inner) + 1e-6
    l_eh = np.linalg.norm(l_bot   - l_top)   + 1e-6
    r_eh = np.linalg.norm(r_bot   - r_top)   + 1e-6

    # Normalised iris offset inside eye  (centred at 0)
    lx = (l_iris[0] - l_inner[0]) / l_ew - 0.5
    ly = (l_iris[1] - l_top[1])   / l_eh - 0.5
    rx = (r_iris[0] - r_inner[0]) / r_ew - 0.5
    ry = (r_iris[1] - r_top[1])   / r_eh - 0.5

    gx = (lx + rx) / 2.0   # horizontal: -0.5 (left) … +0.5 (right)
    gy = (ly + ry) / 2.0   # vertical:   -0.5 (up)   … +0.5 (down)

    # Head yaw/pitch blend (30 % weight)
    nose   = pt(1); le = pt(33); re = pt(263); chin = pt(152)
    em     = (le + re) / 2
    es     = np.linalg.norm(re - le) + 1e-6
    fh     = np.linalg.norm(chin - em) + 1e-6
    yaw    = (nose[0] - em[0]) / es
    pitch  = (nose[1] - em[1]) / fh

    sx = gx * 0.70 + yaw   * 0.30
    sy = gy * 0.70 + pitch * 0.30

    # Map to [0, 1]
    nx = max(0.0, min(1.0, sx + 0.5))
    ny = max(0.0, min(1.0, sy + 0.5))
    return float(nx), float(ny)


def deepface_attention(frame: np.ndarray, box: tuple) -> str:
    """
    Run DeepFace emotion analysis on the face crop.
    Returns "Focused" (neutral/happy) or "Distracted" or "Unknown".
    Called only every DEEPFACE_EVERY_N frames to keep framerate healthy.
    """
    x, y, bw, bh = box
    face = frame[y:y+bh, x:x+bw]
    try:
        result  = DeepFace.analyze(
            face,
            actions=['emotion'],
            enforce_detection=False,
            detector_backend='opencv'
        )
        emotion = result[0]['dominant_emotion']
        return "Focused" if emotion in ["neutral", "happy"] else "Distracted"
    except Exception:
        return "Unknown"


def draw_ui(frame, box, dist, gaze_label, gaze_conf,
            pose_label, pose_conf, df_label,
            gaze_nx: float = 0.5, gaze_ny: float = 0.5):
    """
    Draw bounding box, status overlay and gaze point onto the OpenCV frame.
    Used by the standalone __main__ loop AND by InferenceWorker for
    the annotated frame sent to Front_1.py.
    """
    x, y, bw, bh = box

    model_focused = (gaze_label == "Focused") and (pose_label == "Focused")
    final_focused = model_focused and (df_label != "Distracted")
    color  = (34, 197, 94)  if final_focused else (239, 68, 68)
    status = "FOCUSED"      if final_focused else "DISTRACTED"

    cv2.rectangle(frame, (x, y), (x+bw, y+bh), color, 2)
    cv2.putText(frame, status,
                (x, y-50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
    cv2.putText(frame, f"Dist: {int(dist)} cm",
                (x, y-30), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
    cv2.putText(frame, f"Gaze: {gaze_label} ({gaze_conf:.2f})",
                (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
    cv2.putText(frame, f"Pose: {pose_label} ({pose_conf:.2f})",
                (x, y+bh+20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 150, 0), 1)
    cv2.putText(frame, f"Face: {df_label}",
                (x, y+bh+40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 200, 0), 1)

    # Gaze point projected onto frame
    h_f, w_f = frame.shape[:2]
    gx_px = int(gaze_nx * w_f); gy_px = int(gaze_ny * h_f)
    cv2.circle(frame, (gx_px, gy_px), 9, (34, 197, 94), 2)
    cv2.circle(frame, (gx_px, gy_px), 2, (34, 197, 94), -1)


# ══════════════════════════════════════════════════════════════════════════════
#  AttentionEngine  —  the class Front_1.py imports and runs in a QThread
# ══════════════════════════════════════════════════════════════════════════════
class AttentionEngine:
    """
    Encapsulates model loading + one-frame inference.

    Usage (from Front_1.py InferenceWorker):
        engine = AttentionEngine()
        engine.load()                        # load models once
        for frame in camera:
            result = engine.process(frame)   # returns a state dict
    """

    def __init__(self):
        self._gaze_model  = None
        self._gaze_scaler = None
        self._pose_model  = None
        self._pose_scaler = None
        self._face_mesh   = None
        self._models_ok   = False
        self._df_label    = "Unknown"   # cached between DeepFace calls
        self._frame_n     = 0

        # Public state snapshot (updated by process())
        self.state = {
            "gaze_label" : "—",
            "gaze_conf"  : 0.0,
            "pose_label" : "—",
            "pose_conf"  : 0.0,
            "df_label"   : "Unknown",
            "df_emotion" : "unknown",
            "focused"    : False,
            "dist"       : 0.0,
            "frame"      : 0,
            "yaw"        : 0.0,
            "pitch"      : 0.0,
            "roll"       : 0.0,
            "gaze_nx"    : 0.5,
            "gaze_ny"    : 0.5,
        }

    # ── model loading ────────────────────────────────────────────────────────
    def load(self, log_fn=None):
        """
        Load all model artefacts produced by Test_1.py.
        log_fn  is an optional callable(str) for status messages.
        Returns True on success, False if files are missing (demo mode).
        """
        def log(msg):
            print(f"[AttentionEngine] {msg}")
            if log_fn: log_fn(msg)

        try:
            log("Loading gaze model…")
            self._gaze_model  = tf.keras.models.load_model(GAZE_MODEL_PATH)
            self._gaze_scaler = joblib.load(GAZE_SCALER_PATH)

            log("Loading pose model…")
            self._pose_model  = tf.keras.models.load_model(POSE_MODEL_PATH)
            self._pose_scaler = joblib.load(POSE_SCALER_PATH)

            try:
                feat_cols = joblib.load(GAZE_FEAT_PATH)
                log(f"Gaze model expects {len(feat_cols)} features")
            except FileNotFoundError:
                log("gaze_feature_cols.pkl not found — using padded 43-feature vector")

            self._exp_gaze = self._gaze_model.input_shape[1]
            self._exp_pose = self._pose_model.input_shape[1]
            self._models_ok = True
            log("All models loaded ✓")

        except Exception as exc:
            self._models_ok = False
            log(f"[DEMO MODE] Models not found: {exc}")

        self._face_mesh = mp.solutions.face_mesh.FaceMesh(
            max_num_faces=1, refine_landmarks=True
        )

    # ── per-frame inference ──────────────────────────────────────────────────
    def process(self, frame: np.ndarray) -> dict:
        """
        Process one BGR frame.

        Mutates and returns self.state — a dict with all inference results.
        Also draws overlays directly onto the frame (in-place).
        """
        self._frame_n += 1
        h, w = frame.shape[:2]
        rgb  = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res  = self._face_mesh.process(rgb)

        state = dict(self.state)
        state["frame"] = self._frame_n

        if not res.multi_face_landmarks:
            self.state = state
            return state

        lms = res.multi_face_landmarks[0].landmark

        # ── bounding box ──
        xs = [l.x for l in lms]; ys = [l.y for l in lms]
        x0 = int(min(xs)*w); x1 = int(max(xs)*w)
        y0 = int(min(ys)*h); y1 = int(max(ys)*h)
        fw = x1 - x0; fh = y1 - y0

        dist = estimate_distance(fw)
        state["dist"] = dist

        # ── head pose (always computed — needed for gaze screen point) ──
        pv = get_head_pose_vector(lms, w, h)
        state["yaw"]   = float(pv[0])
        state["pitch"] = float(pv[1])
        state["roll"]  = float(pv[2])

        # ── gaze screen position (always computed) ──
        gaze_nx, gaze_ny = estimate_gaze_screen_point(lms, w, h)
        state["gaze_nx"] = gaze_nx
        state["gaze_ny"] = gaze_ny

        if dist <= MAX_DISTANCE_CM:
            # ── gaze model ──
            if self._models_ok:
                gl, gc = self._predict_gaze(get_gaze_vector(lms, w, h))
                pl, pc = self._predict_pose(pv)
            else:
                gl, gc, pl, pc = self._demo_predictions()

            # ── DeepFace (throttled) ──
            if self._frame_n % DEEPFACE_EVERY_N == 0:
                self._df_label = deepface_attention(frame, (x0, y0, fw, fh))
                state["df_label"]   = self._df_label
            else:
                state["df_label"] = self._df_label

            # ── final decision: AND logic ──
            focused = (gl == "Focused") and (pl == "Focused") and \
                      (self._df_label != "Distracted")

            state.update(
                gaze_label=gl, gaze_conf=gc,
                pose_label=pl, pose_conf=pc,
                focused=focused,
            )

            # ── annotate frame for camera preview ──
            draw_ui(frame, (x0, y0, fw, fh),
                    dist, gl, gc, pl, pc,
                    self._df_label, gaze_nx, gaze_ny)

        self.state = state
        return state

    # ── private helpers ──────────────────────────────────────────────────────
    def _predict_gaze(self, gaze_vec: np.ndarray):
        """Pad to GAZE_FEATURES (43) and run gaze model."""
        if len(gaze_vec) < GAZE_FEATURES:
            gaze_vec = np.pad(gaze_vec, (0, GAZE_FEATURES - len(gaze_vec)))
        else:
            gaze_vec = gaze_vec[:GAZE_FEATURES]

        pred = self._gaze_model.predict(
            self._gaze_scaler.transform(gaze_vec.reshape(1, -1)),
            verbose=0
        )[0][0]
        return ("Distracted", float(pred)) if pred > 0.5 else ("Focused", float(pred))

    def _predict_pose(self, pose_vec: np.ndarray):
        """Pad / trim to expected pose features and run pose model."""
        exp = self._exp_pose
        if len(pose_vec) < exp:
            pose_vec = np.pad(pose_vec, (0, exp - len(pose_vec)))
        else:
            pose_vec = pose_vec[:exp]

        pred = self._pose_model.predict(
            self._pose_scaler.transform(pose_vec.reshape(1, -1)),
            verbose=0
        )[0][0]
        return ("Distracted", float(pred)) if pred > 0.5 else ("Focused", float(pred))

    def _demo_predictions(self):
        """Smooth sinusoidal demo predictions when models are missing."""
        t  = self._frame_n / 30.0
        gc = max(0.0, min(1.0, 0.68 + 0.28 * math.sin(t * 0.32)))
        pc = max(0.0, min(1.0, 0.74 + 0.22 * math.cos(t * 0.26)))
        gl = "Distracted" if gc > 0.82 else "Focused"
        pl = "Distracted" if pc > 0.82 else "Focused"
        return gl, gc, pl, pc


# ══════════════════════════════════════════════════════════════════════════════
#  Standalone debug mode  —  raw OpenCV window, no Qt needed
#  Run:  python main.py
# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("[INFO] Running main.py in standalone debug mode")
    print("[INFO] Press  ESC  to quit")

    engine = AttentionEngine()
    engine.load()

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Cannot open camera")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        state = engine.process(frame)

        h, w = frame.shape[:2]
        cv2.putText(frame, f"Range: <= {MAX_DISTANCE_CM} cm",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
        cv2.putText(frame, f"Frame: {state['frame']}",
                    (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)
        cv2.putText(frame,
                    f"Gaze screen: ({state['gaze_nx']:.2f}, {state['gaze_ny']:.2f})",
                    (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 255, 100), 1)

        cv2.imshow("FocusSense — Debug", frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()


# """
# main.py — Attention Engine (Backend)
# ======================================
# This module is the SINGLE source of truth for all ML / CV logic.
# It is imported by Front_1.py.  It can also be run standalone for a
# raw OpenCV debug window (python main.py).

# Responsibilities:
#   • Load gaze_model.keras + pose_model.keras (trained by Test_1.py)
#   • Open webcam via OpenCV
#   • Run MediaPipe FaceMesh every frame
#   • Predict gaze + pose (Focused / Distracted)
#   • Run DeepFace emotion every DEEPFACE_EVERY_N frames
#   • Estimate WHERE on screen the user is looking  (gaze_nx, gaze_ny)
#   • Expose AttentionEngine class for Front_1.py to import

# Architecture:
#   Test_1.py  →  trains models  →  saves .keras / .pkl files
#   main.py    →  loads models, runs inference, exposes AttentionEngine
#   Front_1.py →  imports AttentionEngine, runs it in a QThread, draws UI
# """

# import ssl, certifi
# ssl._create_default_https_context = lambda: ssl.create_default_context(
#     cafile=certifi.where()
# )

# import cv2
# import math
# import time
# import threading
# import numpy as np
# import tensorflow as tf
# import joblib
# import mediapipe as mp
# from deepface import DeepFace

# # ══════════════════════════════════════════════════════════════════════════════
# #  Constants  (shared with Front_1.py via import)
# # ══════════════════════════════════════════════════════════════════════════════
# FOCAL_LENGTH       = 600
# KNOWN_FACE_WIDTH   = 14.0
# MAX_DISTANCE_CM    = 90
# DEEPFACE_EVERY_N   = 15        # run DeepFace every N frames
# WARNING_INTERVAL_S = 15        # distraction warning interval in seconds
# GAZE_FEATURES      = 43        # fixed feature width expected by gaze_model

# GAZE_MODEL_PATH  = "gaze_model.keras"
# GAZE_SCALER_PATH = "gaze_scaler.pkl"
# GAZE_FEAT_PATH   = "gaze_feature_cols.pkl"
# POSE_MODEL_PATH  = "pose_model.keras"
# POSE_SCALER_PATH = "pose_scaler.pkl"

# # MediaPipe landmark indices
# LEFT_IRIS         = [474, 475, 476, 477]
# RIGHT_IRIS        = [469, 470, 471, 472]
# LEFT_EYE_CORNERS  = [33,  133]
# RIGHT_EYE_CORNERS = [362, 263]
# LEFT_EYE_TOP      = 159
# LEFT_EYE_BOT      = 145
# RIGHT_EYE_TOP     = 386
# RIGHT_EYE_BOT     = 374


# # ══════════════════════════════════════════════════════════════════════════════
# #  Pure helper functions  (stateless — safe to call from any thread)
# # ══════════════════════════════════════════════════════════════════════════════

# def estimate_distance(face_width_px: float) -> float:
#     """Pinhole camera distance estimate in cm."""
#     if face_width_px == 0:
#         return float("inf")
#     return (KNOWN_FACE_WIDTH * FOCAL_LENGTH) / face_width_px


# def get_gaze_vector(landmarks, w: int, h: int) -> np.ndarray:
#     """
#     Returns a 4-element raw gaze vector [lx, ly, rx, ry].
#     Each component is the iris centre offset from eye corner,
#     normalised by eye width, then centred at 0.
#     """
#     def iris_center(ids):
#         pts = np.array([[landmarks[i].x * w, landmarks[i].y * h] for i in ids])
#         return pts.mean(axis=0)

#     def eye_width(ids):
#         l = np.array([landmarks[ids[0]].x * w, landmarks[ids[0]].y * h])
#         r = np.array([landmarks[ids[1]].x * w, landmarks[ids[1]].y * h])
#         return np.linalg.norm(r - l) + 1e-6

#     l_iris   = iris_center(LEFT_IRIS)
#     r_iris   = iris_center(RIGHT_IRIS)
#     l_corner = np.array([landmarks[LEFT_EYE_CORNERS[0]].x * w,
#                          landmarks[LEFT_EYE_CORNERS[0]].y * h])
#     r_corner = np.array([landmarks[RIGHT_EYE_CORNERS[0]].x * w,
#                          landmarks[RIGHT_EYE_CORNERS[0]].y * h])

#     l_gaze = (l_iris - l_corner) / eye_width(LEFT_EYE_CORNERS) - 0.5
#     r_gaze = (r_iris - r_corner) / eye_width(RIGHT_EYE_CORNERS) - 0.5

#     return np.array([l_gaze[0], l_gaze[1],
#                      r_gaze[0], r_gaze[1]], dtype=np.float32)


# def get_head_pose_vector(landmarks, w: int, h: int) -> np.ndarray:
#     """
#     Returns [yaw, pitch, roll, magnitude] from facial landmarks.
#     Nose tip + chin + eye corners used as reference geometry.
#     """
#     def lm(idx):
#         return np.array([landmarks[idx].x * w, landmarks[idx].y * h])

#     nose  = lm(1); chin = lm(152)
#     l_eye = lm(33); r_eye = lm(263)

#     eye_mid  = (l_eye + r_eye) / 2
#     eye_span = np.linalg.norm(r_eye - l_eye) + 1e-6
#     face_h   = np.linalg.norm(chin - eye_mid) + 1e-6

#     yaw   = (nose[0] - eye_mid[0]) / eye_span
#     pitch = (nose[1] - eye_mid[1]) / face_h
#     roll  = np.arctan2((r_eye - l_eye)[1], (r_eye - l_eye)[0])
#     mag   = math.sqrt(yaw**2 + pitch**2 + roll**2)

#     return np.array([yaw, pitch, roll, mag], dtype=np.float32)


# def estimate_gaze_screen_point(landmarks, w: int, h: int):
#     """
#     Returns (gaze_x_norm, gaze_y_norm) both in [0.0, 1.0]:
#       • 0, 0  →  top-left of screen
#       • 1, 1  →  bottom-right of screen

#     Uses iris centre offsets (horizontal + vertical), averaged across
#     both eyes, then blended 70/30 with head yaw/pitch direction to
#     compensate for head rotation.
#     """
#     def pt(idx):
#         return np.array([landmarks[idx].x * w, landmarks[idx].y * h])

#     # Iris centres
#     l_iris = np.mean([pt(i) for i in LEFT_IRIS],  axis=0)
#     r_iris = np.mean([pt(i) for i in RIGHT_IRIS], axis=0)

#     # Eye corner / eyelid spans
#     l_inner = pt(LEFT_EYE_CORNERS[0]);  l_outer = pt(LEFT_EYE_CORNERS[1])
#     r_inner = pt(RIGHT_EYE_CORNERS[0]); r_outer = pt(RIGHT_EYE_CORNERS[1])
#     l_top   = pt(LEFT_EYE_TOP);         l_bot   = pt(LEFT_EYE_BOT)
#     r_top   = pt(RIGHT_EYE_TOP);        r_bot   = pt(RIGHT_EYE_BOT)

#     l_ew = np.linalg.norm(l_outer - l_inner) + 1e-6
#     r_ew = np.linalg.norm(r_outer - r_inner) + 1e-6
#     l_eh = np.linalg.norm(l_bot   - l_top)   + 1e-6
#     r_eh = np.linalg.norm(r_bot   - r_top)   + 1e-6

#     # Normalised iris offset inside eye  (centred at 0)
#     lx = (l_iris[0] - l_inner[0]) / l_ew - 0.5
#     ly = (l_iris[1] - l_top[1])   / l_eh - 0.5
#     rx = (r_iris[0] - r_inner[0]) / r_ew - 0.5
#     ry = (r_iris[1] - r_top[1])   / r_eh - 0.5

#     gx = (lx + rx) / 2.0   # horizontal: -0.5 (left) … +0.5 (right)
#     gy = (ly + ry) / 2.0   # vertical:   -0.5 (up)   … +0.5 (down)

#     # Head yaw/pitch blend (30 % weight)
#     nose   = pt(1); le = pt(33); re = pt(263); chin = pt(152)
#     em     = (le + re) / 2
#     es     = np.linalg.norm(re - le) + 1e-6
#     fh     = np.linalg.norm(chin - em) + 1e-6
#     yaw    = (nose[0] - em[0]) / es
#     pitch  = (nose[1] - em[1]) / fh

#     sx = gx * 0.70 + yaw   * 0.30
#     sy = gy * 0.70 + pitch * 0.30

#     # Map to [0, 1]
#     nx = max(0.0, min(1.0, sx + 0.5))
#     ny = max(0.0, min(1.0, sy + 0.5))
#     return float(nx), float(ny)


# def deepface_attention(frame: np.ndarray, box: tuple) -> str:
#     """
#     Run DeepFace emotion analysis on the face crop.
#     Returns "Focused" (neutral/happy) or "Distracted" or "Unknown".
#     Called only every DEEPFACE_EVERY_N frames to keep framerate healthy.
#     """
#     x, y, bw, bh = box
#     face = frame[y:y+bh, x:x+bw]
#     try:
#         result  = DeepFace.analyze(
#             face,
#             actions=['emotion'],
#             enforce_detection=False,
#             detector_backend='opencv'
#         )
#         emotion = result[0]['dominant_emotion']
#         return "Focused" if emotion in ["neutral", "happy"] else "Distracted"
#     except Exception:
#         return "Unknown"


# def draw_ui(frame, box, dist, gaze_label, gaze_conf,
#             pose_label, pose_conf, df_label,
#             gaze_nx: float = 0.5, gaze_ny: float = 0.5):
#     """
#     Draw bounding box, status overlay and gaze point onto the OpenCV frame.
#     Used by the standalone __main__ loop AND by InferenceWorker for
#     the annotated frame sent to Front_1.py.
#     """
#     x, y, bw, bh = box

#     model_focused = (gaze_label == "Focused") and (pose_label == "Focused")
#     final_focused = model_focused and (df_label != "Distracted")
#     color  = (34, 197, 94)  if final_focused else (239, 68, 68)
#     status = "FOCUSED"      if final_focused else "DISTRACTED"

#     cv2.rectangle(frame, (x, y), (x+bw, y+bh), color, 2)
#     cv2.putText(frame, status,
#                 (x, y-50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
#     cv2.putText(frame, f"Dist: {int(dist)} cm",
#                 (x, y-30), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
#     cv2.putText(frame, f"Gaze: {gaze_label} ({gaze_conf:.2f})",
#                 (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
#     cv2.putText(frame, f"Pose: {pose_label} ({pose_conf:.2f})",
#                 (x, y+bh+20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 150, 0), 1)
#     cv2.putText(frame, f"Face: {df_label}",
#                 (x, y+bh+40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 200, 0), 1)

#     # Gaze point projected onto frame
#     h_f, w_f = frame.shape[:2]
#     gx_px = int(gaze_nx * w_f); gy_px = int(gaze_ny * h_f)
#     cv2.circle(frame, (gx_px, gy_px), 9, (34, 197, 94), 2)
#     cv2.circle(frame, (gx_px, gy_px), 2, (34, 197, 94), -1)


# # ══════════════════════════════════════════════════════════════════════════════
# #  AttentionEngine  —  the class Front_1.py imports and runs in a QThread
# # ══════════════════════════════════════════════════════════════════════════════
# class AttentionEngine:
#     """
#     Encapsulates model loading + one-frame inference.

#     Usage (from Front_1.py InferenceWorker):
#         engine = AttentionEngine()
#         engine.load()                        # load models once
#         for frame in camera:
#             result = engine.process(frame)   # returns a state dict
#     """

#     def __init__(self):
#         self._gaze_model  = None
#         self._gaze_scaler = None
#         self._pose_model  = None
#         self._pose_scaler = None
#         self._face_mesh   = None
#         self._models_ok   = False
#         self._df_label    = "Unknown"   # cached between DeepFace calls
#         self._frame_n     = 0

#         # Public state snapshot (updated by process())
#         self.state = {
#             "gaze_label" : "—",
#             "gaze_conf"  : 0.0,
#             "pose_label" : "—",
#             "pose_conf"  : 0.0,
#             "df_label"   : "Unknown",
#             "df_emotion" : "unknown",
#             "focused"    : False,
#             "dist"       : 0.0,
#             "frame"      : 0,
#             "yaw"        : 0.0,
#             "pitch"      : 0.0,
#             "roll"       : 0.0,
#             "gaze_nx"    : 0.5,
#             "gaze_ny"    : 0.5,
#         }

#     # ── model loading ────────────────────────────────────────────────────────
#     def load(self, log_fn=None):
#         """
#         Load all model artefacts produced by Test_1.py.
#         log_fn  is an optional callable(str) for status messages.
#         Returns True on success, False if files are missing (demo mode).
#         """
#         def log(msg):
#             print(f"[AttentionEngine] {msg}")
#             if log_fn: log_fn(msg)

#         try:
#             log("Loading gaze model…")
#             self._gaze_model  = tf.keras.models.load_model(GAZE_MODEL_PATH)
#             self._gaze_scaler = joblib.load(GAZE_SCALER_PATH)

#             log("Loading pose model…")
#             self._pose_model  = tf.keras.models.load_model(POSE_MODEL_PATH)
#             self._pose_scaler = joblib.load(POSE_SCALER_PATH)

#             try:
#                 feat_cols = joblib.load(GAZE_FEAT_PATH)
#                 log(f"Gaze model expects {len(feat_cols)} features")
#             except FileNotFoundError:
#                 log("gaze_feature_cols.pkl not found — using padded 43-feature vector")

#             self._exp_gaze = self._gaze_model.input_shape[1]
#             self._exp_pose = self._pose_model.input_shape[1]
#             self._models_ok = True
#             log("All models loaded ✓")

#         except Exception as exc:
#             self._models_ok = False
#             log(f"[DEMO MODE] Models not found: {exc}")

#         self._face_mesh = mp.solutions.face_mesh.FaceMesh(
#             max_num_faces=1, refine_landmarks=True
#         )

#     # ── per-frame inference ──────────────────────────────────────────────────
#     def process(self, frame: np.ndarray) -> dict:
#         """
#         Process one BGR frame.

#         Mutates and returns self.state — a dict with all inference results.
#         Also draws overlays directly onto the frame (in-place).
#         """
#         self._frame_n += 1
#         h, w = frame.shape[:2]
#         rgb  = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#         res  = self._face_mesh.process(rgb)

#         state = dict(self.state)
#         state["frame"] = self._frame_n

#         if not res.multi_face_landmarks:
#             self.state = state
#             return state

#         lms = res.multi_face_landmarks[0].landmark

#         # ── bounding box ──
#         xs = [l.x for l in lms]; ys = [l.y for l in lms]
#         x0 = int(min(xs)*w); x1 = int(max(xs)*w)
#         y0 = int(min(ys)*h); y1 = int(max(ys)*h)
#         fw = x1 - x0; fh = y1 - y0

#         dist = estimate_distance(fw)
#         state["dist"] = dist

#         # ── head pose (always computed — needed for gaze screen point) ──
#         pv = get_head_pose_vector(lms, w, h)
#         state["yaw"]   = float(pv[0])
#         state["pitch"] = float(pv[1])
#         state["roll"]  = float(pv[2])

#         # ── gaze screen position (always computed) ──
#         gaze_nx, gaze_ny = estimate_gaze_screen_point(lms, w, h)
#         state["gaze_nx"] = gaze_nx
#         state["gaze_ny"] = gaze_ny

#         if dist <= MAX_DISTANCE_CM:
#             # ── gaze model ──
#             if self._models_ok:
#                 gl, gc = self._predict_gaze(get_gaze_vector(lms, w, h))
#                 pl, pc = self._predict_pose(pv)
#             else:
#                 gl, gc, pl, pc = self._demo_predictions()

#             # ── DeepFace (throttled) ──
#             if self._frame_n % DEEPFACE_EVERY_N == 0:
#                 self._df_label = deepface_attention(frame, (x0, y0, fw, fh))
#                 state["df_label"]   = self._df_label
#             else:
#                 state["df_label"] = self._df_label

#             # ── final decision: AND logic ──
#             focused = (gl == "Focused") and (pl == "Focused") and \
#                       (self._df_label != "Distracted")

#             state.update(
#                 gaze_label=gl, gaze_conf=gc,
#                 pose_label=pl, pose_conf=pc,
#                 focused=focused,
#             )

#             # ── annotate frame for camera preview ──
#             draw_ui(frame, (x0, y0, fw, fh),
#                     dist, gl, gc, pl, pc,
#                     self._df_label, gaze_nx, gaze_ny)

#         self.state = state
#         return state

#     # ── private helpers ──────────────────────────────────────────────────────
#     def _predict_gaze(self, gaze_vec: np.ndarray):
#         """Pad to GAZE_FEATURES (43) and run gaze model."""
#         if len(gaze_vec) < GAZE_FEATURES:
#             gaze_vec = np.pad(gaze_vec, (0, GAZE_FEATURES - len(gaze_vec)))
#         else:
#             gaze_vec = gaze_vec[:GAZE_FEATURES]

#         pred = self._gaze_model.predict(
#             self._gaze_scaler.transform(gaze_vec.reshape(1, -1)),
#             verbose=0
#         )[0][0]
#         return ("Distracted", float(pred)) if pred > 0.5 else ("Focused", float(pred))

#     def _predict_pose(self, pose_vec: np.ndarray):
#         """Pad / trim to expected pose features and run pose model."""
#         exp = self._exp_pose
#         if len(pose_vec) < exp:
#             pose_vec = np.pad(pose_vec, (0, exp - len(pose_vec)))
#         else:
#             pose_vec = pose_vec[:exp]

#         pred = self._pose_model.predict(
#             self._pose_scaler.transform(pose_vec.reshape(1, -1)),
#             verbose=0
#         )[0][0]
#         return ("Distracted", float(pred)) if pred > 0.5 else ("Focused", float(pred))

#     def _demo_predictions(self):
#         """Smooth sinusoidal demo predictions when models are missing."""
#         t  = self._frame_n / 30.0
#         gc = max(0.0, min(1.0, 0.68 + 0.28 * math.sin(t * 0.32)))
#         pc = max(0.0, min(1.0, 0.74 + 0.22 * math.cos(t * 0.26)))
#         gl = "Distracted" if gc > 0.82 else "Focused"
#         pl = "Distracted" if pc > 0.82 else "Focused"
#         return gl, gc, pl, pc


# # ══════════════════════════════════════════════════════════════════════════════
# #  Standalone debug mode  —  raw OpenCV window, no Qt needed
# #  Run:  python main.py
# # ══════════════════════════════════════════════════════════════════════════════
# if __name__ == "__main__":
#     print("[INFO] Running main.py in standalone debug mode")
#     print("[INFO] Press  ESC  to quit")

#     engine = AttentionEngine()
#     engine.load()

#     cap = cv2.VideoCapture(0)
#     if not cap.isOpened():
#         raise RuntimeError("Cannot open camera")

#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             break

#         state = engine.process(frame)

#         h, w = frame.shape[:2]
#         cv2.putText(frame, f"Range: <= {MAX_DISTANCE_CM} cm",
#                     (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
#         cv2.putText(frame, f"Frame: {state['frame']}",
#                     (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)
#         cv2.putText(frame,
#                     f"Gaze screen: ({state['gaze_nx']:.2f}, {state['gaze_ny']:.2f})",
#                     (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 255, 100), 1)

#         cv2.imshow("FocusSense — Debug", frame)
#         if cv2.waitKey(1) & 0xFF == 27:
#             break

#     cap.release()
#     cv2.destroyAllWindows()