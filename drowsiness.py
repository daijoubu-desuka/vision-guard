"""
drowsiness.py - Topics (c) HAAR CASCADE, (f) FACIAL LANDMARKS, (d) ANALYSIS & TRANSFORMATION.

Pipeline for every frame
    1. Haar cascade finds the face (and, inside it, the eyes).
    2. The LBF facial-landmark model places 68 points on the face.
    3. From the landmarks we compute
         EAR  - Eye Aspect Ratio   (how open the eyes are)
         MAR  - Mouth Aspect Ratio (how open the mouth is -> yawning)
         roll - head tilt angle
    4. A small state machine converts those numbers into
         AWAKE / WARNING / DROWSY / NO FACE.
    5. An affine rotation (cv2.warpAffine) straightens the face for display.

EAR (Soukupova & Cech, 2016):
            ||p2-p6|| + ||p3-p5||
     EAR = -----------------------
                2 * ||p1-p4||
    p1,p4 = eye corners, p2,p3 = upper-lid points, p6,p5 = lower-lid points.
    EAR ~0.30 for an open eye and falls towards 0 as the eye closes.
"""
from collections import deque
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import cv2
import numpy as np

import config
import utils

# --- 68-point landmark index groups (iBUG 300-W convention) -----------------
RIGHT_EYE = list(range(36, 42))   # person's right eye (left side of the image)
LEFT_EYE = list(range(42, 48))
INNER_MOUTH = list(range(60, 68))
OUTER_MOUTH = list(range(48, 60))
JAW = list(range(0, 17))


def _dist(a, b):
    return float(np.linalg.norm(np.asarray(a, float) - np.asarray(b, float)))


def eye_aspect_ratio(eye):
    """EAR for one eye given its 6 landmark points (see module docstring)."""
    vertical = _dist(eye[1], eye[5]) + _dist(eye[2], eye[4])
    horizontal = 2.0 * _dist(eye[0], eye[3])
    return vertical / horizontal if horizontal > 0 else 0.0


def mouth_aspect_ratio(mouth):
    """MAR from the 8 inner-lip points (indices 60-67). Large value = yawn."""
    vertical = _dist(mouth[1], mouth[7]) + _dist(mouth[2], mouth[6]) + _dist(mouth[3], mouth[5])
    horizontal = 2.0 * _dist(mouth[0], mouth[4])
    return vertical / horizontal if horizontal > 0 else 0.0


def head_roll_degrees(pts):
    """Angle of the line joining both eye centres (0 deg = head upright)."""
    right_c = pts[RIGHT_EYE].mean(axis=0)
    left_c = pts[LEFT_EYE].mean(axis=0)
    dx, dy = left_c[0] - right_c[0], left_c[1] - right_c[1]
    return float(np.degrees(np.arctan2(dy, dx)))


@dataclass
class DrowsinessResult:
    """Everything the HUD needs to draw one frame."""
    face: Optional[Tuple[int, int, int, int]] = None
    eyes: List[Tuple[int, int, int, int]] = field(default_factory=list)  # Haar eye boxes
    landmarks: Optional[np.ndarray] = None                                # (68, 2)
    ear: float = 0.0
    mar: float = 0.0
    roll: float = 0.0
    ear_threshold: float = config.EAR_FALLBACK_THRESHOLD
    eyes_closed: bool = False
    yawning: bool = False
    state: str = "NO FACE"
    closed_duration: float = 0.0
    perclos: float = 0.0
    blinks: int = 0
    yawns: int = 0
    calibrated: bool = False
    baseline_ear: float = 0.0
    aligned_face: Optional[np.ndarray] = None
    events: List[str] = field(default_factory=list)   # new events this frame (for the log)


class DrowsinessDetector:
    """Haar face/eye detection + landmark based drowsiness logic."""

    def __init__(self, landmark_model_path):
        # load_cascade() tries OpenCV's data folder, then models/, then downloads the XML
        self.face_cascade = utils.load_cascade("haarcascade_frontalface_alt2.xml")
        self.eye_cascade = utils.load_cascade("haarcascade_eye.xml")

        self.facemark = cv2.face.createFacemarkLBF()
        self.facemark.loadModel(landmark_model_path)

        self.reset()

    # ------------------------------------------------------------------ state
    def reset(self):
        """Forget calibration and counters (key 'c' / 'r' in the app)."""
        self._face_smooth = None
        self._last_face = None
        self._missed = 0
        self._cal_samples: List[float] = []
        self.baseline_ear = 0.0
        self.calibrated = False
        self._closed_since = None
        self._yawn_since = None
        self._yawn_counted = False
        self._no_face_since = None
        self._away_logged = False
        self._history = deque()          # (time, closed?) for PERCLOS
        self._ear_hist = deque(maxlen=5)  # tiny moving average to calm landmark jitter
        self._mar_hist = deque(maxlen=5)
        self._was_drowsy = False
        self.blinks = 0
        self.yawns = 0

    # --------------------------------------------------------- Haar detection
    def detect_face(self, gray):
        """Return the largest face box (x, y, w, h) or None (Haar cascade)."""
        faces = self.face_cascade.detectMultiScale(
            gray, scaleFactor=config.FACE_SCALE_FACTOR, minNeighbors=5, minSize=config.FACE_MIN_SIZE
        )
        if len(faces) == 0:
            self._missed += 1
            if self._last_face is not None and self._missed <= config.FACE_GRACE_FRAMES:
                return self._last_face           # short grace period, avoids flicker
            self._face_smooth = None
            self._last_face = None
            return None

        self._missed = 0
        x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
        box = np.array([x, y, w, h], dtype=float)
        if self._face_smooth is None:
            self._face_smooth = box
        else:   # exponential smoothing -> steadier landmarks
            a = config.FACE_SMOOTHING
            self._face_smooth = a * self._face_smooth + (1 - a) * box
        self._last_face = tuple(int(v) for v in self._face_smooth)
        return self._last_face

    def detect_eyes(self, gray, face):
        """Haar eye cascade, searched only in the upper half of the face."""
        x, y, w, h = face
        roi = gray[y + int(0.15 * h): y + int(0.60 * h), x: x + w]
        if roi.size == 0:
            return []
        eyes = self.eye_cascade.detectMultiScale(
            roi, scaleFactor=1.1, minNeighbors=6, minSize=(max(w // 8, 12), max(w // 8, 12))
        )
        eyes = sorted(eyes, key=lambda e: e[2] * e[3], reverse=True)[:2]
        return [(x + ex, y + int(0.15 * h) + ey, ew, eh) for ex, ey, ew, eh in eyes]

    # -------------------------------------------------------------- landmarks
    def detect_landmarks(self, gray, face):
        """68 facial landmarks inside the Haar face box (LBF regression model)."""
        rect = np.array([face], dtype=np.int32)
        ok, shapes = self.facemark.fit(gray, rect)
        if not ok or len(shapes) == 0:
            return None
        # OpenCV versions differ in shape ((1,68,2) vs (68,1,2)) -> always flatten to (68, 2)
        return np.asarray(shapes[0], dtype=np.float32).reshape(-1, 2)

    # ----------------------------------------------------------- transformation
    @staticmethod
    def align_face(frame, pts, face, out_size=128):
        """
        Affine transformation: rotate the frame about the eye mid-point so the
        eyes become horizontal, then crop the face. Demonstrates
        cv2.getRotationMatrix2D + cv2.warpAffine (topic d).
        """
        right_c = pts[RIGHT_EYE].mean(axis=0)
        left_c = pts[LEFT_EYE].mean(axis=0)
        center = (float((right_c[0] + left_c[0]) / 2), float((right_c[1] + left_c[1]) / 2))
        angle = head_roll_degrees(pts)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(frame, M, (frame.shape[1], frame.shape[0]))
        x, y, w, h = face
        x0, y0 = max(x, 0), max(y, 0)
        crop = rotated[y0: y + h, x0: x + w]
        if crop.size == 0:
            return None
        return cv2.resize(crop, (out_size, out_size))

    # ------------------------------------------------------------ main entry
    def analyse(self, frame, gray, now) -> DrowsinessResult:
        """Run the full pipeline on one frame. `now` is a timestamp in seconds."""
        res = DrowsinessResult(blinks=self.blinks, yawns=self.yawns,
                               calibrated=self.calibrated, baseline_ear=self.baseline_ear)

        face = self.detect_face(gray)
        if face is None:
            return self._no_face(res, now)
        self._no_face_since = None
        self._away_logged = False
        res.face = face
        res.eyes = self.detect_eyes(gray, face)

        pts = self.detect_landmarks(gray, face)
        if pts is None:
            return self._no_face(res, now)
        res.landmarks = pts

        # --- measurements (smoothed over the last few frames)
        ear_raw = (eye_aspect_ratio(pts[RIGHT_EYE]) + eye_aspect_ratio(pts[LEFT_EYE])) / 2.0
        mar_raw = mouth_aspect_ratio(pts[INNER_MOUTH])
        self._ear_hist.append(ear_raw)
        self._mar_hist.append(mar_raw)
        res.ear = float(np.mean(self._ear_hist))
        res.mar = float(np.mean(self._mar_hist))
        res.roll = head_roll_degrees(pts)
        res.aligned_face = self.align_face(frame, pts, face)

        self._calibrate(res.ear, res)
        res.ear_threshold = self._ear_threshold()
        self._update_state(res, now)
        return res

    # ----------------------------------------------------------- internals
    def _calibrate(self, ear, res):
        """Learn this person's normal open-eye EAR from the first frames."""
        if self.calibrated:
            return
        self._cal_samples.append(ear)
        if len(self._cal_samples) >= config.CALIBRATION_FRAMES:
            baseline = float(np.median(self._cal_samples))
            # an absurd baseline (eyes shut during calibration) -> keep fallback
            if 0.18 <= baseline <= 0.45:
                self.baseline_ear = baseline
            self.calibrated = True
        res.calibrated, res.baseline_ear = self.calibrated, self.baseline_ear

    def _ear_threshold(self):
        if self.calibrated and self.baseline_ear > 0:
            return self.baseline_ear * config.EAR_CLOSED_RATIO
        return config.EAR_FALLBACK_THRESHOLD

    def _update_state(self, res, now):
        # ---- eyes closed? (landmark EAR is primary; Haar eye count is a cue)
        closed = res.ear < res.ear_threshold
        if not closed and len(res.eyes) == 0 and res.ear < res.ear_threshold * 1.15:
            closed = True     # borderline EAR *and* Haar sees no eyes -> closed
        res.eyes_closed = closed

        if closed:
            if self._closed_since is None:
                self._closed_since = now
            res.closed_duration = now - self._closed_since
        else:
            if self._closed_since is not None:
                dur = now - self._closed_since
                if config.BLINK_MIN_SECONDS <= dur <= config.BLINK_MAX_SECONDS:
                    self.blinks += 1
            self._closed_since = None

        # ---- yawning? (mouth open continuously)
        if res.mar > config.MAR_YAWN_THRESHOLD:
            if self._yawn_since is None:
                self._yawn_since = now
            if now - self._yawn_since >= config.YAWN_SECONDS:
                res.yawning = True
                if not self._yawn_counted:
                    self.yawns += 1
                    self._yawn_counted = True
                    res.events.append("YAWN")
        else:
            self._yawn_since = None
            self._yawn_counted = False

        # ---- PERCLOS = fraction of recent time with eyes closed
        self._history.append((now, closed))
        while self._history and now - self._history[0][0] > config.PERCLOS_WINDOW:
            self._history.popleft()
        if len(self._history) > 20:
            res.perclos = sum(1 for _, c in self._history if c) / len(self._history)

        # ---- final state
        if res.closed_duration >= config.EYE_CLOSED_SECONDS:
            res.state = "DROWSY"
        elif res.yawning or res.perclos >= config.PERCLOS_WARN:
            res.state = "WARNING"
        else:
            res.state = "AWAKE"

        if res.state == "DROWSY" and not self._was_drowsy:
            res.events.append("DROWSY")
        self._was_drowsy = res.state == "DROWSY"
        res.blinks, res.yawns = self.blinks, self.yawns

    def _no_face(self, res, now):
        """Handle frames without a usable face."""
        self._closed_since = None
        self._yawn_since = None
        if self._no_face_since is None:
            self._no_face_since = now
        away = now - self._no_face_since >= config.NO_FACE_SECONDS
        res.state = "AWAY" if away else "SEARCHING"
        if away and not self._away_logged:      # log the event only once per absence
            self._away_logged = True
            res.events.append("AWAY")
        res.blinks, res.yawns = self.blinks, self.yawns
        return res
