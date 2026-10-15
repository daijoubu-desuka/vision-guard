"""
mood.py - MOOD DETECTOR (rule based, no deep learning).
Topics: (f) FACIAL LANDMARKS  +  (c) HAAR CASCADE (smile)  +  (d) ANALYSIS.

Idea (easy to explain in the demo)
    1. From the 68 landmarks we measure a few distances, always divided by the
       face width so the numbers do not change when you move closer / farther:
           lift     - are the mouth corners ABOVE (smile) or BELOW (sad) the lip centre?
           smile_w  - how wide is the mouth?
           brow_gap - how far are the eyebrows above the eyes?
           furrow   - how close are the two INNER eyebrow ends? (frown)
           MAR / EAR - mouth openness / eye openness (from drowsiness.py)
    2. In the first ~1.5 s we record YOUR neutral face (baseline).
    3. Every frame we compare the new measurements with the baseline using
       simple IF rules  ->  HAPPY / SAD / SURPRISED / ANGRY / TIRED / NEUTRAL.
    4. A majority vote over the last 12 frames removes flicker.
"""
from collections import Counter, deque
from dataclasses import dataclass, field

import numpy as np

import config
import utils
from drowsiness import LEFT_EYE, RIGHT_EYE

MOODS = ["HAPPY", "NEUTRAL", "SAD", "SURPRISED", "ANGRY", "TIRED"]


def _d(a, b):
    return float(np.linalg.norm(np.asarray(a, float) - np.asarray(b, float)))


def mood_features(pts, mar, ear):
    """Measure the face (all distances are relative to face width)."""
    face_w = _d(pts[0], pts[16]) or 1.0                     # jaw left <-> jaw right
    smile_w = _d(pts[48], pts[54]) / face_w                 # mouth corner to corner
    corner_y = (pts[48][1] + pts[54][1]) / 2.0              # y grows DOWNWARDS in images
    lip_mid_y = (pts[51][1] + pts[57][1]) / 2.0             # centre of the lips
    lift = (lip_mid_y - corner_y) / face_w                  # > 0  : corners above centre
    eye_r_y = pts[RIGHT_EYE][:, 1].mean()
    eye_l_y = pts[LEFT_EYE][:, 1].mean()
    brow_r_y = pts[17:22][:, 1].mean()
    brow_l_y = pts[22:27][:, 1].mean()
    brow_gap = ((eye_r_y - brow_r_y) + (eye_l_y - brow_l_y)) / 2.0 / face_w
    furrow = _d(pts[21], pts[22]) / face_w                  # gap between inner brow ends
    return {"lift": lift, "smile_w": smile_w, "brow_gap": brow_gap,
            "furrow": furrow, "mar": float(mar), "ear": float(ear)}


def feature_deltas(f, base):
    """How different is the face from the neutral baseline? (used by rules + debug view)"""
    return {
        "lift": f["lift"] - base["lift"],
        "width": f["smile_w"] / max(base["smile_w"], 1e-6),
        "brow": f["brow_gap"] / max(base["brow_gap"], 1e-6),
        "furrow": f["furrow"] / max(base["furrow"], 1e-6),
        "mar": f["mar"] - base["mar"],
        "ear": f["ear"] / max(base["ear"], 1e-6),
    }


def classify_mood(f, base, haar_smile=False, fatigue=False):
    """The rule table. Order matters: the first rule that matches wins."""
    d = feature_deltas(f, base)
    if d["mar"] > config.MOOD_SURPRISE_MAR and d["brow"] > config.MOOD_SURPRISE_BROW:
        return "SURPRISED"                                   # mouth open + brows up
    if (d["lift"] > config.MOOD_SMILE_LIFT and d["width"] > config.MOOD_SMILE_WIDTH) or \
       (haar_smile and d["lift"] > config.MOOD_SMILE_LIFT_HAAR):
        return "HAPPY"                                       # corners up + wide (or Haar smile)
    if d["furrow"] < config.MOOD_ANGRY_FURROW and d["brow"] < config.MOOD_ANGRY_BROW:
        return "ANGRY"                                       # brows pulled together and down
    if d["lift"] < config.MOOD_SAD_LIFT and d["width"] < 1.03:
        return "SAD"                                         # corners dropped
    if fatigue or d["ear"] < config.MOOD_TIRED_EAR_RATIO:
        return "TIRED"                                       # droopy eyes / fatigue warning
    return "NEUTRAL"


@dataclass
class MoodResult:
    mood: str = "NO FACE"
    raw: str = ""
    confidence: float = 0.0
    calibrated: bool = False
    haar_smile: bool = False
    changed: bool = False
    deltas: dict = field(default_factory=dict)


class MoodDetector:
    def __init__(self):
        self.smile_cascade = utils.load_cascade("haarcascade_smile.xml")
        self.reset()

    def reset(self):
        """Forget the baseline (key C) - the user should show a neutral face."""
        self.baseline = None
        self._cal = []
        self.history = deque(maxlen=config.MOOD_SMOOTH_FRAMES)
        self.counts = Counter()
        self.timeline = deque(maxlen=140)
        self.mood = "NEUTRAL"
        self._frame = 0
        self._smile_cache = False

    @property
    def calibrated(self):
        return self.baseline is not None

    def set_baseline(self, features):
        self.baseline = dict(features)

    def _haar_smile(self, gray, face):
        """Haar smile cascade on the lower half of the face (every 3rd frame to save time)."""
        self._frame += 1
        if self._frame % 3 == 1 and face is not None:
            x, y, w, h = face
            roi = gray[y + int(0.55 * h): y + h, x: x + w]
            if roi.size:
                found = self.smile_cascade.detectMultiScale(
                    roi, scaleFactor=1.7, minNeighbors=20, minSize=(max(w // 5, 20), max(h // 12, 10)))
                self._smile_cache = len(found) > 0
        return self._smile_cache

    def update(self, res, gray):
        """res = DrowsinessResult of this frame. Returns a MoodResult."""
        if res.landmarks is None:
            return MoodResult(mood="NO FACE", calibrated=self.calibrated)
        f = mood_features(res.landmarks, res.mar, res.ear)

        if not self.calibrated:                               # step 2: learn the neutral face
            self._cal.append(f)
            if len(self._cal) >= config.MOOD_CALIBRATION_FRAMES:
                self.baseline = {k: float(np.median([c[k] for c in self._cal])) for k in f}
            return MoodResult(mood="CALIBRATING", calibrated=False)

        smile = self._haar_smile(gray, res.face)
        fatigue = res.state in ("WARNING", "DROWSY")
        raw = classify_mood(f, self.baseline, smile, fatigue)

        self.history.append(raw)                              # step 4: majority vote
        label, votes = Counter(self.history).most_common(1)[0]
        new = label if votes >= config.MOOD_MIN_VOTES else self.mood
        changed = new != self.mood
        self.mood = new
        self.counts[new] += 1
        self.timeline.append(new)
        return MoodResult(mood=new, raw=raw, confidence=votes / len(self.history), calibrated=True,
                          haar_smile=smile, changed=changed, deltas=feature_deltas(f, self.baseline))

    def percentages(self):
        total = sum(self.counts.values())
        return {m: (100.0 * self.counts[m] / total if total else 0.0) for m in MOODS}
