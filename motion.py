"""
motion.py - Topics (e) MOTION & OBJECT TRACKING, (b) IMAGE SEGMENTATION,
            (d) IMAGE ANALYSIS (contours, moments) and (c) HAAR CASCADE.

Mode 2 = PEOPLE COUNTER (still driven by motion)
    1. MOG2 background subtraction  -> foreground mask      (segmentation)
    2. threshold + morphology       -> clean binary mask    (segmentation)
    3. findContours + moments       -> boxes & centroids    (analysis)
    4. CentroidTracker              -> persistent object IDs, trails, speed (tracking)
    5. PeopleCounter                -> which moving blobs are PEOPLE (Haar face inside the blob,
                                       or a full-body shape) + people who stand still (Haar face
                                       anywhere in the frame) -> number of people in view,
                                       smoothed by a majority vote, ALERT when more than one
    6. LineCounter                  -> people / objects IN and OUT of a virtual line
"""
from collections import Counter, OrderedDict, deque
from dataclasses import dataclass, field
from typing import List, Tuple

import cv2
import numpy as np

import config
import utils


def load_face_cascade():
    """Haar frontal-face cascade, or None if it cannot be loaded (motion detection must still work)."""
    try:
        return utils.load_cascade("haarcascade_frontalface_alt2.xml")
    except (SystemExit, Exception) as exc:
        print(f"[warning] Haar face cascade unavailable ({exc}). Face based people counting disabled.")
        return None


# ============================================================================
# Object tracker (centroid based)
# ============================================================================
@dataclass
class TrackedObject:
    object_id: int
    centroid: Tuple[int, int]
    bbox: Tuple[int, int, int, int]
    trail: deque = field(default_factory=lambda: deque(maxlen=config.TRACK_TRAIL_LENGTH))
    disappeared: int = 0
    hits: int = 1                  # frames this object has been matched (confirmation)
    side: int = 0                  # which side of the counting line (-1 above, +1 below)
    velocity: Tuple[float, float] = (0.0, 0.0)
    is_person: bool = False
    people: int = 0                # how many people this blob contains (0 = not a person)

    @property
    def confirmed(self):
        """Only objects seen for several frames count - ignores one-frame noise."""
        return self.hits >= config.TRACK_CONFIRM_FRAMES


class CentroidTracker:
    """
    Assigns a persistent ID to every moving blob.

    Each frame: compute the distance between every existing centroid and every
    new detection, then greedily match the closest pairs. Unmatched detections
    become new objects; objects unmatched for too long are removed.
    """

    def __init__(self, max_disappeared=config.TRACK_MAX_DISAPPEARED,
                 max_distance=config.TRACK_MAX_DISTANCE):
        self.next_id = 0
        self.objects: "OrderedDict[int, TrackedObject]" = OrderedDict()
        self.max_disappeared = max_disappeared
        self.max_distance = max_distance

    def reset(self):
        self.next_id = 0
        self.objects.clear()

    def _register(self, centroid, bbox):
        obj = TrackedObject(self.next_id, centroid, bbox)
        obj.trail.append(centroid)
        self.objects[self.next_id] = obj
        self.next_id += 1

    def update(self, detections):
        """detections: list of (bbox, centroid). Returns the dict of live objects."""
        if len(detections) == 0:
            for oid in list(self.objects):
                self.objects[oid].disappeared += 1
                if self.objects[oid].disappeared > self.max_disappeared:
                    del self.objects[oid]
            return self.objects

        if len(self.objects) == 0:
            for bbox, c in detections:
                self._register(c, bbox)
            return self.objects

        ids = list(self.objects.keys())
        old = np.array([self.objects[i].centroid for i in ids], dtype=float)
        new = np.array([c for _, c in detections], dtype=float)
        dist = np.linalg.norm(old[:, None, :] - new[None, :, :], axis=2)

        used_old, used_new = set(), set()
        for flat in np.argsort(dist, axis=None):          # closest pairs first
            r, c = divmod(int(flat), dist.shape[1])
            if r in used_old or c in used_new or dist[r, c] > self.max_distance:
                continue
            obj = self.objects[ids[r]]
            bbox, centroid = detections[c]
            obj.velocity = (centroid[0] - obj.centroid[0], centroid[1] - obj.centroid[1])
            obj.centroid, obj.bbox, obj.disappeared = centroid, bbox, 0
            obj.hits += 1
            obj.trail.append(centroid)
            used_old.add(r)
            used_new.add(c)

        for r, oid in enumerate(ids):                      # lost objects
            if r not in used_old:
                self.objects[oid].disappeared += 1
                if self.objects[oid].disappeared > self.max_disappeared:
                    del self.objects[oid]
        for c, (bbox, centroid) in enumerate(detections):  # new objects
            if c not in used_new:
                self._register(centroid, bbox)
        return self.objects


# ============================================================================
# Virtual line counter
# ============================================================================
class LineCounter:
    """Counts objects crossing a horizontal line: downwards = IN, upwards = OUT."""

    def __init__(self, frame_height):
        self.line_y = int(frame_height * config.COUNT_LINE_POSITION)
        self.count_in = 0
        self.count_out = 0

    def reset(self):
        self.count_in = self.count_out = 0

    def update(self, objects):
        events = []
        m = config.COUNT_LINE_MARGIN
        for obj in objects.values():
            if obj.disappeared > 0:
                continue
            y = obj.centroid[1]
            new_side = -1 if y < self.line_y - m else (1 if y > self.line_y + m else 0)
            if new_side != 0:
                # side is remembered from the first sighting; only CONFIRMED objects are counted
                if obj.side != 0 and new_side != obj.side and obj.confirmed:      # crossed the line!
                    if new_side == 1:
                        self.count_in += 1
                        events.append(f"IN id={obj.object_id}")
                    else:
                        self.count_out += 1
                        events.append(f"OUT id={obj.object_id}")
                obj.side = new_side
        return events


def merge_detections(pieces, gap=config.MOTION_MERGE_GAP):
    """
    A walking person is usually detected as several pieces (head, arms, torso).
    Pieces whose boxes are closer than `gap` pixels are merged into ONE object.
    pieces = [(x, y, w, h, area, cx, cy)] -> merged list of the same format.
    """
    items = [list(p) for p in pieces]

    def near(a, b):
        return (a[0] - gap <= b[0] + b[2] and b[0] - gap <= a[0] + a[2] and
                a[1] - gap <= b[1] + b[3] and b[1] - gap <= a[1] + a[3])

    merged = True
    while merged:
        merged = False
        for i in range(len(items)):
            for j in range(i + 1, len(items)):
                if near(items[i], items[j]):
                    a, b = items[i], items[j]
                    x0, y0 = min(a[0], b[0]), min(a[1], b[1])
                    x1, y1 = max(a[0] + a[2], b[0] + b[2]), max(a[1] + a[3], b[1] + b[3])
                    area = a[4] + b[4]
                    cx = (a[5] * a[4] + b[5] * b[4]) / area       # area-weighted centre
                    cy = (a[6] * a[4] + b[6] * b[4]) / area
                    items[i] = [x0, y0, x1 - x0, y1 - y0, area, cx, cy]
                    del items[j]
                    merged = True
                    break
            if merged:
                break
    return [tuple(i) for i in items]


# ============================================================================
# Motion detector
# ============================================================================
@dataclass
class MotionResult:
    mask: np.ndarray
    detections: List[Tuple[Tuple[int, int, int, int], Tuple[int, int]]]
    warming_up: bool
    motion_percent: float       # % of the frame that is moving
    face_boxes: List[Tuple[int, int, int, int]]
    global_change: bool = False # True when almost the whole frame changed (lighting)


class MotionDetector:
    """MOG2 background subtraction + morphological clean-up + contour analysis."""

    def __init__(self, detect_faces=True):
        # detect_faces=False: the PeopleCounter does the (full-frame) face search instead.
        # The cascade only adds the "PERSON" label. If it cannot be loaded, motion
        # detection must still work, so we continue without it instead of crashing.
        self.face_cascade = load_face_cascade() if detect_faces else None
        self.kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        k = config.MOTION_CLOSE_KERNEL
        self.close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        self.reset()

    def reset(self):
        # detectShadows=True marks shadows grey (127) so we can threshold them away
        self.bg = cv2.createBackgroundSubtractorMOG2(
            history=config.MOG2_HISTORY, varThreshold=config.MOG2_VAR_THRESHOLD,
            detectShadows=True)
        self.frames_seen = 0

    def process(self, blurred_bgr, gray) -> MotionResult:
        self.frames_seen += 1
        fg = self.bg.apply(blurred_bgr, learningRate=config.MOG2_LEARNING_RATE)

        # --- segmentation clean-up
        _, mask = cv2.threshold(fg, 200, 255, cv2.THRESH_BINARY)   # drop shadow pixels (127)
        mask = cv2.medianBlur(mask, 5)                              # remove salt-and-pepper
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self.kernel, iterations=1)
        mask = cv2.dilate(mask, self.kernel, iterations=2)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self.close_kernel)  # merge fragments of one body

        warming = self.frames_seen <= config.MOTION_WARMUP_FRAMES
        percent = 100.0 * cv2.countNonZero(mask) / mask.size
        global_change = percent > config.MOTION_GLOBAL_CHANGE      # light switched / exposure jump
        detections, faces = [], []
        if not warming and not global_change:
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            pieces = []
            for cnt in contours:
                area = cv2.contourArea(cnt)
                if area < config.MOTION_FRAGMENT_AREA:
                    continue
                x, y, w, h = cv2.boundingRect(cnt)
                m = cv2.moments(cnt)                                # centre of mass
                cx = m["m10"] / m["m00"] if m["m00"] else x + w / 2
                cy = m["m01"] / m["m00"] if m["m00"] else y + h / 2
                pieces.append((x, y, w, h, area, cx, cy))

            for x, y, w, h, area, cx, cy in merge_detections(pieces):   # one box per person
                if area < config.MOTION_MIN_AREA:
                    continue
                detections.append(((int(x), int(y), int(w), int(h)), (int(cx), int(cy))))

                # Haar cascade only inside the moving blob -> is it a person?
                if h > 90 and self.face_cascade is not None:
                    roi = gray[int(y): int(y + h * 0.6), int(x): int(x + w)]
                    found = self.face_cascade.detectMultiScale(
                        roi, 1.1, 5, minSize=(40, 40)) if roi.size else []
                    for fx, fy, fw, fh in found:
                        faces.append((int(x + fx), int(y + fy), fw, fh))

        return MotionResult(mask, detections, warming, percent, faces, global_change)

    @staticmethod
    def mark_persons(objects, face_boxes):
        """A tracked blob that contains a Haar face is labelled PERSON."""
        for obj in objects.values():
            x, y, w, h = obj.bbox
            for fx, fy, fw, fh in face_boxes:
                if x <= fx + fw / 2 <= x + w and y <= fy + fh / 2 <= y + h:
                    obj.is_person = True


# ============================================================================
# People counter
# ============================================================================
@dataclass
class PeopleResult:
    faces: List[Tuple[int, int, int, int]]              # every Haar face found in the frame
    standalone_faces: List[Tuple[int, int, int, int]]   # faces of people who are NOT inside a moving blob
    raw_count: int                                      # this frame's count (before smoothing)
    count: int                                          # stable count shown to the user
    peak: int                                           # highest stable count so far
    multiple: bool                                      # True while count >= PEOPLE_ALERT_COUNT


class PeopleCounter:
    """
    Counts the people in view in real time.

    People come from two sources that never overlap:
      A) MOVING people  - a tracked motion blob that contains a Haar face (1 person per face, so a
         merged blob holding two faces counts 2) or has a full-body shape (tall and narrow).
      B) STILL people   - a Haar face found anywhere in the frame that is not inside a moving blob
         (somebody sitting or standing still is absorbed into the MOG2 background, but the face
         search still sees them).
    count = A + B for this frame; a majority vote over the last few frames gives the stable
    count, so one missed or false detection does not flip the number or trigger the beep.
    """

    def __init__(self):
        self.face_cascade = load_face_cascade()
        self.reset()

    def reset(self):
        self.history = deque(maxlen=config.PEOPLE_SMOOTH_FRAMES)
        self.count = 0
        self.peak = 0
        self._frame_no = 0
        self._faces = []

    def detect_faces(self, gray_eq):
        if self.face_cascade is None:
            return []
        found = self.face_cascade.detectMultiScale(
            gray_eq, scaleFactor=config.PEOPLE_FACE_SCALE, minNeighbors=config.PEOPLE_FACE_NEIGHBORS,
            minSize=config.PEOPLE_FACE_MIN_SIZE)
        return [tuple(int(v) for v in f) for f in found]

    @staticmethod
    def _inside(face, bbox):
        fx, fy, fw, fh = face
        x, y, w, h = bbox
        return x <= fx + fw / 2 <= x + w and y <= fy + fh / 2 <= y + h

    @staticmethod
    def _nearest_free_face(bbox, faces, used):
        """Index of the closest face not yet used, if it is within PEOPLE_FACE_LINK px of the box."""
        x, y, w, h = bbox
        best, best_d = None, config.PEOPLE_FACE_LINK
        for i, (fx, fy, fw, fh) in enumerate(faces):
            if i in used:
                continue
            cx, cy = fx + fw / 2, fy + fh / 2
            d = float(np.hypot(max(x - cx, 0, cx - (x + w)), max(y - cy, 0, cy - (y + h))))
            if d <= best_d:
                best, best_d = i, d
        return best

    def update(self, objects, gray_eq, frame_h) -> PeopleResult:
        if self._frame_no % config.PEOPLE_FACE_EVERY == 0:      # full-frame search is the slowest step,
            self._faces = self.detect_faces(gray_eq)           # so it runs every N-th frame only
        self._frame_no += 1
        faces = self._faces
        lo, hi = config.PEOPLE_BODY_ASPECT
        used = set()                                    # indices of faces that belong to a blob
        candidates, faceless = [], []
        for obj in objects.values():
            obj.people = 0
            if obj.disappeared > config.PEOPLE_HOLD_FRAMES or obj.hits < config.PEOPLE_CONFIRM_FRAMES:
                continue
            inside = [i for i, f in enumerate(faces) if self._inside(f, obj.bbox)]
            if inside:                                  # a face inside the blob: it is a person
                obj.is_person = True
                obj.people = len(inside)                # two faces in one merged blob = two people
                used.update(inside)
            else:
                faceless.append(obj)
            candidates.append(obj)

        for obj in faceless:                            # blobs without a visible face
            if obj.disappeared == 0:
                x, y, w, h = obj.bbox
                if h >= config.PEOPLE_BODY_MIN_HEIGHT * frame_h and lo <= h / max(w, 1) <= hi:
                    obj.is_person = True                # full-body shape (e.g. back turned)
            if obj.is_person:
                obj.people = 1
                near = self._nearest_free_face(obj.bbox, faces, used)
                if near is not None:                    # e.g. a person who just stopped: the leftover blob
                    used.add(near)                      # and the face are ONE person, not two
        raw = sum(o.people for o in candidates)
        standalone = [f for i, f in enumerate(faces) if i not in used]
        raw += len(standalone)

        self.history.append(raw)
        winner, votes = Counter(self.history).most_common(1)[0]
        if votes >= config.PEOPLE_MIN_VOTES:
            self.count = winner
        self.peak = max(self.peak, self.count)
        return PeopleResult(faces, standalone, raw, self.count, self.peak,
                            self.count >= config.PEOPLE_ALERT_COUNT)
