"""
utils.py - Helpers that are not computer vision themselves:
model download, Haar cascade loading, FPS counter, CSV event log, snapshots, sound alarm.
"""
import csv
import os
import sys
import threading
import time
import urllib.request
from collections import deque
from datetime import datetime

import cv2

import config


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CASCADE_URL = "https://raw.githubusercontent.com/opencv/opencv/4.x/data/haarcascades/"


def ensure_landmark_model(path=config.LANDMARK_MODEL_FILE):
    """Download the 56 MB LBF facial-landmark model the first time it is needed."""
    if os.path.exists(path) and os.path.getsize(path) > 1_000_000:
        return path
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    print(f"[setup] Downloading facial landmark model to {path} (~56 MB, one time only)...")

    def progress(blocks, block_size, total):
        if total > 0:
            pct = min(100, blocks * block_size * 100 // total)
            print(f"\r[setup]   {pct:3d}%", end="", flush=True)

    try:
        urllib.request.urlretrieve(config.LANDMARK_MODEL_URL, path, progress)
        print("\n[setup] Done.")
    except Exception as exc:                       # network problems, proxy, etc.
        if os.path.exists(path):
            os.remove(path)
        raise SystemExit(
            f"\nCould not download the landmark model ({exc}).\n"
            f"Download it manually from:\n  {config.LANDMARK_MODEL_URL}\n"
            f"and save it as {path}"
        )
    return path


def _try_load(path):
    """Load a cascade normally; if the path breaks OpenCV, read it from memory."""
    c = cv2.CascadeClassifier(path)
    if not c.empty():
        return c
    try:  # fallback for non-ASCII paths on Windows
        with open(path, "r", encoding="utf-8") as f:
            xml = f.read()
        fs = cv2.FileStorage(xml, cv2.FILE_STORAGE_READ | cv2.FILE_STORAGE_MEMORY)
        c = cv2.CascadeClassifier()
        if c.read(fs.getFirstTopLevelNode()):
            return c
    except Exception:
        pass
    return None


def load_cascade(name):
    """Find a Haar cascade: OpenCV's data folder -> models/ -> download from GitHub."""
    local = os.path.join(BASE_DIR, "models", name)
    candidates = []
    data = getattr(cv2, "data", None)
    if data is not None:
        candidates.append(os.path.join(data.haarcascades, name))
    candidates.append(local)

    for p in candidates:
        if os.path.isfile(p):
            c = _try_load(p)
            if c is not None:
                return c

    # last resort: download the XML once into models/
    try:
        os.makedirs(os.path.dirname(local), exist_ok=True)
        print(f"[setup] Downloading {name} ...")
        urllib.request.urlretrieve(CASCADE_URL + name, local)
        c = _try_load(local)
        if c is not None:
            return c
    except Exception as exc:
        print("[setup] download failed:", exc)
    raise SystemExit(f"Could not load Haar cascade '{name}'. Download it from\n  {CASCADE_URL}{name}\n"
                     f"and save it in the models/ folder.")


class FPSCounter:
    """Moving-average frames-per-second."""

    def __init__(self, n=20):
        self.times = deque(maxlen=n)

    def tick(self):
        self.times.append(time.time())
        if len(self.times) < 2:
            return 0.0
        return (len(self.times) - 1) / (self.times[-1] - self.times[0] + 1e-9)


class EventLogger:
    """Appends important events (DROWSY, YAWN, COUNT, MULTIPLE_PEOPLE, IN/OUT...) to a CSV file."""

    def __init__(self, path=config.EVENT_LOG_FILE):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        new = not os.path.exists(path)
        self.file = open(path, "a", newline="", encoding="utf-8")
        self.writer = csv.writer(self.file)
        if new:
            self.writer.writerow(["timestamp", "mode", "event", "details"])

    def log(self, mode, event, details=""):
        self.writer.writerow([datetime.now().isoformat(timespec="seconds"), mode, event, details])
        self.file.flush()
        print(f"[event] {mode:<10} {event} {details}")

    def close(self):
        self.file.close()


def save_snapshot(frame, tag):
    """Save the current annotated frame; returns the file path."""
    os.makedirs(config.SNAPSHOT_DIR, exist_ok=True)
    name = f"{tag}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
    path = os.path.join(config.SNAPSHOT_DIR, name)
    cv2.imwrite(path, frame)
    return path


class Alarm:
    """Non-blocking beep. Windows -> winsound, others -> terminal bell."""

    def __init__(self, enabled=True):
        self.enabled = enabled
        self._last = 0.0

    def beep(self, min_interval=1.0, freq=1200, dur=350):
        """freq (Hz) / dur (ms) let each alarm have its own sound (Windows only; others use the bell)."""
        now = time.time()
        if not self.enabled or now - self._last < min_interval:
            return
        self._last = now
        threading.Thread(target=self._play, args=(freq, dur), daemon=True).start()

    @staticmethod
    def _play(freq=1200, dur=350):
        try:
            if sys.platform.startswith("win"):
                import winsound
                winsound.Beep(freq, dur)
            else:
                print("\a", end="", flush=True)
        except Exception:
            pass
