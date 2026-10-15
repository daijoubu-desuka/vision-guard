"""
object_tracker.py - Topics (b) SEGMENTATION and (e) OBJECT TRACKING.

Colour based tracker:
    1. The user selects an object (cv2.selectROI).
    2. We convert the region to HSV and build a HUE histogram of it.
    3. Every new frame: cv2.calcBackProject turns the frame into a probability
       image ("how much does this pixel look like my object?")  -> this is a
       colour SEGMENTATION of the object.
    4. cv2.CamShift moves/resizes/rotates the search window to the peak of that
       probability image -> TRACKING.
"""
import cv2
import numpy as np

import config


class CamShiftTracker:
    def __init__(self):
        self.reset()

    def reset(self):
        self.hist = None
        self.window = None
        self.template = None
        self.initialised = False
        self.lost = False
        self.confidence = 0.0

    def init(self, frame, roi):
        """Learn the colour model of the selected region roi=(x, y, w, h)."""
        x, y, w, h = [int(v) for v in roi]
        if w < 5 or h < 5:
            return False
        patch = frame[y: y + h, x: x + w]
        hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
        # ignore dark / washed-out pixels: their hue is meaningless
        mask = cv2.inRange(hsv, (0, config.CAMSHIFT_MIN_SATURATION, config.CAMSHIFT_MIN_VALUE),
                           (180, 255, 255))
        self.hist = cv2.calcHist([hsv], [0], mask, [32], [0, 180])
        cv2.normalize(self.hist, self.hist, 0, 255, cv2.NORM_MINMAX)
        self.window = (x, y, w, h)
        self.template = patch.copy()
        self.initialised, self.lost = True, False
        return True

    def update(self, frame):
        """
        Track the object in a new frame.
        Returns (rotated_rect, back_projection) - rotated_rect is None if lost.
        """
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        back = cv2.calcBackProject([hsv], [0], self.hist, [0, 180], 1)
        valid = cv2.inRange(hsv, (0, config.CAMSHIFT_MIN_SATURATION, config.CAMSHIFT_MIN_VALUE),
                            (180, 255, 255))
        back = cv2.bitwise_and(back, valid)               # remove unreliable pixels

        criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1)
        rot_rect, window = cv2.CamShift(back, self.window, criteria)

        x, y, w, h = window
        self.confidence = float(back[y: y + h, x: x + w].mean() / 255.0) if w * h > 0 else 0.0
        if w * h < config.CAMSHIFT_LOST_MIN_AREA:
            self.lost = True
            return None, back
        self.lost = False
        self.window = window
        return rot_rect, back

    @staticmethod
    def hue_histogram_image(hist, size=(280, 56)):
        """Visualise the learned hue histogram, each bar painted in its own hue."""
        w, h = size
        canvas = np.zeros((h, w, 3), np.uint8)
        if hist is None:
            return canvas
        vals = hist.flatten() / (hist.max() + 1e-6)
        bw = w / len(vals)
        for i, v in enumerate(vals):
            colour = cv2.cvtColor(np.uint8([[[int(i * 180 / len(vals)), 255, 255]]]),
                                  cv2.COLOR_HSV2BGR)[0][0]
            cv2.rectangle(canvas, (int(i * bw), h - 1), (int((i + 1) * bw) - 1, h - 1 - int(v * (h - 4))),
                          tuple(int(c) for c in colour), -1)
        return canvas
