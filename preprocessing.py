"""
preprocessing.py - Topic (a) IMAGE OPERATIONS and part of topic (d) IMAGE ANALYSIS.

Everything that prepares a raw camera frame before detection happens here:
resizing, mirroring, colour-space conversion, brightness measurement, gamma
correction for dark rooms, CLAHE contrast enhancement and Gaussian blur.
"""
import cv2
import numpy as np

import config


def resize_to_width(frame, width=config.FRAME_WIDTH):
    """Resize a frame to `width` pixels wide while keeping the aspect ratio."""
    h, w = frame.shape[:2]
    if w == width:
        return frame
    scale = width / float(w)
    return cv2.resize(frame, (width, int(h * scale)), interpolation=cv2.INTER_AREA)


def mirror(frame):
    """Flip horizontally so the webcam behaves like a mirror (natural to watch)."""
    return cv2.flip(frame, 1)


def to_gray(frame):
    """BGR -> single-channel grayscale (Haar cascades and LBF work on gray)."""
    return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)


def mean_brightness(gray):
    """Average pixel intensity (0-255). Used to detect dark rooms."""
    return float(np.mean(gray))


def auto_gamma(frame, brightness, target=config.TARGET_BRIGHTNESS):
    """
    Brighten a dark frame with gamma correction using a lookup table (LUT).

    gamma is chosen so that the *mean* brightness moves towards `target`:
        (brightness/255) ** gamma  =  target/255   ->   gamma = log(t) / log(b)
    A LUT makes this a single fast cv2.LUT call instead of a per-pixel loop.
    """
    if brightness >= config.LOW_LIGHT_BRIGHTNESS or brightness < 1:
        return frame, 1.0
    gamma = np.log(target / 255.0) / np.log(max(brightness, 1.0) / 255.0)
    gamma = float(np.clip(gamma, 0.3, 1.0))   # <1 brightens
    table = np.array([((i / 255.0) ** gamma) * 255 for i in range(256)], dtype=np.uint8)
    return cv2.LUT(frame, table), gamma


_CLAHE = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))


def enhance_contrast(gray):
    """
    CLAHE (Contrast Limited Adaptive Histogram Equalisation).
    Unlike plain equalizeHist it works tile by tile, so one bright window does
    not wash out the face. This makes Haar / landmark detection far more
    stable under uneven lighting.
    """
    return _CLAHE.apply(gray)


def blur(img, ksize=5):
    """Gaussian blur - removes sensor noise before background subtraction."""
    return cv2.GaussianBlur(img, (ksize, ksize), 0)


def histogram_image(gray, size=(280, 56), color=(200, 200, 200)):
    """
    Draw the grayscale histogram (cv2.calcHist) as a small bar chart.
    Part of image *analysis*: shows the lighting distribution of the scene.
    """
    w, h = size
    hist = cv2.calcHist([gray], [0], None, [64], [0, 256]).flatten()
    hist = hist / (hist.max() + 1e-6)
    canvas = np.zeros((h, w, 3), np.uint8)
    bar_w = w / len(hist)
    for i, v in enumerate(hist):
        x0, x1 = int(i * bar_w), int((i + 1) * bar_w) - 1
        cv2.rectangle(canvas, (x0, h - 1), (x1, h - 1 - int(v * (h - 4))), color, -1)
    return canvas
