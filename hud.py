"""
hud.py - Visualisation: bounding boxes, landmarks, overlays, masks, graphs and
the side dashboard. Nothing in here makes decisions; it only draws results.
"""
from collections import deque

import cv2
import numpy as np

import config
import drowsiness as dz

GREEN, RED, YELLOW = (80, 220, 80), (60, 60, 255), (0, 220, 255)
CYAN, WHITE, ORANGE, GREY = (255, 220, 60), (255, 255, 255), (0, 140, 255), (150, 150, 150)

MOOD_COLOURS = {"HAPPY": (0, 215, 255), "NEUTRAL": (200, 200, 200), "SAD": (255, 150, 60),
                "SURPRISED": (255, 90, 220), "ANGRY": (70, 70, 255), "TIRED": (190, 120, 160),
                "CALIBRATING": (150, 150, 150), "NO FACE": (110, 110, 110)}

STATE_COLOURS = {"AWAKE": GREEN, "WARNING": ORANGE, "DROWSY": RED,
                 "SEARCHING": GREY, "AWAY": ORANGE}


# ----------------------------------------------------------------------------
# small drawing helpers
# ----------------------------------------------------------------------------
def put_text(img, text, org, scale=0.5, color=WHITE, thickness=1, bg=None):
    """Text with an optional filled background so it stays readable on video."""
    (tw, th), base = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)
    x, y = org
    if bg is not None:
        cv2.rectangle(img, (x - 3, y - th - 4), (x + tw + 3, y + base), bg, -1)
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)


def banner(img, text, color, sub=None):
    """Big coloured banner across the top of the video (status / alarm)."""
    overlay = img.copy()
    cv2.rectangle(overlay, (0, 0), (img.shape[1], 44), color, -1)
    cv2.addWeighted(overlay, 0.65, img, 0.35, 0, img)
    put_text(img, text, (12, 29), 0.85, WHITE, 2)
    if sub:
        (tw, _), _ = cv2.getTextSize(sub, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        put_text(img, sub, (img.shape[1] - tw - 12, 28), 0.5, WHITE, 1)


def red_flash(img, strength=0.25):
    """Tint the whole frame red (used while the drowsiness alarm is active)."""
    red = np.zeros_like(img)
    red[:] = (0, 0, 255)
    cv2.addWeighted(red, strength, img, 1 - strength, 0, img)


def plot_series(values, size=(280, 70), threshold=None, vmax=None, color=GREEN, label=""):
    """Tiny line chart of the most recent values (EAR, motion %, confidence...)."""
    w, h = size
    canvas = np.full((h, w, 3), 28, np.uint8)
    cv2.rectangle(canvas, (0, 0), (w - 1, h - 1), (70, 70, 70), 1)
    vals = list(values)
    if len(vals) >= 2:
        top = vmax if vmax else max(max(vals) * 1.2, 1e-3)
        pts = [(int(i * (w - 1) / (len(vals) - 1)),
                int((h - 4) - np.clip(v / top, 0, 1) * (h - 8))) for i, v in enumerate(vals)]
        cv2.polylines(canvas, [np.array(pts, np.int32)], False, color, 1, cv2.LINE_AA)
        if threshold is not None:
            ty = int((h - 4) - np.clip(threshold / top, 0, 1) * (h - 8))
            cv2.line(canvas, (0, ty), (w - 1, ty), RED, 1, cv2.LINE_AA)
    put_text(canvas, label, (5, 13), 0.38, GREY)
    return canvas


def thumb(img, title, size=(136, 102)):
    """Resize an image to a labelled thumbnail (works for gray or colour)."""
    if img is None:
        img = np.zeros((size[1], size[0], 3), np.uint8)
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    t = cv2.resize(img, size, interpolation=cv2.INTER_AREA)
    cv2.rectangle(t, (0, 0), (size[0] - 1, size[1] - 1), (90, 90, 90), 1)
    put_text(t, title, (4, 13), 0.38, YELLOW, 1, bg=(0, 0, 0))
    return t


# ----------------------------------------------------------------------------
# Mode 1: drowsiness overlays
# ----------------------------------------------------------------------------
def draw_drowsiness(frame, res):
    """Face box, Haar eye boxes, landmarks, eye/mouth contours and EAR label."""
    colour = STATE_COLOURS.get(res.state, GREEN)
    if res.face is not None:
        x, y, w, h = res.face
        cv2.rectangle(frame, (x, y), (x + w, y + h), colour, 2)
        put_text(frame, "Face (Haar)", (x, y - 6), 0.45, WHITE, 1, bg=(0, 0, 0))
    for (ex, ey, ew, eh) in res.eyes:                       # Haar eyes
        cv2.rectangle(frame, (ex, ey), (ex + ew, ey + eh), YELLOW, 1)
    if res.landmarks is not None:
        pts = res.landmarks.astype(np.int32)
        for (px, py) in pts:                                # all 68 landmarks
            cv2.circle(frame, (int(px), int(py)), 1, CYAN, -1, cv2.LINE_AA)
        eye_col = RED if res.eyes_closed else GREEN
        for idx in (dz.RIGHT_EYE, dz.LEFT_EYE):             # convex hull = eye contour
            cv2.polylines(frame, [cv2.convexHull(pts[idx])], True, eye_col, 1, cv2.LINE_AA)
        mouth_col = ORANGE if res.yawning else GREEN
        cv2.polylines(frame, [cv2.convexHull(pts[dz.OUTER_MOUTH])], True, mouth_col, 1, cv2.LINE_AA)
        cv2.polylines(frame, [pts[dz.JAW].reshape(-1, 1, 2)], False, (200, 200, 200), 1, cv2.LINE_AA)
        x, y, w, h = res.face
        put_text(frame, f"EAR {res.ear:.2f}", (x, y + h + 16), 0.5, eye_col, 1, bg=(0, 0, 0))
        put_text(frame, f"MAR {res.mar:.2f}", (x + 90, y + h + 16), 0.5, mouth_col, 1, bg=(0, 0, 0))


def drowsiness_status(frame, res):
    """Top banner + red flash depending on the drowsiness state."""
    if res.state == "DROWSY":
        red_flash(frame)
        banner(frame, "!! DROWSINESS ALERT - WAKE UP !!", RED, f"eyes closed {res.closed_duration:.1f}s")
    elif res.state == "WARNING":
        reason = "yawning" if res.yawning else f"PERCLOS {res.perclos*100:.0f}%"
        banner(frame, "WARNING - signs of fatigue", ORANGE, reason)
    elif res.state == "AWAKE":
        sub = "calibrating..." if not res.calibrated else "driver alert"
        banner(frame, "AWAKE", (60, 140, 60), sub)
    elif res.state == "AWAY":
        banner(frame, "NO FACE - user away", (90, 90, 90))
    else:
        banner(frame, "Searching for face...", (90, 90, 90))


# ----------------------------------------------------------------------------
# Mood overlays
# ----------------------------------------------------------------------------
def draw_mood_icon(img, centre, r, mood):
    """Draw a simple cartoon face that matches the detected mood (pure OpenCV shapes)."""
    cx, cy = centre
    dark = (30, 30, 30)
    cv2.circle(img, centre, r, MOOD_COLOURS.get(mood, GREY), -1, cv2.LINE_AA)
    cv2.circle(img, centre, r, dark, 2, cv2.LINE_AA)
    ex, ey, er = int(r * 0.38), int(r * 0.22), max(2, int(r * 0.11))
    lx, rx, y = cx - ex, cx + ex, cy - ey
    mw = int(r * 0.5)
    # eyes
    if mood == "TIRED":
        for x in (lx, rx):
            cv2.line(img, (x - er * 2, y), (x + er * 2, y), dark, 2, cv2.LINE_AA)
    elif mood == "SURPRISED":
        for x in (lx, rx):
            cv2.circle(img, (x, y), er * 2, (255, 255, 255), -1, cv2.LINE_AA)
            cv2.circle(img, (x, y), er, dark, -1, cv2.LINE_AA)
    else:
        for x in (lx, rx):
            cv2.circle(img, (x, y), er, dark, -1, cv2.LINE_AA)
    # eyebrows for angry
    if mood == "ANGRY":
        cv2.line(img, (lx - er * 2, y - er * 3), (lx + er * 2, y - er), dark, 2, cv2.LINE_AA)
        cv2.line(img, (rx + er * 2, y - er * 3), (rx - er * 2, y - er), dark, 2, cv2.LINE_AA)
    # mouth (cv2.ellipse angles are clockwise, y points down: 20..160 = lower arc)
    if mood == "HAPPY":
        cv2.ellipse(img, (cx, cy + int(r * 0.12)), (mw, int(r * 0.5)), 0, 20, 160, dark, 2, cv2.LINE_AA)
    elif mood == "SAD":
        cv2.ellipse(img, (cx, cy + int(r * 0.65)), (mw, int(r * 0.4)), 0, 200, 340, dark, 2, cv2.LINE_AA)
    elif mood == "SURPRISED":
        cv2.circle(img, (cx, cy + int(r * 0.45)), max(2, int(r * 0.17)), dark, -1, cv2.LINE_AA)
    else:
        cv2.line(img, (cx - mw // 2, cy + int(r * 0.45)), (cx + mw // 2, cy + int(r * 0.45)), dark, 2, cv2.LINE_AA)


def draw_mood_badge(frame, mood_res):
    """Mood icon + text in the bottom-right corner of the video."""
    h, w = frame.shape[:2]
    if mood_res.mood == "CALIBRATING":
        put_text(frame, "Mood: keep a NEUTRAL face...", (w - 250, h - 12), 0.5, YELLOW, 1, bg=(0, 0, 0))
        return
    if mood_res.mood == "NO FACE":
        return
    overlay = frame.copy()
    cv2.rectangle(overlay, (w - 190, h - 62), (w - 8, h - 8), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
    draw_mood_icon(frame, (w - 160, h - 35), 20, mood_res.mood)
    put_text(frame, "MOOD", (w - 130, h - 38), 0.4, GREY)
    put_text(frame, mood_res.mood, (w - 130, h - 18), 0.65, MOOD_COLOURS.get(mood_res.mood, WHITE), 2)


def mood_panel_image(percentages, timeline, size=(280, 58)):
    """Top: colour strip of the last frames. Middle: stacked % bar. Bottom: top-3 moods."""
    w, h = size
    canvas = np.full((h, w, 3), 28, np.uint8)
    tl = list(timeline)
    if tl:
        step = w / 140.0
        for i, m in enumerate(tl[-140:]):
            x0 = int(i * step)
            cv2.rectangle(canvas, (x0, 2), (x0 + max(int(step), 1), 14), MOOD_COLOURS.get(m, GREY), -1)
    x = 0
    for m, p in percentages.items():
        seg = int(w * p / 100.0)
        if seg > 0:
            cv2.rectangle(canvas, (x, 20), (x + seg, 34), MOOD_COLOURS.get(m, GREY), -1)
            x += seg
    cv2.rectangle(canvas, (0, 20), (w - 1, 34), (90, 90, 90), 1)
    top = sorted(percentages.items(), key=lambda kv: -kv[1])[:3]
    px = 4
    for m, p in top:
        if p > 0:
            txt = f"{m} {p:.0f}%"
            put_text(canvas, txt, (px, 52), 0.38, MOOD_COLOURS.get(m, WHITE), 1)
            px += 8 * len(txt) + 6
    return canvas


# ----------------------------------------------------------------------------
# Toolbar (clickable buttons) and mini view
# ----------------------------------------------------------------------------
def toolbar(width, buttons, active=None):
    """
    Draw a strip of buttons. buttons = [(label, action), ...].
    Returns (image, rects) where rects = [(x0, y0, x1, y1, action)] in strip coordinates.
    """
    h = config.TOOLBAR_HEIGHT
    bar = np.full((h, width, 3), 35, np.uint8)
    n = len(buttons)
    gap = 6
    bw = (width - gap * (n + 1)) // n
    rects = []
    for i, (label, action) in enumerate(buttons):
        x0 = gap + i * (bw + gap)
        x1 = x0 + bw
        on = action == active
        fill = (0, 120, 200) if on else ((40, 40, 150) if action == "exit" else (75, 75, 75))
        cv2.rectangle(bar, (x0, 5), (x1, h - 5), fill, -1)
        cv2.rectangle(bar, (x0, 5), (x1, h - 5), (150, 150, 150), 1)
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        cv2.putText(bar, label, (x0 + (bw - tw) // 2, (h + th) // 2), cv2.FONT_HERSHEY_SIMPLEX,
                    0.45, WHITE, 1, cv2.LINE_AA)
        rects.append((x0, 0, x1, h, action))
    return bar, rects


def small_buttons(img, buttons):
    """Draw small buttons in the bottom-right of the mini view. Returns hit rectangles."""
    h, w = img.shape[:2]
    rects, x1 = [], w - 6
    for label, action in reversed(buttons):
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.42, 1)
        x0 = x1 - tw - 14
        fill = (40, 40, 150) if action == "exit" else (0, 120, 200)
        cv2.rectangle(img, (x0, h - 30), (x1, h - 8), fill, -1)
        cv2.rectangle(img, (x0, h - 30), (x1, h - 8), WHITE, 1)
        cv2.putText(img, label, (x0 + 7, h - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.42, WHITE, 1, cv2.LINE_AA)
        rects.append((x0, h - 30, x1, h - 8, action))
        x1 = x0 - 6
    return rects


# ----------------------------------------------------------------------------
# Mode 2: people counter overlays
# ----------------------------------------------------------------------------
def draw_people(frame, objects, pres, counter):
    """Mode 2: one box per person (green, red when more than one), Haar faces, trails and the big count."""
    cv2.line(frame, (0, counter.line_y), (frame.shape[1], counter.line_y), YELLOW, 1)
    put_text(frame, "count line (down = IN, up = OUT)", (8, counter.line_y - 8), 0.4, YELLOW, 1, bg=(0, 0, 0))
    colour = RED if pres.multiple else GREEN
    for (fx, fy, fw, fh) in pres.faces:                       # every Haar face
        cv2.rectangle(frame, (fx, fy), (fx + fw, fy + fh), CYAN, 1)
    for (fx, fy, fw, fh) in pres.standalone_faces:            # people who are not moving
        cv2.rectangle(frame, (fx - 8, fy - 8), (fx + fw + 8, fy + fh + 8), colour, 2)
        put_text(frame, "PERSON (still)", (fx - 8, max(fy - 14, 14)), 0.5, WHITE, 1, bg=colour)
    for obj in objects.values():
        if obj.disappeared > 0:
            continue
        x, y, w, h = obj.bbox
        if obj.people > 0:
            cv2.rectangle(frame, (x, y), (x + w, y + h), colour, 2)
            label = f"PERSON #{obj.object_id}" + (f" x{obj.people}" if obj.people > 1 else "")
            put_text(frame, label, (x, max(y - 6, 14)), 0.5, WHITE, 1, bg=colour)
        elif obj.confirmed:                                   # moving, but not a person: not counted
            cv2.rectangle(frame, (x, y), (x + w, y + h), GREY, 1)
            put_text(frame, "motion (not a person)", (x, max(y - 6, 14)), 0.4, WHITE, 1, bg=GREY)
            continue
        _trail_and_arrow(frame, obj)
    put_text(frame, f"PEOPLE: {pres.count}", (10, 78), 0.9, WHITE, 2, bg=colour if pres.count else GREY)


def _trail_and_arrow(frame, obj):
    """Trail of past centroids, centroid dot and a direction-of-travel arrow."""
    trail = np.array(obj.trail, np.int32).reshape(-1, 1, 2)
    if len(trail) > 1:
        cv2.polylines(frame, [trail], False, YELLOW, 2, cv2.LINE_AA)
    cv2.circle(frame, obj.centroid, 4, WHITE, -1)
    speed = float(np.hypot(*obj.velocity))
    if speed > 2:
        tip = (int(obj.centroid[0] + obj.velocity[0] * 4), int(obj.centroid[1] + obj.velocity[1] * 4))
        cv2.arrowedLine(frame, obj.centroid, tip, GREEN, 2, tipLength=0.35)


# ----------------------------------------------------------------------------
# Side dashboard
# ----------------------------------------------------------------------------
def build_dashboard(frame, title, rows, thumbs, graph=None, extra=None):
    """
    Compose [video | panel]. `rows` = list of (label, value, colour),
    `thumbs` = list of (title, image) shown two per row, `graph`/`extra` are
    full-width images appended at the bottom of the panel.
    """
    h, w = frame.shape[:2]
    pw = config.PANEL_WIDTH
    panel = np.full((h, pw, 3), 22, np.uint8)
    put_text(panel, title, (10, 22), 0.6, YELLOW, 1)
    cv2.line(panel, (8, 30), (pw - 8, 30), (80, 80, 80), 1)
    y = 50
    for label, value, colour in rows:
        put_text(panel, f"{label}", (10, y), 0.45, GREY)
        put_text(panel, f"{value}", (130, y), 0.45, colour or WHITE)
        y += 19
    y += 4
    tiles = [thumb(img, t) for t, img in thumbs]
    for i in range(0, len(tiles), 2):
        row = tiles[i:i + 2]
        for j, tile in enumerate(row):
            th, tw = tile.shape[:2]
            if y + th < h:
                panel[y:y + th, 10 + j * (tw + 8): 10 + j * (tw + 8) + tw] = tile
        y += tiles[0].shape[0] + 6
    for extra_img in (graph, extra):
        if extra_img is not None and y + extra_img.shape[0] < h + 40:
            eh, ew = extra_img.shape[:2]
            if y + eh > h:                      # grow the panel if it does not fit
                pad = np.full((y + eh - h + 6, pw, 3), 22, np.uint8)
                panel = np.vstack([panel, pad])
            panel[y:y + eh, 10:10 + ew] = extra_img
            y += eh + 6
    if panel.shape[0] > h:                      # keep video and panel equal height
        frame = np.vstack([frame, np.zeros((panel.shape[0] - h, w, 3), np.uint8)])
    return np.hstack([frame, panel])


def help_overlay(img):
    lines = ["KEYS", "1  Drowsiness + mood", "2  People counter (beeps if > 1)",
             "3  Object tracker (S = select)", "M  Mini / full view", "F  Fullscreen on/off",
             "C  Recalibrate (neutral face!)", "R  Reset counters", "P  Save snapshot",
             "D  Debug mood numbers", "Space  Pause", "H  Toggle this help", "Q / Esc  Quit"]
    overlay = img.copy()
    cv2.rectangle(overlay, (12, 56), (292, 56 + 22 * len(lines) + 10), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.7, img, 0.3, 0, img)
    for i, line in enumerate(lines):
        put_text(img, line, (22, 78 + 22 * i), 0.5, YELLOW if i == 0 else WHITE, 1)
