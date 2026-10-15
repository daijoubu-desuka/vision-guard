"""
VisionGuard - Drowsiness + Mood Detection and Motion-Based People Counter
CSC 126 Computer Vision - Midterm Project (Python + OpenCV)

Three modes in one real-time app (switch with keys 1 / 2 / 3 or the toolbar buttons):
    1  Drowsiness + Mood    Haar cascade + 68 facial landmarks + EAR/MAR + mood rules  (a, c, d, f)
    2  People counter       MOG2 motion + Haar faces + tracking; beeps if more than 1   (a, b, c, d, e)
    3  Object tracker       HSV back-projection + CamShift                             (a, b, e)

Window: resizable / maximizable, M = mini floating view, F = fullscreen.
Closing the window (X), pressing Q / Esc or clicking EXIT ends the program immediately.

Run examples
    python main.py                          # live webcam, drowsiness + mood mode
    python main.py --mode people            # live webcam, people counter (beeps when > 1 person)
    python main.py --video samples/people_demo.mp4 --mode people
    python main.py --image samples/face_sample.jpg
"""
import argparse
import os
import sys
import time
import traceback
from collections import deque
from datetime import datetime

import cv2
import numpy as np

import config
import hud
import preprocessing as pre
import utils
from drowsiness import DrowsinessDetector
from mood import MOODS, MoodDetector
from motion import CentroidTracker, LineCounter, MotionDetector, PeopleCounter
from object_tracker import CamShiftTracker

WIN = config.WINDOW_NAME
MODE_NAMES = {1: "DROWSINESS + MOOD", 2: "PEOPLE COUNTER", 3: "OBJECT TRACKER"}
MODE_KEYS = {"drowsiness": 1, "people": 2, "security": 2, "tracker": 3}   # "security" = old name of mode 2
TOOLBAR = [("1 Drowsy+Mood", "mode1"), ("2 People", "mode2"), ("3 Tracker", "mode3"),
           ("Mini (M)", "mini"), ("Fullscreen (F)", "fullscreen"), ("Help (H)", "help"), ("EXIT", "exit")]


class VisionGuardApp:
    def __init__(self, args):
        self.args = args
        self.mode = MODE_KEYS[args.mode]
        self.is_camera = args.video is None and args.image is None
        self.logger = utils.EventLogger()
        self.alarm = utils.Alarm(enabled=not args.no_alarm)
        self.fps_counter = utils.FPSCounter()
        self.writer = None
        self.session_start = time.time()

        # detectors are created lazily so modes you never use cost nothing
        self.drowsy = None
        self.mood = None
        self.motion = None
        self.people = None
        self.tracker_ids = None
        self.counter = None
        self.camshift = CamShiftTracker()

        # rolling histories for the dashboard graphs
        self.ear_hist = deque(maxlen=140)
        self.people_hist = deque(maxlen=140)
        self.conf_hist = deque(maxlen=140)

        # people-counter state
        self.prev_count = 0
        self.last_snapshot_time = -1e9
        self.alerts = 0               # times the count rose to 2 or more
        self.pending_roi = tuple(int(v) for v in args.roi.split(",")) if args.roi else None

        # window / GUI state
        self.running = True
        self.view = "full"                # "full" or "mini"
        self.fullscreen = False
        self.show_help = False
        self.debug = False
        self.window_ready = False
        self.shown_once = False
        self.full_size = (940, 516)
        self.button_rects = []            # clickable areas of the current image
        self.actions = []                 # clicks waiting to be handled
        self.last_comp = None             # dashboard without toolbar
        self.last_video = None            # annotated video only (used by mini view)
        self._reported_errors = set()

    # ------------------------------------------------------------------ setup
    def _drowsy(self):
        if self.drowsy is None:
            self.drowsy = DrowsinessDetector(utils.ensure_landmark_model())
            self.mood = MoodDetector()
        return self.drowsy

    def _motion(self, frame_h):
        if self.motion is None:
            self.motion = MotionDetector(detect_faces=False)   # faces are searched by the PeopleCounter
            self.people = PeopleCounter()
            self.tracker_ids = CentroidTracker()
            self.counter = LineCounter(frame_h)
        return self.motion

    def open_source(self):
        a = self.args
        if a.image:
            img = cv2.imread(a.image)
            if img is None:
                sys.exit(f"Cannot read image: {a.image}")
            return None, img
        if a.video:
            cap = cv2.VideoCapture(a.video)
            if not cap.isOpened():
                sys.exit(f"Cannot open video: {a.video}")
            return cap, None
        cap = cv2.VideoCapture(a.camera, cv2.CAP_DSHOW) if sys.platform.startswith("win") \
            else cv2.VideoCapture(a.camera)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        if not cap.isOpened():
            sys.exit("Cannot open the webcam. Try --camera 1, close other apps using it, "
                     "or use --video / --image.")
        return cap, None

    # ------------------------------------------------------------ mode 1
    def process_drowsiness(self, frame, now):
        det = self._drowsy()
        brightness = pre.mean_brightness(pre.to_gray(frame))
        frame, gamma = pre.auto_gamma(frame, brightness)         # (a) fix dark rooms
        gray = pre.to_gray(frame)
        gray_eq = pre.enhance_contrast(gray)                      # (a) CLAHE
        res = det.analyse(frame, gray_eq, now)                    # (c) Haar + (f) landmarks
        mood = self.mood.update(res, gray_eq)                     # mood from landmark geometry

        vis = frame.copy()
        hud.draw_drowsiness(vis, res)
        hud.drowsiness_status(vis, res)
        hud.draw_mood_badge(vis, mood)
        if self.debug and mood.deltas:
            self._draw_debug(vis, mood)

        if res.face is not None:
            self.ear_hist.append(res.ear)
        for ev in res.events:
            self.logger.log("drowsiness", ev, f"EAR={res.ear:.2f} MAR={res.mar:.2f}")
            if ev == "DROWSY":
                utils.save_snapshot(vis, "drowsy")
        if mood.changed:
            self.logger.log("mood", "MOOD_CHANGE", mood.mood)
        if res.state == "DROWSY":
            self.alarm.beep()

        thr_txt = f"{res.ear_threshold:.2f}" + ("" if res.calibrated else " (cal..)")
        rows = [
            ("Haar face / eyes", f"{'yes' if res.face else 'no'} / {len(res.eyes)}",
             hud.GREEN if res.face else hud.RED),
            ("EAR / thr", f"{res.ear:.2f} / {thr_txt}", hud.RED if res.eyes_closed else hud.GREEN),
            ("MAR / roll", f"{res.mar:.2f} / {res.roll:+.0f} deg", hud.ORANGE if res.yawning else None),
            ("Eyes closed", f"{res.closed_duration:.1f} s", hud.RED if res.closed_duration > 0.5 else None),
            ("PERCLOS 30s", f"{res.perclos * 100:.0f} %", None),
            ("Blinks / Yawns", f"{res.blinks} / {res.yawns}", None),
            ("Brightness", f"{brightness:.0f}" + (f" (g={gamma:.2f})" if gamma < 1 else ""), None),
            ("MOOD", mood.mood, hud.MOOD_COLOURS.get(mood.mood)),
            ("STATE", res.state, hud.STATE_COLOURS.get(res.state)),
        ]
        thumbs = [("Aligned face (affine)", res.aligned_face), ("CLAHE gray", gray_eq)]
        graph = hud.plot_series(self.ear_hist, threshold=res.ear_threshold, vmax=0.5,
                                label="EAR over time (red = threshold)")
        mood_img = hud.mood_panel_image(self.mood.percentages(), self.mood.timeline)
        self.last_video = vis
        return hud.build_dashboard(vis, MODE_NAMES[1], rows, thumbs, graph, mood_img)

    @staticmethod
    def _draw_debug(vis, mood):
        """Live numbers behind the mood decision (key D) - helps tune config.py."""
        d = mood.deltas
        lines = [f"lift {d['lift']:+.3f}  (smile > {config.MOOD_SMILE_LIFT})",
                 f"width x{d['width']:.2f}  brow x{d['brow']:.2f}",
                 f"furrow x{d['furrow']:.2f}  dMAR {d['mar']:+.2f}  EAR x{d['ear']:.2f}",
                 f"raw={mood.raw}  haar_smile={mood.haar_smile}"]
        for i, t in enumerate(lines):
            hud.put_text(vis, t, (8, vis.shape[0] - 70 + i * 16), 0.42, hud.YELLOW, 1, bg=(0, 0, 0))

    # ------------------------------------------------------------ mode 2
    def process_people(self, frame, now):
        """People counter: motion finds moving people, Haar faces find the ones who stand still."""
        motion = self._motion(frame.shape[0])
        gray = pre.to_gray(frame)
        blurred = pre.blur(frame, 5)                               # (a) denoise
        result = motion.process(blurred, pre.blur(gray, 5))        # (b) segmentation + (d) contours
        objects = self.tracker_ids.update(result.detections)       # (e) tracking
        pres = self.people.update(objects, pre.enhance_contrast(gray), frame.shape[0])   # (c) Haar + count
        crossing_events = self.counter.update(objects)

        count = pres.count
        vis = frame.copy()
        hud.draw_people(vis, objects, pres, self.counter)
        if pres.multiple:                                           # MORE THAN ONE PERSON -> alarm
            hud.red_flash(vis, 0.12)
            hud.banner(vis, f"!! {count} PEOPLE DETECTED !!", hud.RED, "more than 1 person")
            self.alarm.beep(config.PEOPLE_BEEP_INTERVAL, freq=1000, dur=250)
        elif result.warming_up and count == 0:
            hud.banner(vis, "LEARNING BACKGROUND - stay out of view", (120, 90, 40),
                       f"{motion.frames_seen}/{config.MOTION_WARMUP_FRAMES}")
        elif result.global_change and count == 0:
            hud.banner(vis, "LIGHTING / CAMERA CHANGE - adapting...", (0, 120, 200),
                       f"{result.motion_percent:.0f}% of frame changed")
        elif count == 1:
            hud.banner(vis, "1 PERSON IN VIEW", (60, 140, 60), "no alarm")
        else:
            hud.banner(vis, "NO ONE IN VIEW", (90, 90, 90), "counting...")

        if count != self.prev_count:                                # log every change of the count
            self.logger.log("people", "COUNT", f"{self.prev_count}->{count}")
            if count >= config.PEOPLE_ALERT_COUNT > self.prev_count:
                self.alerts += 1
                path = ""
                if now - self.last_snapshot_time >= config.SNAPSHOT_INTERVAL:
                    path = utils.save_snapshot(vis, "multiple_people")
                    self.last_snapshot_time = now
                self.logger.log("people", "MULTIPLE_PEOPLE", f"count={count} snapshot={path}")
        self.prev_count = count
        for ev in crossing_events:
            self.logger.log("people", "LINE_CROSS", ev)

        self.people_hist.append(count)
        active = [o for o in objects.values() if o.disappeared == 0 and o.confirmed]
        small = cv2.resize(frame, (160, 120), interpolation=cv2.INTER_AREA)       # the thumbnail is tiny, so mask
        small_mask = cv2.resize(result.mask, (160, 120), interpolation=cv2.INTER_NEAREST)   # a small copy (fast)
        segmented = cv2.bitwise_and(small, small, mask=small_mask)   # foreground only
        rows = [
            ("PEOPLE IN VIEW", count, hud.RED if pres.multiple else (hud.GREEN if count else None)),
            ("Peak (session)", pres.peak, None),
            ("Faces (Haar)", f"{len(pres.faces)}  ({len(pres.standalone_faces)} still)", None),
            ("Moving objects", len(active), None),
            ("IN / OUT", f"{self.counter.count_in} / {self.counter.count_out}", None),
            ("Motion area", f"{result.motion_percent:.1f} %", None),
            ("Alarms (2+)", self.alerts, hud.RED if self.alerts else None),
            ("STATE", "ALERT" if pres.multiple else ("LEARNING" if result.warming_up else "COUNTING"),
             hud.RED if pres.multiple else hud.GREEN),
        ]
        thumbs = [("Foreground (MOG2)", result.mask), ("Segmented", segmented)]
        graph = hud.plot_series(self.people_hist, vmax=max(3, pres.peak + 1),
                                threshold=config.PEOPLE_ALERT_COUNT - 0.5,
                                color=hud.CYAN, label="people in view (red = alarm level)")
        self.last_video = vis
        return hud.build_dashboard(vis, MODE_NAMES[2], rows, thumbs, graph)

    # ------------------------------------------------------------ mode 3
    def process_tracker(self, frame, now):
        vis = frame.copy()
        cs = self.camshift
        if self.pending_roi and not cs.initialised:
            cs.init(frame, self.pending_roi)
        back = None
        if not cs.initialised:
            hud.banner(vis, "Press S and drag a box around an object", (120, 90, 40))
        else:
            rot, back = cs.update(frame)                           # (b) back-projection + (e) CamShift
            self.conf_hist.append(cs.confidence)
            if rot is None:
                hud.banner(vis, "OBJECT LOST - press S to re-select", hud.RED)
            else:
                box = cv2.boxPoints(rot).astype(np.int32)
                cv2.polylines(vis, [box], True, hud.GREEN, 3, cv2.LINE_AA)
                cx, cy = int(rot[0][0]), int(rot[0][1])
                cv2.circle(vis, (cx, cy), 5, hud.RED, -1)
                hud.put_text(vis, f"TRACKING  conf {cs.confidence:.2f}",
                             (box[:, 0].min(), max(box[:, 1].min() - 8, 14)), 0.55, hud.WHITE, 1, bg=(0, 120, 0))
                hud.banner(vis, "TRACKING OBJECT", (60, 140, 60), f"angle {rot[2]:.0f} deg")
        rows = [
            ("Initialised", "yes" if cs.initialised else "no", None),
            ("Lost", "yes" if cs.lost else "no", hud.RED if cs.lost else hud.GREEN),
            ("Confidence", f"{cs.confidence:.2f}", None),
            ("Window", str(tuple(cs.window)) if cs.window else "-", None),
        ]
        thumbs = [("Back-projection", back), ("Selected object", cs.template)]
        graph = hud.plot_series(self.conf_hist, vmax=1.0, label="tracking confidence")
        extra = cs.hue_histogram_image(cs.hist)
        self.last_video = vis
        return hud.build_dashboard(vis, MODE_NAMES[3], rows, thumbs, graph, extra)

    # --------------------------------------------------------------- dispatch
    def process(self, frame, now):
        """Process one frame. Returns (image_to_display, clean_frame)."""
        frame = pre.resize_to_width(frame)                          # (a) resize
        if self.is_camera:
            frame = pre.mirror(frame)                               # (a) flip
        try:
            if self.mode == 1:
                comp = self.process_drowsiness(frame, now)
            elif self.mode == 2:
                comp = self.process_people(frame, now)
            else:
                comp = self.process_tracker(frame, now)
        except Exception as exc:                  # a failing mode must never close the app
            comp = self.error_screen(frame, exc)
        fps = self.fps_counter.tick()
        hud.put_text(comp, f"FPS {fps:4.1f}", (80, comp.shape[0] - 10), 0.5, hud.YELLOW, 1, bg=(0, 0, 0))
        hud.put_text(comp, "H = help", (8, comp.shape[0] - 10), 0.45, hud.WHITE, 1, bg=(0, 0, 0))
        if self.show_help:
            hud.help_overlay(comp)
        self.last_comp = comp
        return self.render(comp), frame

    def error_screen(self, frame, exc):
        """Show the error on screen (and print the traceback once) instead of crashing."""
        key = (self.mode, type(exc).__name__, str(exc))
        if key not in self._reported_errors:
            self._reported_errors.add(key)
            print(f"\n[ERROR] {MODE_NAMES[self.mode]} failed - the app keeps running. Details:")
            traceback.print_exc()
        vis = frame.copy()
        hud.banner(vis, f"{MODE_NAMES[self.mode]} ERROR - see terminal", hud.RED)
        hud.put_text(vis, f"{type(exc).__name__}: {str(exc)[:70]}", (10, 70), 0.5, hud.WHITE, 1, bg=hud.RED)
        hud.put_text(vis, "Press 1 / 2 / 3 to try another mode", (10, 95), 0.5, hud.YELLOW, 1, bg=(0, 0, 0))
        self.last_video = vis
        return hud.build_dashboard(vis, "ERROR", [("Mode", MODE_NAMES[self.mode], hud.RED),
                                                  ("Error", type(exc).__name__, hud.RED)], [])

    # ------------------------------------------------------------ rendering
    def render(self, comp):
        """Turn the dashboard into what is shown: full view (+toolbar) or mini view."""
        if self.view == "mini" and self.last_video is not None:
            v = self.last_video
            mw = config.MINI_WIDTH
            mh = int(v.shape[0] * mw / v.shape[1])
            mini = cv2.resize(v, (mw, mh), interpolation=cv2.INTER_AREA)
            self.button_rects = hud.small_buttons(mini, [("EXPAND", "full"), ("EXIT", "exit")])
            return mini
        active = {1: "mode1", 2: "mode2", 3: "mode3"}[self.mode]
        bar, rects = hud.toolbar(comp.shape[1], TOOLBAR, active)
        y0 = comp.shape[0]
        self.button_rects = [(x0, y0 + a, x1, y0 + b, act) for x0, a, x1, b, act in rects]
        self.full_size = (comp.shape[1], comp.shape[0] + bar.shape[0])
        return np.vstack([comp, bar])

    def on_mouse(self, event, x, y, flags, param):
        """Clicks on toolbar / mini-view buttons are queued and handled in the main loop."""
        if event == cv2.EVENT_LBUTTONDOWN:
            for x0, y0, x1, y1, action in self.button_rects:
                if x0 <= x <= x1 and y0 <= y <= y1:
                    self.actions.append(action)
                    break

    def set_view(self, view):
        """Switch between the full dashboard and the small always-on-top mini window."""
        if self.fullscreen:
            self.toggle_fullscreen()
        self.view = view
        if self.last_comp is not None:
            img = self.render(self.last_comp)
            try:
                cv2.resizeWindow(WIN, img.shape[1], img.shape[0])
                cv2.setWindowProperty(WIN, cv2.WND_PROP_TOPMOST, 1 if view == "mini" else 0)
            except cv2.error:
                pass          # some OpenCV builds do not support these properties
        print(f"[view] {view}")

    def toggle_fullscreen(self):
        if self.view == "mini":
            self.set_view("full")
        self.fullscreen = not self.fullscreen
        try:
            cv2.setWindowProperty(WIN, cv2.WND_PROP_FULLSCREEN,
                                  cv2.WINDOW_FULLSCREEN if self.fullscreen else cv2.WINDOW_NORMAL)
            if not self.fullscreen:
                cv2.resizeWindow(WIN, *self.full_size)
        except cv2.error:
            pass

    def select_roi(self, clean_frame):
        """Let the user draw a box around the object to track (mode 3)."""
        roi = cv2.selectROI("Select object - ENTER to confirm", clean_frame, False)
        cv2.destroyWindow("Select object - ENTER to confirm")
        if roi[2] > 0 and roi[3] > 0:
            self.camshift.init(clean_frame, roi)

    def write_output(self, img):
        if not self.args.output or img is None:
            return
        if self.writer is None:
            os.makedirs(os.path.dirname(self.args.output) or ".", exist_ok=True)
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            self.writer = cv2.VideoWriter(self.args.output, fourcc, 20.0, (img.shape[1], img.shape[0]))
        self.writer.write(img)

    def switch_mode(self, new_mode):
        if new_mode != self.mode:
            self.mode = new_mode
            print(f"[mode] {MODE_NAMES[new_mode]}")

    def window_closed(self):
        """True when the user pressed the window's X button."""
        if not self.shown_once:
            return False
        try:
            return cv2.getWindowProperty(WIN, cv2.WND_PROP_VISIBLE) < 1
        except cv2.error:
            return True

    # ---------------------------------------------------------------- input
    def do_action(self, action, clean=None):
        if action in ("mode1", "mode2", "mode3"):
            self.switch_mode(int(action[-1]))
        elif action == "mini":
            self.set_view("mini")
        elif action == "full":
            self.set_view("full")
        elif action == "fullscreen":
            self.toggle_fullscreen()
        elif action == "help":
            self.show_help = not self.show_help
        elif action == "exit":
            self.running = False

    def handle_key(self, key, clean):
        if key in (ord("q"), 27):
            self.running = False
        elif key in (ord("1"), ord("2"), ord("3")):
            self.switch_mode(int(chr(key)))
        elif key == ord("s") and self.mode == 3 and clean is not None:
            self.select_roi(clean)
        elif key == ord("m"):
            self.set_view("full" if self.view == "mini" else "mini")
        elif key == ord("f"):
            self.toggle_fullscreen()
        elif key == ord("d"):
            self.debug = not self.debug
        elif key == ord("c"):
            if self.drowsy:
                self.drowsy.reset()
                self.mood.reset()
            if self.motion:
                self.motion.reset()
                self.tracker_ids.reset()
                self.people.reset()
                self.prev_count = 0
            print("[reset] recalibrating - show a NEUTRAL face / empty room")
        elif key == ord("r"):
            if self.counter:
                self.counter.reset()
                self.people.peak = 0
            self.alerts = 0
            if self.drowsy:
                self.drowsy.blinks = self.drowsy.yawns = 0
        elif key == ord("p") and self.last_comp is not None:
            print("[snapshot]", utils.save_snapshot(self.last_comp, "manual"))
        elif key == ord("h"):
            self.show_help = not self.show_help

    def show(self, img):
        """Display one image and process keyboard / mouse / window-close events."""
        if not self.window_ready:
            cv2.namedWindow(WIN, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)   # resizable + maximizable
            cv2.setMouseCallback(WIN, self.on_mouse)
            self.window_ready = True
        cv2.imshow(WIN, img)
        if not self.shown_once:
            self.shown_once = True
            cv2.resizeWindow(WIN, img.shape[1], img.shape[0])
        return cv2.waitKey(1) & 0xFF

    # -------------------------------------------------------------------- run
    def run(self):
        cap, still = self.open_source()
        show = not self.args.no_display
        try:
            if still is not None:
                self.run_still(still, show)
            else:
                self.run_stream(cap, show)
        finally:
            self.shutdown(cap, show)

    def run_still(self, still, show):
        self.mode = 1
        img, _ = self.process(still, 0.0)
        dest = self.args.output or os.path.join(config.OUTPUT_DIR, "image_result.jpg")
        os.makedirs(os.path.dirname(dest) or ".", exist_ok=True)
        cv2.imwrite(dest, img)
        print(f"[done] saved {dest}")
        while show and self.running:
            key = self.show(self.render(self.last_comp))
            while self.actions:
                self.do_action(self.actions.pop(0))
            if key in (ord("q"), 27) or self.window_closed():
                break
            if key != 255:
                self.handle_key(key, None)
                self.mode = 1

    def run_stream(self, cap, show):
        fps_src = cap.get(cv2.CAP_PROP_FPS) or 30.0
        idx, t0, paused = 0, time.time(), False
        out, clean = None, None
        print("[ready] 1/2/3 = modes, M = mini view, F = fullscreen, H = help, Q = quit")
        while self.running:
            if not paused:
                ok, frame = cap.read()
                if not ok:
                    break
                # webcams use wall-clock time, video files use their own timeline
                now = (time.time() - t0) if self.is_camera else idx / fps_src
                out, clean = self.process(frame, now)
                self.write_output(self.last_comp)
                idx += 1
                if self.args.max_frames and idx >= self.args.max_frames:
                    break
            elif self.last_comp is not None:
                out = self.render(self.last_comp)
            if show:
                key = self.show(out)
                if self.window_closed():              # user clicked the X button -> quit
                    break
                while self.actions:
                    self.do_action(self.actions.pop(0), clean)
                if key == ord(" "):
                    paused = not paused
                elif key != 255:
                    self.handle_key(key, clean)

    # --------------------------------------------------------------- shutdown
    def session_summary(self):
        """Printed and saved when the program ends."""
        secs = time.time() - self.session_start
        lines = [f"VisionGuard session summary  ({datetime.now():%Y-%m-%d %H:%M:%S})",
                 f"Duration: {secs / 60:.1f} min"]
        if self.drowsy is not None:
            lines.append(f"Blinks: {self.drowsy.blinks}   Yawns: {self.drowsy.yawns}")
        if self.mood is not None and sum(self.mood.counts.values()) > 0:
            lines.append("Mood mix: " + ", ".join(f"{m} {p:.0f}%" for m, p in
                                                  self.mood.percentages().items() if p > 0))
        if self.counter is not None:
            lines.append(f"People entered: {self.counter.count_in}  exited: {self.counter.count_out}  "
                         f"peak in view: {self.people.peak}  multi-person alerts: {self.alerts}")
        return "\n".join(lines)

    def shutdown(self, cap, show):
        """Release everything so the program really ends when the window is closed."""
        try:
            summary = self.session_summary()
            print("\n" + summary)
            os.makedirs(config.OUTPUT_DIR, exist_ok=True)
            with open(os.path.join(config.OUTPUT_DIR, "session_summary.txt"), "w", encoding="utf-8") as f:
                f.write(summary + "\n")
        except Exception:
            pass
        if cap is not None:
            cap.release()
        if self.writer:
            self.writer.release()
            print(f"[done] saved {self.args.output}")
        self.logger.close()
        if show:
            cv2.destroyAllWindows()
            for _ in range(5):          # let the GUI process the close events
                cv2.waitKey(1)


def parse_args():
    p = argparse.ArgumentParser(description="VisionGuard - drowsiness, mood + people counter (OpenCV)")
    src = p.add_mutually_exclusive_group()
    src.add_argument("--video", help="path to a recorded video file")
    src.add_argument("--image", help="path to a static image")
    p.add_argument("--camera", type=int, default=0, help="webcam index (default 0)")
    p.add_argument("--mode", choices=list(MODE_KEYS), default="drowsiness",
                   help="drowsiness | people (alias: security) | tracker")
    p.add_argument("--roi", help="x,y,w,h of the object to track (mode 3, for videos)")
    p.add_argument("--output", help="save the processed result (video .mp4 or image .jpg)")
    p.add_argument("--no-display", action="store_true", help="run without opening a window")
    p.add_argument("--no-alarm", action="store_true", help="disable the beep")
    p.add_argument("--max-frames", type=int, default=0, help="stop after N frames (testing)")
    return p.parse_args()


if __name__ == "__main__":
    exit_code = 0
    try:
        VisionGuardApp(parse_args()).run()
    except KeyboardInterrupt:
        pass
    except SystemExit as e:
        exit_code = e.code if isinstance(e.code, int) else 1
        if isinstance(e.code, str):
            print(e.code)
    except Exception:
        traceback.print_exc()
        exit_code = 1
    finally:
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(exit_code)      # terminate immediately: no lingering threads / debugger session
