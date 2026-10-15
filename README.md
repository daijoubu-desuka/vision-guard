# VisionGuard
### Real-Time Drowsiness + Mood Detection and Motion-Based People Counter
**CSC 126 – Computer Vision · Midterm Project · Python + OpenCV (classical techniques only, no deep learning)**

**Members:** Adlawon, Anthony N. · Bantillo, Daniel M. · Abejay, Franz Jedd M.

VisionGuard is a single real-time application with three modes you can switch between
while it is running. It works on a **live webcam feed**, a **recorded video**, or a
**static image**.

| Mode | Real-world problem it solves | Lab topics used |
|------|------------------------------|-----------------|
| **1 – Drowsiness + Mood** | Detects a tired driver / student (eyes closed too long, yawning, high PERCLOS) and sounds an alarm; also estimates the **mood** (happy, neutral, sad, surprised, angry, tired) with a live timeline | **c** Haar Cascade (face, eyes, smile) · **f** Facial Landmarks · **a** Image Operations · **d** Analysis & Transformation |
| **2 – People counter** | Detects people (motion + Haar faces), shows **how many are in view in real time**, keeps counting people who stand still, **beeps when there is more than one**, and saves a snapshot + event log | **b** Segmentation · **e** Motion & Object Tracking · **d** Contour Analysis · **c** Haar (people check) · **a** |
| **3 – Object tracker** | Tracks any object you select by its colour | **b** Segmentation (HSV back-projection) · **e** CamShift Tracking · **a** |

**All six lab topics (a–f) are used** – the rubric requires at least three.

---

## 1. Setup

Requires **Python 3.9+** and a webcam (a webcam is optional – you can use `--video` / `--image`).

```bash
git clone <your-repo-url>
cd visionguard

python -m venv .venv
# Windows:  .venv\Scripts\activate        macOS/Linux:  source .venv/bin/activate

pip install -r requirements.txt
python download_models.py        # one time, ~56 MB facial-landmark model (also automatic on first run)
```

> Use **`opencv-contrib-python`** (already in `requirements.txt`). If you previously installed
> `opencv-python`, run `pip uninstall opencv-python` first – having both causes import errors.

## 2. How to run

```bash
python main.py                                   # live webcam, drowsiness mode
python main.py --mode people                     # live webcam, people counter (beeps when > 1 person)
python main.py --mode tracker                    # live webcam, press S and drag a box on an object

# recorded video / static image (no webcam needed)
python main.py --video samples/drowsy_demo.mp4 --mode drowsiness
python main.py --video samples/walkers.mp4     --mode people      # 2 people on screen together -> alarm
python main.py --video samples/people_demo.mp4 --mode people      # one person stands still, a second joins
python main.py --video samples/color_ball.mp4  --mode tracker --roi 286,306,68,68
python main.py --image samples/face_sample.jpg

# save the processed result
python main.py --video samples/walkers.mp4 --mode people --output output/result.mp4
```

Useful options: `--camera 1` (another webcam), `--no-alarm` (silence), `--no-display`
(headless), `--max-frames N`.

### Keyboard controls

| Key | Action |
|-----|--------|
| `1` `2` `3` | Switch mode (or click the toolbar buttons) |
| `M` | **Mini view** (small always-on-top window) ⇄ full view |
| `F` | Fullscreen on / off |
| `D` | Show the live numbers behind the mood decision (for tuning) |
| `S` | (Mode 3) select the object to track – drag a box, press **ENTER** |
| `C` | Recalibrate (mode 1: **show a neutral face**) / relearn background (mode 2) |
| `R` | Reset counters |
| `P` | Save a screenshot to `output/snapshots/` |
| `Space` | Pause (video / webcam) |
| `H` | Show / hide help |
| `Q` / `Esc` | Quit |

### Window and exit behaviour
* The window is **resizable**: use the normal minimize / maximize buttons or drag the corners.
* The toolbar at the bottom has clickable buttons (modes, **Mini**, **Fullscreen**, Help, **EXIT**). In the mini view you get **EXPAND** and **EXIT**.
* **Closing the window (X), pressing `Q` / `Esc`, or clicking EXIT ends the program immediately** – the camera is released and the
  Python process terminates, so you do not need to press Stop in VS Code. A summary is saved to `output/session_summary.txt`.

### Tips for the best results
* **Mode 1:** sit 40–80 cm from the camera, face the camera, keep a **neutral face with eyes open for the first ~2 seconds**
  (the app learns *your* normal eye opening and neutral mood baseline). Press `C` to recalibrate if you change seats or started with a smile.
* **Mode 2:** stay out of the frame for the first ~2 seconds while the background is learned (banner shows *LEARNING BACKGROUND*). Keep the camera **still**, switch off auto-exposure if possible, and avoid sitting in front of a window with moving light. The big **PEOPLE: n** number is the live count; at **2 or more** the banner turns red and the program beeps (every `PEOPLE_BEEP_INTERVAL` seconds), saves one snapshot and logs `MULTIPLE_PEOPLE`. Face the camera so the face search can see you (faces must be at least `PEOPLE_FACE_MIN_SIZE` px wide). If one person is counted as several, raise `MOTION_MERGE_GAP` in `config.py`; press `C` to relearn the background. Walking **down** across the yellow line = IN, **up** = OUT.
* **Mode 3:** pick an object with a colour that differs from the background (a coloured marker, ball, phone case).

---

## 3. How it works

```
 camera / video / image
        │
        ▼
 preprocessing.py  ── resize · mirror · BGR→GRAY · brightness · gamma LUT · CLAHE · Gaussian blur · histogram   (a, d)
        │
        ├── Mode 1  drowsiness.py + mood.py
        │     Haar face cascade ─► Haar eye cascade ─► 68 LBF landmarks ─► EAR / MAR / head roll
        │     affine face alignment (warpAffine) ─► state machine: AWAKE → WARNING → DROWSY (+ alarm, log, snapshot)
        │     landmark measurements vs. your neutral baseline (+ Haar smile) ─► rules ─► majority vote ─► MOOD
        │
        ├── Mode 2  motion.py
        │     MOG2 background subtraction ─► threshold + morphology ─► contours + moments
        │     CentroidTracker (IDs, trails, direction) ─► LineCounter (IN / OUT)
        │     PeopleCounter: Haar face inside a blob / full-body shape = PERSON (moving people)
        │                    + Haar face search over the whole frame (people who stand still)
        │                    ─► majority vote over 8 frames ─► PEOPLE COUNT ─► beep + snapshot + log if > 1
        │
        └── Mode 3  object_tracker.py
              HSV hue histogram ─► calcBackProject (segmentation) ─► CamShift (tracking)
        │
        ▼
 hud.py  ── bounding boxes · labels · landmarks · contours · masks · graphs · side dashboard
```

### How the mood detector works (explain it like this)
1. Measure the face from the landmarks, always divided by face width: **mouth-corner lift**, **mouth width**, **eyebrow height**, **eyebrow gap**, MAR, EAR.
2. For the first 45 frames record **your neutral face** (baseline).
3. Compare each new frame with the baseline using simple IF rules (see `classify_mood` in `mood.py`):
   happy = corners up + wider · surprised = brows up + mouth open · angry = inner brows together and down · sad = corners down · tired = eyes less open / fatigue warning.
4. A **majority vote over 12 frames** removes flicker. All thresholds are in `config.py`.

It reports *facial expression*, not true feelings, and is rule based on purpose (easy to explain, no training data needed).

### Key formulas
* **EAR** (eye aspect ratio) `= (‖p2−p6‖ + ‖p3−p5‖) / (2‖p1−p4‖)` ≈ 0.30 open, → 0 closed.
  The "closed" threshold is **74 % of the user's own calibrated baseline**.
* **MAR** (mouth aspect ratio) from the 8 inner-lip landmarks; > 0.55 for ≥ 1 s = yawn.
* **PERCLOS** = share of the last 30 s with eyes closed; ≥ 25 % = WARNING.
* **DROWSY** = eyes continuously closed ≥ 1.5 s.

### Topic → code map (for the documentation / demo)

| Topic | File | OpenCV functions |
|-------|------|------------------|
| **a. Image Operations** | `preprocessing.py` | `resize`, `flip`, `cvtColor`, `LUT` (gamma), `createCLAHE`, `GaussianBlur`, `bitwise_and`, `addWeighted` |
| **b. Image Segmentation** | `motion.py`, `object_tracker.py` | `createBackgroundSubtractorMOG2`, `threshold`, `morphologyEx`, `dilate`, `inRange`, `calcBackProject` |
| **c. Haar Cascade Classifier** | `drowsiness.py`, `mood.py`, `motion.py` | `CascadeClassifier.detectMultiScale` (face, eyes, smile) |
| **d. Image Analysis & Transformation** | `drowsiness.py`, `motion.py`, `preprocessing.py` | `findContours`, `contourArea`, `moments`, `boundingRect`, `convexHull`, `calcHist`, `getRotationMatrix2D`, `warpAffine` |
| **e. Motion & Object Tracking** | `motion.py`, `object_tracker.py` | MOG2 motion detection, centroid tracking with IDs/trails, virtual-line counting, `CamShift` |
| **f. Facial Landmark Detection** | `drowsiness.py`, `mood.py` | `cv2.face.createFacemarkLBF` (68 points) → EAR, MAR, mood measurements |

---

## 4. Project structure

```
visionguard/
├── main.py                 # app: input handling, modes, window/GUI, key controls, clean exit
├── config.py               # every threshold / constant in one place
├── preprocessing.py        # (a)(d) image operations
├── drowsiness.py           # (c)(f)(d) Haar + landmarks + EAR/MAR + state machine
├── mood.py                 # (f)(c) mood rules from landmarks + Haar smile
├── motion.py               # (b)(e)(d)(c) segmentation, contours, tracking, counting
├── object_tracker.py       # (b)(e) HSV back-projection + CamShift
├── hud.py                  # drawing: boxes, labels, overlays, masks, dashboard
├── utils.py                # model download, FPS, CSV event log, snapshots, alarm
├── download_models.py
├── samples/                # sample inputs (image + 4 videos)
├── docs/                   # report PDF + screenshots + build_report.py
├── scripts/                # make_samples.py, capture_screenshots.py
├── tests/test_core.py      # unit tests (python -m unittest discover -s tests -v)
├── requirements.txt
└── output/                 # snapshots/ and events.csv are written here at run time
```

Run the tests (17): `python -m unittest discover -s tests -v`

## 5. Outputs
* Live window: processed frames with **bounding boxes, labels, landmarks, contours, masks, overlays** and a dashboard of live measurements.
* `output/events.csv` – timestamped log (DROWSY, YAWN, COUNT, MULTIPLE_PEOPLE, LINE_CROSS…).
* `output/session_summary.txt` – duration, blinks, yawns, mood mix, people counted.
* `output/snapshots/` – automatic drowsiness / multi-person snapshots and manual `P` screenshots.

## 6. Presentation guide (15 min demo + 5 min Q&A)

| Time | What to show | What to say |
|------|--------------|-------------|
| 0–2 min | Title, problem, the 3 modes | "One app, three real problems, all six lab topics, classical OpenCV only." |
| 2–7 min | **Mode 1** – calibrate with a neutral face, smile, look surprised, blink, close eyes 2 s, yawn | Haar finds the face → 68 landmarks → EAR/MAR → rules; mood = measurements vs. your neutral face; dashboard graphs |
| 7–11 min | **Mode 2** – leave the frame while the background is learned, walk in (count 1), let a friend join (count 2 + beep), stand still, leave | MOG2 segmentation → contours/moments → centroid tracker IDs → Haar faces → people count + majority vote; snapshot + CSV log |
| 11–13 min | **Mode 3** – press S, select an object, move it | Hue histogram back-projection (segmentation) + CamShift (tracking) |
| 13–15 min | Press **M** (mini view), **F** (fullscreen), close the window | GUI features, clean exit, `output/session_summary.txt` |
| Q&A | Open the code | See the topic → code table above; limitations section below |

**Likely questions:** *Why EAR and not deep learning?* (fast, explainable, works on a laptop CPU.) *Why calibrate?* (eye shape and neutral face differ per person.) *Why a majority vote?* (removes flicker from one noisy frame.) *What is CLAHE for?* (even out lighting before Haar / landmarks.) *Why a low MOG2 learning rate?* (slow walkers must not be absorbed into the background.)

## 7. Limitations (honest list)
* Haar cascades and LBF landmarks are classical methods: they work best with a **frontal, well-lit face**; strong head turns, heavy glare on glasses, or very low light reduce accuracy (gamma + CLAHE help but do not fix everything).
* Mode 2 needs a **fixed camera**; sudden lighting changes can trigger false motion until the background model adapts. Two people standing shoulder to shoulder merge into one blob and are only counted as two if both faces are visible. A person who stays motionless **with their back turned** fades into the background and is dropped from the count. The count follows reality with a delay of roughly 0.2–0.65 s (the 8-frame vote).
* The centroid tracker can swap IDs when objects cross paths, which can affect the IN/OUT count.
* The mood detector is rule based and reports facial **expression**; it needs a neutral face at calibration, subtle expressions can be missed, and the thresholds in `config.py` may need tuning for each person (press `D` to see the live numbers).
* Mode 3 (colour tracking) fails if the background has the same colour as the object.

## 8. Possible improvements
Head-pose estimation (`solvePnP`) for nodding, a trained expression classifier (small CNN) to replace the mood rules, a DNN face detector for large head angles, Kalman filter for smoother tracking, multi-camera support, SMS / e-mail alerts, and a GUI for tuning thresholds.
