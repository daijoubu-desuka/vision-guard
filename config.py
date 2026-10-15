"""
config.py - Central configuration for VisionGuard.

Every "magic number" used by the project lives here so it can be tuned in
one place (and explained in the documentation) instead of being scattered
through the code.
"""

import os

# Absolute base folder: the app works no matter where it is launched from (VS Code, terminal...)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ----------------------------------------------------------------------------
# General
# ----------------------------------------------------------------------------
FRAME_WIDTH = 640            # every frame is resized to this width before processing
PANEL_WIDTH = 300            # width of the dashboard panel drawn next to the video
LOW_LIGHT_BRIGHTNESS = 80    # mean gray level below this -> apply gamma correction
TARGET_BRIGHTNESS = 115      # brightness the gamma correction tries to reach

# ----------------------------------------------------------------------------
# Mode 1 - Drowsiness monitor (Haar cascade + facial landmarks)
# ----------------------------------------------------------------------------
LANDMARK_MODEL_FILE = os.path.join(BASE_DIR, "models", "lbfmodel.yaml")
LANDMARK_MODEL_URL = (
    "https://raw.githubusercontent.com/kurnianggoro/GSOC2017/master/data/lbfmodel.yaml"
)

FACE_SCALE_FACTOR = 1.2      # Haar pyramid step: 1.2 is ~2x faster than 1.1 with similar accuracy
FACE_MIN_SIZE = (90, 90)     # smallest face (px) that Haar is allowed to return
FACE_GRACE_FRAMES = 5        # keep last face box this many frames if Haar misses once
FACE_SMOOTHING = 0.6         # exponential smoothing of the face box (0..1, higher = smoother)

CALIBRATION_FRAMES = 45      # frames used to learn the user's open-eye EAR baseline
EAR_CLOSED_RATIO = 0.74      # eyes are "closed" when EAR < ratio * personal baseline
EAR_FALLBACK_THRESHOLD = 0.21  # used before calibration finishes / if baseline is unusable
MAR_YAWN_THRESHOLD = 0.55    # mouth aspect ratio above this = mouth wide open

EYE_CLOSED_SECONDS = 1.5     # continuous eye closure that triggers the DROWSY alarm
YAWN_SECONDS = 1.0           # mouth open this long = a yawn
BLINK_MIN_SECONDS = 0.08     # closure shorter than this is noise
BLINK_MAX_SECONDS = 0.50     # closure longer than this is not a blink
PERCLOS_WINDOW = 30.0        # seconds of history used for PERCLOS
PERCLOS_WARN = 0.25          # >25% of the window with eyes closed = WARNING
NO_FACE_SECONDS = 3.0        # no face for this long = "driver/student away"

# ----------------------------------------------------------------------------
# Mood detector (landmark geometry + Haar smile cascade) - part of mode 1
# All values are compared with the user's own NEUTRAL face (calibrated at start)
# ----------------------------------------------------------------------------
MOOD_CALIBRATION_FRAMES = 45   # frames of neutral face used as the personal baseline
MOOD_SMOOTH_FRAMES = 12        # majority vote over the last N frames (stops flicker)
MOOD_MIN_VOTES = 6             # a mood must win at least this many of those frames
MOOD_SMILE_LIFT = 0.015        # mouth corners rise by this (fraction of face width) = smile
MOOD_SMILE_WIDTH = 1.05        # mouth gets >= 5% wider than neutral
MOOD_SMILE_LIFT_HAAR = 0.008   # smaller lift is enough when the Haar smile cascade agrees
MOOD_SAD_LIFT = -0.012         # mouth corners drop by this = sad
MOOD_SURPRISE_MAR = 0.25       # mouth opens this much more than neutral (MAR units)
MOOD_SURPRISE_BROW = 1.12      # eyebrows raised >= 12% above neutral
MOOD_ANGRY_FURROW = 0.90       # inner eyebrows >= 10% closer together than neutral
MOOD_ANGRY_BROW = 0.95         # and brows slightly lowered
MOOD_TIRED_EAR_RATIO = 0.80    # eyes less than 80% as open as neutral

# ----------------------------------------------------------------------------
# Window / GUI
# ----------------------------------------------------------------------------
WINDOW_NAME = "VisionGuard"
MINI_WIDTH = 360               # width of the small floating "mini view"
TOOLBAR_HEIGHT = 36

# ----------------------------------------------------------------------------
# Mode 2 - People counter (motion detection + Haar faces + tracking)
# ----------------------------------------------------------------------------
MOTION_MIN_AREA = 3000       # smallest MERGED object (px^2) treated as a moving object
MOTION_FRAGMENT_AREA = 500   # smallest single contour piece kept before merging
MOTION_MERGE_GAP = 70        # pieces closer than this (px) are merged into one person/object
MOTION_GLOBAL_CHANGE = 35    # % of frame moving at once = lighting change, not a person
TRACK_CONFIRM_FRAMES = 6     # an object must be seen this many frames before it counts
MOTION_WARMUP_FRAMES = 40    # frames the background model learns before arming
MOG2_HISTORY = 400
MOG2_VAR_THRESHOLD = 50
MOG2_LEARNING_RATE = 0.002   # low = slow-moving people are not absorbed into the background
MOTION_CLOSE_KERNEL = 25     # closing kernel (px) that merges fragments of one body
TRACK_MAX_DISAPPEARED = 15   # frames an object may vanish before its ID is dropped
TRACK_MAX_DISTANCE = 90      # max centroid jump (px) to still be the same object
TRACK_TRAIL_LENGTH = 30      # how many past centroids are drawn as a trail
COUNT_LINE_POSITION = 0.55   # virtual counting line, as a fraction of frame height
COUNT_LINE_MARGIN = 8        # hysteresis band (px) around the line to avoid double counts
SNAPSHOT_INTERVAL = 5.0      # min seconds between automatic multi-person snapshots

# People counting: a moving blob is a PERSON if a Haar face is inside it, or if it has a full-body shape.
# Faces are also searched in the WHOLE frame so a person who stops moving (and fades into the
# background model) is still counted.
PEOPLE_FACE_SCALE = 1.2         # Haar scale factor for the full-frame face search (1.2 = fast)
PEOPLE_FACE_EVERY = 2           # full-frame face search every N-th frame (boxes are reused in between)
PEOPLE_FACE_NEIGHBORS = 5       # higher = fewer false faces
PEOPLE_FACE_MIN_SIZE = (36, 36) # smallest face (px) that counts as a person
PEOPLE_CONFIRM_FRAMES = 3       # a blob must be tracked this long before it can count as a person
PEOPLE_HOLD_FRAMES = 4          # a person that vanishes for <= N frames still counts (flicker guard)
PEOPLE_FACE_LINK = 60           # a faceless person-blob and a free face this close (px) are the SAME person
PEOPLE_BODY_MIN_HEIGHT = 0.40   # faceless blob is a person only if taller than this x frame height ...
PEOPLE_BODY_ASPECT = (1.4, 4.0) # ... and its height / width is inside this range (standing figure)
PEOPLE_SMOOTH_FRAMES = 8        # the displayed count is a majority vote over the last N frames
PEOPLE_MIN_VOTES = 5            # a count must win at least this many of those frames
PEOPLE_ALERT_COUNT = 2          # beep + alert when this many people (i.e. MORE THAN ONE) are in view
PEOPLE_BEEP_INTERVAL = 1.5      # seconds between beeps while the alert lasts

# ----------------------------------------------------------------------------
# Mode 3 - Object tracker (HSV segmentation + CamShift)
# ----------------------------------------------------------------------------
CAMSHIFT_MIN_SATURATION = 60   # ignore washed-out pixels when building the histogram
CAMSHIFT_MIN_VALUE = 40        # ignore very dark pixels
CAMSHIFT_LOST_MIN_AREA = 150   # tracking window smaller than this = object lost

# ----------------------------------------------------------------------------
# Output folders
# ----------------------------------------------------------------------------
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
SNAPSHOT_DIR = os.path.join(OUTPUT_DIR, "snapshots")
EVENT_LOG_FILE = os.path.join(OUTPUT_DIR, "events.csv")
