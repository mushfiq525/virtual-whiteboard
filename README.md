# ✋ Gesture-Controlled Smart Whiteboard

A webcam-based virtual whiteboard controlled entirely by hand gestures — no
mouse, no keyboard, no touchscreen. Draw with your fingertip in the air,
trigger a dedicated gesture to auto-correct rough shapes into clean
triangles/rectangles/circles, lasso-select anything you've drawn and
move or resize it in place, and paint-bucket fill closed regions — all
tracked live from a single RGB camera.

<!-- Add your demo video/GIF here — this is the single most important
     asset in this README. A screen recording with webcam picture-in-
     picture works well: e.g.

![demo](assets/demo.gif)

or a link to a hosted video:

[Watch the demo](your-video-link-here)
-->

## Live demo

<!-- Once deployed: [Try it on Hugging Face Spaces](your-space-link-here) -->

## Features

- **Freehand drawing** with real-time hand-tremor smoothing on the live line
- **Smart shape correction** — hold one gesture and a wobbly hand-drawn
  triangle/rectangle/circle snaps into a clean version, powered entirely by
  classical computer vision (convex hull + contour analysis), no ML model
  training involved
- **Select, move, and scale** any drawn item (or group of items) with
  Figma/Canva-style rubber-band selection and live pinch-to-scale
- **Paint-bucket fill** for closed regions, safely bounded so a gap in a
  stroke can't flood the whole canvas
- **12-color palette + undo**, both hover-and-hold to trigger — no keyboard
  required (a `'z'` keyboard shortcut for undo is also available when
  running the local OpenCV test harness)
- **FPS-adaptive tracking** that backs off hand-landmark detection under
  load instead of letting the whole app stutter
- Runs in-browser via **Streamlit + streamlit-webrtc**, deployable to
  Hugging Face Spaces

## Gesture legend

| Gesture | Action |
|---|---|
| Index finger up only (thumb closed) | Draw |
| Index + middle finger up together | Move / navigate (no drawing) |
| Open palm (all 5 fingers up) | Erase (area under palm) |
| Pinky only, held ~1 second | Trigger smart shape correction on the last-drawn stroke |
| Thumb + index + middle up, drag | Select — draws a box; on release, locks in every item the box fully encloses |
| Fist (all 5 fingers closed) | Move the locked selection — drag your fist and the selected content follows |
| Pinch: thumb + index together, other 3 fingers closed, index kept straight | Scale the locked selection live — spread apart to grow, bring together to shrink |
| Index + middle + ring up, held ~1 second | Paint-bucket fill — floods the closed region under your fingertip with the active color |

There's no handwriting/OCR correction in this version — smart correction
only recognizes triangles, rectangles, and circles.

## How it works

### Hand tracking
[MediaPipe's](https://developers.google.com/mediapipe) pretrained
`HandLandmarker` (Tasks API) returns 21 hand landmarks per frame. All
gesture logic downstream is plain landmark geometry — no custom model
training anywhere in this project.

- **Finger up/down** for the four non-thumb fingers is a simple tip-vs-PIP
  y-coordinate check.
- **Thumb up/down** uses a rotation-invariant distance check (tip vs. base
  knuckle, both relative to the pinky knuckle) instead of comparing raw
  x-coordinates — the naive x-based approach breaks whenever the hand is
  rotated diagonally, e.g. reaching toward a frame corner.
- **Pinch vs. fist** are geometrically similar (both curl the index toward
  the palm) but are told apart using the index fingertip's straightness:
  genuinely extended in a pinch, fully curled in a fist.

### Gesture state machine
Raw per-frame gesture classification is debounced with a rolling
majority-vote buffer to kill single-frame flicker, and any "hold for ~1s"
gesture (shape correction, fill) uses a shared hold-timer that fires its
one-shot trigger exactly once per hold, not once per frame.

### Smart shape correction
A completed stroke's points are smoothed (moving average) to remove
hand-tremor jitter, then reduced to a convex hull — this is what makes
classification robust to a wobbly real-world stroke, since inward zigzags
get discarded instead of inflating the perimeter or adding phantom
vertices. Before classifying, two checks reject anything that isn't a
plausible closed shape:

- **Closure ratio** — how far apart the stroke's start and end points are,
  relative to its own bounding-box diagonal. Real shapes come back near
  where they started; most handwriting doesn't.
- **Solidity** — traced area vs. hull area. A real shape is nearly as
  convex as its own hull; a loopy letter (P, R, B, D) leaves a large gap
  between what was drawn and what the hull bridges over.

What's left is classified via `cv2.approxPolyDP()` vertex count (3 →
triangle, 4 with ~90° angles → rectangle) or circularity
(`4π·Area / Perimeter²`) for a circle. Anything unrecognized is left as
the original stroke.

### Selection, move, and scale
Every stroke and corrected shape is tracked as an "item" with its own
bounding box, not just raw pixels — this is what makes selection possible.
A `SELECT` drag locks in every item whose bbox is **fully contained**
within the drag box (containment, not mere overlap — this is what keeps
selecting a small shape nested inside a bigger one from also dragging the
bigger one along with it). The lock persists through `DRAW`/`MOVE`/`ERASE`
until a new `SELECT` drag starts or the canvas is cleared.

`GRAB` translates the whole selected group by the frame-to-frame palm-center
delta, with a small deadzone so landmark jitter doesn't accumulate into
visible drift while your hand is still. `PINCH` scales the group live,
anchored on the selection's own center so it doesn't drift while resizing,
with each frame's scale factor clamped to a narrow band so one noisy
landmark frame can't cause a visible "pop" in size.

### Paint-bucket fill
`cv2.floodFill` only ever replaces background-colored pixels, so any
stroke or shape border already on the canvas stops the flood dead — filling
the ring between two nested closed shapes only paints the ring, never
crossing into another enclosed region or covering the border strokes
themselves. Since a single-pixel gap in a hand-drawn border would
otherwise let the flood leak across the whole canvas, every fill is first
tried on a scratch copy and only committed if the filled area stays under
a safety threshold — anything bigger is treated as "not actually closed"
and aborted.

## Architecture

```
Webcam
  │
  ▼
MediaPipe HandLandmarker  →  21 hand landmarks per frame
  │
  ▼
Gesture State Machine  →  finger-state geometry → debounced gesture name
  │
  ▼
Drawing / Correction / Selection / Fill Logic  →  Canvas item list
  │
  ▼
Rendered Output  →  composited onto the live camera frame
```

## Tech stack

| Library | Purpose |
|---|---|
| `mediapipe` | Hand landmark detection |
| `opencv-python` | Video capture/decoding, drawing, contour & shape analysis |
| `numpy` | Canvas array, point-list math |
| `streamlit` + `streamlit-webrtc` | Browser-based webcam app and deployment |
| `av` | Frame encode/decode for `streamlit-webrtc` |

No model training is involved anywhere — hand tracking comes from
MediaPipe's pretrained model, and shape correction is classical computer
vision (convex hull + contour analysis), not a trained classifier.

## Project structure

```
gesture-whiteboard/
├── src/
│   ├── hand_tracking.py       # MediaPipe landmarks, finger-state/pinch geometry
│   ├── gesture_state.py       # Gesture classification + debouncing + hold-triggers
│   ├── drawing.py             # Canvas, selection, palette/undo UI — local test entry point
│   ├── shape_correction.py    # Convex hull + contour-based shape classification
│   └── main.py                # Throwaway Day-1 camera check only, not used elsewhere
├── assets/
│   └── hand_landmarker.task   # MediaPipe model file (download separately, see below)
├── app.py                     # Streamlit + streamlit-webrtc entry point
├── requirements.txt
└── README.md
```

## Setup

### 1. Camera source
This project was built using a phone as a webcam via
[Iriun Webcam](https://iriun.com) (free) — install it on both your phone
and PC, connect them to the same network (USB is more stable than WiFi for
real-time work), and Iriun exposes itself as a virtual webcam device.

### 2. Environment
Requires Python 3.9–3.11 (MediaPipe doesn't yet reliably support the
newest Python versions).

```bash
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Download the hand-tracking model
```bash
mkdir -p assets
curl -o assets/hand_landmarker.task \
  https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task
```

### 4. Run locally

**OpenCV test harness** (fastest for tuning gesture thresholds — shows raw
FPS, opens a native window, supports `'c'` clear / `'z'` undo keys):
```bash
python src/drawing.py
```

**Streamlit app** (the browser-based version used for deployment):
```bash
streamlit run app.py
```
This opens in your browser and prompts for camera access — it uses
whatever your system's active camera device is (your Iriun virtual camera,
if that's what's active).

## Known limitations

- Single-hand tracking only.
- Shape correction recognizes triangles, rectangles, and circles only —
  no handwriting/OCR.
- A phone-as-webcam setup over WiFi can introduce enough latency to make
  gestures feel delayed; USB connection is recommended.
- Gesture thresholds (pinch vs. fist, thumb up/down margin, etc.) are
  tuned against the author's own hand and camera angle — see the tuning
  constants at the top of `hand_tracking.py` and `gesture_state.py` if
  gestures misfire on a different hand/setup.

## Live Demo
go to https://virtual-whiteboard-project.streamlit.app/

<!-- Add your license here, e.g. MIT -->
