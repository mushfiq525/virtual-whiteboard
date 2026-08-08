"""
Phase 1 — Hand tracking foundation
Uses the MediaPipe TASKS API (HandLandmarker), since mediapipe>=1.0.0
removed the old mp.solutions.hands legacy API used in older tutorials.

Requires: assets/hand_landmarker.task
Download from:
https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task
"""

import math
import os
import time
import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

# ---------------------------------------------------------------------------
# Model path — adjust if you saved it somewhere else
# ---------------------------------------------------------------------------
MODEL_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "assets",
    "hand_landmarker.task",
)

# Standard 21-point hand connection topology (replaces mp.solutions.HAND_CONNECTIONS,
# which no longer exists in the Tasks API).
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),          # thumb
    (0, 5), (5, 6), (6, 7), (7, 8),          # index
    (5, 9), (9, 10), (10, 11), (11, 12),     # middle
    (9, 13), (13, 14), (14, 15), (15, 16),   # ring
    (13, 17), (17, 18), (18, 19), (19, 20),  # pinky
    (0, 17),                                 # palm base
]

FINGER_TIPS = {"index": 8, "middle": 12, "ring": 16, "pinky": 20}
FINGER_PIPS = {"index": 6, "middle": 10, "ring": 14, "pinky": 18}
THUMB_TIP = 4
THUMB_IP = 3


def create_hand_landmarker():
    """Builds a HandLandmarker configured for VIDEO mode (frame-by-frame webcam use).

    Confidence thresholds are set to MediaPipe's recommended default of 0.5.
    Higher values (e.g. 0.7) reject valid detections when the hand is near
    the edge of frame, partially out of view, or slightly motion-blurred —
    which is most of the time in a whiteboard app where the hand moves
    across the whole frame.
    """
    base_options = mp_python.BaseOptions(model_asset_path=MODEL_PATH)
    options = mp_vision.HandLandmarkerOptions(
        base_options=base_options,
        running_mode=mp_vision.RunningMode.VIDEO,
        num_hands=1,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    return mp_vision.HandLandmarker.create_from_options(options)


def get_landmark_positions(hand_landmarks, frame_shape):
    """
    Convert one detected hand's normalized (0-1) landmarks into pixel coordinates.

    Args:
        hand_landmarks: a list of 21 landmark objects (each with .x, .y, .z),
                         e.g. results.hand_landmarks[i] from HandLandmarker.
        frame_shape: frame.shape -> (height, width, channels)

    Returns:
        List of 21 (x, y) pixel coordinate tuples.
    """
    height, width = frame_shape[0], frame_shape[1]
    positions = []
    for lm in hand_landmarks:
        px = int(lm.x * width)
        py = int(lm.y * height)
        positions.append((px, py))
    return positions


def get_finger_states(landmarks):
    """
    Determine which fingers are up/down.

    Args:
        landmarks: list of 21 (x, y) pixel coordinates from get_landmark_positions().

    Returns:
        [thumb, index, middle, ring, pinky] as booleans (True = up).
    """
    states = []

    # Thumb: x-based check, orientation-aware (works for either hand in frame)
    thumb_tip_x = landmarks[THUMB_TIP][0]
    thumb_ip_x = landmarks[THUMB_IP][0]
    wrist_x = landmarks[0][0]
    middle_mcp_x = landmarks[9][0]
    if middle_mcp_x >= wrist_x:
        thumb_up = thumb_tip_x > thumb_ip_x
    else:
        thumb_up = thumb_tip_x < thumb_ip_x
    states.append(thumb_up)

    # Other four fingers: y-based check
    for finger in ("index", "middle", "ring", "pinky"):
        tip_y = landmarks[FINGER_TIPS[finger]][1]
        pip_y = landmarks[FINGER_PIPS[finger]][1]
        states.append(tip_y < pip_y)

    return states


def get_pinch_metrics(landmarks):
    """
    Measures the thumb-index "pinch" shape, used to detect the PINCH/scale
    gesture and to disambiguate it from a closed fist (GRAB). Both metrics
    are normalized by a hand-scale reference (wrist-to-middle-knuckle
    distance) so they stay roughly consistent regardless of how close the
    hand is to the camera.

    Returns a dict:
      - "distance_ratio": thumb tip <-> index tip distance. Small when
        pinching (touching or nearly so), larger as they spread apart.
      - "index_extension_ratio": distance from the index fingertip to its
        own base knuckle (index MCP, landmark 5). This is what separates a
        genuine pinch from a fist: in a pinch the index finger stays
        relatively straight (large ratio) even while its tip touches the
        thumb; in a fist the index is curled all the way in (small ratio)
        regardless of where the thumb happens to rest on top of it.

    Returns None if the hand-scale reference collapses to zero (shouldn't
    happen in practice, but guards a divide-by-zero).
    """
    thumb_tip = landmarks[THUMB_TIP]
    index_tip = landmarks[FINGER_TIPS["index"]]
    index_mcp = landmarks[5]
    wrist = landmarks[0]
    middle_mcp = landmarks[9]

    hand_scale = math.hypot(middle_mcp[0] - wrist[0], middle_mcp[1] - wrist[1])
    if hand_scale == 0:
        return None

    distance = math.hypot(index_tip[0] - thumb_tip[0], index_tip[1] - thumb_tip[1])
    index_extension = math.hypot(index_tip[0] - index_mcp[0], index_tip[1] - index_mcp[1])

    return {
        "distance_ratio": distance / hand_scale,
        "index_extension_ratio": index_extension / hand_scale,
    }


def draw_hand_landmarks(frame, pixel_positions):
    """Manually draws landmark dots + connective lines (replaces mp_drawing.draw_landmarks)."""
    for start_idx, end_idx in HAND_CONNECTIONS:
        cv2.line(frame, pixel_positions[start_idx], pixel_positions[end_idx], (0, 255, 0), 2)
    for (x, y) in pixel_positions:
        cv2.circle(frame, (x, y), 4, (0, 0, 255), -1)


def main():
    """Standalone test loop: webcam feed with landmarks drawn + live finger-state printout."""
    if not os.path.exists(MODEL_PATH):
        print(f"ERROR: model file not found at {MODEL_PATH}")
        print("Download it from: https://storage.googleapis.com/mediapipe-models/"
              "hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task")
        return

    landmarker = create_hand_landmarker()
    cap = cv2.VideoCapture(0)  # adjust index if needed (0, 1, 2...)

    cv2.namedWindow("Phase 1 - Hand Tracking Test", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Phase 1 - Hand Tracking Test", 960, 540)

    start_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # frame = cv2.flip(frame, 1)  # mirror for natural movement
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

        # VIDEO mode requires a monotonically increasing timestamp in ms
        timestamp_ms = int((time.time() - start_time) * 1000)
        result = landmarker.detect_for_video(mp_image, timestamp_ms)

        if result.hand_landmarks:
            for hand_landmarks in result.hand_landmarks:
                positions = get_landmark_positions(hand_landmarks, frame.shape)
                draw_hand_landmarks(frame, positions)

                finger_states = get_finger_states(positions)
                print(f"[thumb, index, middle, ring, pinky] = {finger_states}")

        cv2.imshow("Phase 1 - Hand Tracking Test", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    landmarker.close()


if __name__ == "__main__":
    main()