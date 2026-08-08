"""
Phase 2 — Gesture state machine
Maps finger-state combos to named gestures, debounces flicker with a
rolling majority-vote buffer, and handles the "hold pinky-up for ~1s"
trigger for smart shape correction.

Phase 4.5 — Selection transform gestures added:
  GRAB  (fist, all fingers down)                 -> drags active selection
  PINCH (thumb+index close, other 3 fingers down) -> scales active selection
                                                       in real time as the
                                                       pinch distance changes

PINCH is checked with priority over GRAB, since a fully-closed pinch and a
fist can look identical in plain finger up/down terms (both curl the index
down toward the palm). The `extension_ratio` in pinch_metrics is what
actually tells them apart: a pinch reaches OUT away from the palm, a fist
stays curled IN near it.
"""

import time
from collections import deque

import cv2

from hand_tracking import (
    create_hand_landmarker,
    get_landmark_positions,
    get_finger_states,
    get_pinch_metrics,
    draw_hand_landmarks,
)
import mediapipe as mp

# ---------------------------------------------------------------------------
# Gesture name constants
# ---------------------------------------------------------------------------
DRAW = "DRAW"
MOVE = "MOVE"
ERASE = "ERASE"
CORRECT_CANDIDATE = "CORRECT_CANDIDATE"
CORRECT = "CORRECT"
SELECT = "SELECT"     # thumb+index+middle up, drag a selection box
GRAB = "GRAB"          # fist, drags active selection
PINCH = "PINCH"        # thumb+index pinch, scales active selection live
UNKNOWN = "UNKNOWN"    # finger combo doesn't match any known gesture

BUFFER_SIZE = 8          # rolling buffer length (8-10 frames per the doc)
HOLD_SECONDS = 1.0       # how long CORRECT_CANDIDATE must be stable to fire CORRECT

# --- Pinch detection tuning ---
# Both are starting values — tune against your own hand once you're testing.
PINCH_MAX_DISTANCE_RATIO = 0.6
# thumb-tip-to-index-tip distance, as a fraction of hand scale (wrist-to-
# middle-knuckle). Below this, thumb+index count as "pinching." If your
# scale-up gesture (spreading them apart) stops registering before you've
# spread comfortably far, raise this.

PINCH_MIN_INDEX_EXTENSION_RATIO = 0.6
# How straight the index finger must be (tip-to-its-own-MCP distance) to
# count as reaching toward a pinch rather than curled into a fist. This is
# the fist-vs-pinch discriminator. If a fist is still occasionally
# misread as PINCH, raise this. If a real pinch (index slightly bent) is
# being misread as GRAB, lower this.


def classify_gesture(finger_states, pinch_metrics=None):
    """
    Map a [thumb, index, middle, ring, pinky] boolean list (plus optional
    pinch_metrics from get_pinch_metrics) to a raw gesture name. This is a
    per-frame classification with NO debouncing — flicker is handled
    separately by GestureStateMachine.
    """
    thumb, index, middle, ring, pinky = finger_states

    # PINCH is checked first: it only needs middle/ring/pinky curled down
    # (same as DRAW and as a fist), and uses actual landmark geometry
    # rather than the thumb/index up-down booleans, which are unreliable
    # mid-pinch. The index-straightness check is what keeps a real fist
    # from being misread as a pinch.
    if pinch_metrics is not None and not middle and not ring and not pinky:
        if (pinch_metrics["distance_ratio"] < PINCH_MAX_DISTANCE_RATIO
                and pinch_metrics["index_extension_ratio"] > PINCH_MIN_INDEX_EXTENSION_RATIO):
            return PINCH

    if not any(finger_states):
        return GRAB
    if not thumb and index and not middle and not ring and not pinky:
        return DRAW
    if not thumb and index and middle and not ring and not pinky:
        return MOVE
    if thumb and index and middle and ring and pinky:
        return ERASE
    if pinky and not thumb and not index and not middle and not ring:
        return CORRECT_CANDIDATE
    if thumb and index and middle and not ring and not pinky:
        return SELECT

    return UNKNOWN


class GestureStateMachine:
    """
    Wraps raw per-frame gesture classification with:
      1. A rolling buffer + majority vote, to kill single-frame flicker.
      2. A hold-timer for CORRECT_CANDIDATE -> fires CORRECT once held ~1 second.
    """

    def __init__(self, buffer_size=BUFFER_SIZE, hold_seconds=HOLD_SECONDS):
        self.buffer = deque(maxlen=buffer_size)
        self.hold_seconds = hold_seconds

        self._candidate_start_time = None
        self._correct_already_fired = False

    def _stable_gesture(self):
        """Majority vote across the rolling buffer."""
        if not self.buffer:
            return UNKNOWN
        counts = {}
        for g in self.buffer:
            counts[g] = counts.get(g, 0) + 1
        return max(counts, key=counts.get)

    def update(self, finger_states, pinch_metrics=None):
        """
        Feed one frame's finger states (+ optional pinch metrics) in.
        Returns the current *stable* gesture name, which will be "CORRECT"
        for exactly one update() call when the pinky-hold completes
        (edge-triggered, not repeated every frame).
        """
        raw_gesture = classify_gesture(finger_states, pinch_metrics)
        self.buffer.append(raw_gesture)
        stable = self._stable_gesture()

        if stable == CORRECT_CANDIDATE:
            if self._candidate_start_time is None:
                self._candidate_start_time = time.time()

            held_duration = time.time() - self._candidate_start_time
            if held_duration >= self.hold_seconds and not self._correct_already_fired:
                self._correct_already_fired = True
                return CORRECT
            return CORRECT_CANDIDATE
        else:
            self._candidate_start_time = None
            self._correct_already_fired = False
            return stable


def main():
    """Standalone test: overlay the stable gesture name on screen (Phase 2 checkpoint)."""
    landmarker = create_hand_landmarker()
    state_machine = GestureStateMachine()
    cap = cv2.VideoCapture(0)

    cv2.namedWindow("Phase 2 - Gesture State Test", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Phase 2 - Gesture State Test", 960, 540)

    start_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

        timestamp_ms = int((time.time() - start_time) * 1000)
        result = landmarker.detect_for_video(mp_image, timestamp_ms)

        gesture_label = "NO HAND"
        if result.hand_landmarks:
            hand_landmarks = result.hand_landmarks[0]
            positions = get_landmark_positions(hand_landmarks, frame.shape)
            draw_hand_landmarks(frame, positions)

            finger_states = get_finger_states(positions)
            pinch_metrics = get_pinch_metrics(positions)
            gesture_label = state_machine.update(finger_states, pinch_metrics)

        cv2.putText(
            frame, gesture_label, (20, 50),
            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 3, cv2.LINE_AA
        )

        cv2.imshow("Phase 2 - Gesture State Test", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    landmarker.close()


if __name__ == "__main__":
    main()