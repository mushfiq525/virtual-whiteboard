"""
Phase 2 — Gesture state machine
Maps finger-state combos to named gestures, debounces flicker with a
rolling majority-vote buffer, and handles the "hold pinky-up for ~1s"
trigger for smart correction.
"""

import time
from collections import deque

import cv2

# Import Phase 1 pieces so this file's own test loop can run standalone
from hand_tracking import (
    create_hand_landmarker,
    get_landmark_positions,
    get_finger_states,
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
UNKNOWN = "UNKNOWN"  # finger combo doesn't match any known gesture

BUFFER_SIZE = 8          # rolling buffer length (8-10 frames per the doc)
HOLD_SECONDS = 1.0       # how long CORRECT_CANDIDATE must be stable to fire CORRECT


def classify_gesture(finger_states):
    """
    Map a [thumb, index, middle, ring, pinky] boolean list to a raw gesture name.
    This is a per-frame classification with NO debouncing — flicker is handled
    separately by GestureStateMachine.
    """
    thumb, index, middle, ring, pinky = finger_states

    if not thumb and index and not middle and not ring and not pinky:
        return DRAW
    if not thumb and index and middle and not ring and not pinky:
        return MOVE
    if thumb and index and middle and ring and pinky:
        return ERASE
    if pinky and not thumb and not index and not middle and not ring:
        return CORRECT_CANDIDATE

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

        # Tracks how long CORRECT_CANDIDATE has been continuously stable
        self._candidate_start_time = None
        self._correct_already_fired = False  # prevents re-firing every frame while held

    def _stable_gesture(self):
        """Majority vote across the rolling buffer."""
        if not self.buffer:
            return UNKNOWN
        counts = {}
        for g in self.buffer:
            counts[g] = counts.get(g, 0) + 1
        return max(counts, key=counts.get)

    def update(self, finger_states):
        """
        Feed one frame's finger states in. Returns the current *stable* gesture
        name, which will be "CORRECT" for exactly one update() call when the
        thumbs-up hold completes (edge-triggered, not repeated every frame).
        """
        raw_gesture = classify_gesture(finger_states)
        self.buffer.append(raw_gesture)
        stable = self._stable_gesture()

        if stable == CORRECT_CANDIDATE:
            if self._candidate_start_time is None:
                self._candidate_start_time = time.time()

            held_duration = time.time() - self._candidate_start_time
            if held_duration >= self.hold_seconds and not self._correct_already_fired:
                self._correct_already_fired = True
                return CORRECT  # fires once
            return CORRECT_CANDIDATE
        else:
            # left the candidate pose — reset hold tracking
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

        # frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

        timestamp_ms = int((time.time() - start_time) * 1000)
        result = landmarker.detect_for_video(mp_image, timestamp_ms)

        gesture_label = "NO HAND"
        if result.hand_landmarks:
            hand_landmarks = result.hand_landmarks[0]  # single hand
            positions = get_landmark_positions(hand_landmarks, frame.shape)
            draw_hand_landmarks(frame, positions)

            finger_states = get_finger_states(positions)
            gesture_label = state_machine.update(finger_states)

        # Overlay gesture name
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