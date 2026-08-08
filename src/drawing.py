"""
Phase 3 — Core drawing logic
Phase 4 — Now also tracks strokes as *objects* (point lists), not just
pixels, so a completed stroke can be erased and replaced with a clean
shape after smart correction.
Phase 5 — If a stroke isn't a recognized shape, fall through to
handwriting correction (OCR) and replace it with a clean rendered
character instead.
"""

import time

import cv2
import numpy as np
import mediapipe as mp

from hand_tracking import (
    create_hand_landmarker,
    get_landmark_positions,
    get_finger_states,
    draw_hand_landmarks,
)
from gesture_state import GestureStateMachine, DRAW, MOVE, ERASE, CORRECT, CORRECT_CANDIDATE
from shape_correction import classify_stroke_debug, render_clean_shape
from handwriting_correction import recognize_character_debug, render_clean_text

# ---------------------------------------------------------------------------
# Canvas config
# ---------------------------------------------------------------------------
BACKGROUND_COLOR = (255, 255, 255)  # white, BGR
STROKE_THICKNESS = 5
ERASE_RADIUS = 40

# Palette (BGR tuples)
PALETTE_COLORS = [
    (0, 0, 0),        # black
    (0, 0, 255),      # red
    (0, 200, 0),      # green
    (255, 0, 0),      # blue
    (0, 255, 255),    # yellow
    (255, 0, 255),    # magenta
]
SWATCH_SIZE = 40
SWATCH_GAP = 10
SWATCH_MARGIN = 15
PALETTE_HOVER_SECONDS = 1.0
# Palette occupies roughly y=15 to y=63 (swatch bottom + hover progress bar).
# Gesture label must sit below that so nothing overlaps.
GESTURE_LABEL_Y = 100

PALM_LANDMARK_IDS = (0, 5, 9, 13, 17)
INDEX_TIP_ID = 8

# UI text styling
TEXT_COLOR = (0, 59, 6)  # BGR for #063B00 (dark green)
GESTURE_LABEL_FONT_SCALE = 0.8
STATUS_MESSAGE_FONT_SCALE = 0.6


class Canvas:
    """
    Persistent drawing surface, PLUS a list of "items" (completed strokes,
    corrected shapes, or corrected text) that lets us fully re-render the
    surface whenever a stroke needs to be erased-and-replaced (smart
    correction), without losing everything else that was drawn before it.
    """

    def __init__(self, height, width, bg_color=BACKGROUND_COLOR):
        self.bg_color = bg_color
        self.height = height
        self.width = width
        self.surface = np.full((height, width, 3), bg_color, dtype=np.uint8)

        self.prev_point = None       # last fingertip position while actively drawing
        self.items = []              # ordered list of drawn items
        self._current_stroke = None  # points for the stroke currently being drawn
        self._was_drawing = False    # tracks DRAW -> non-DRAW transitions

    # --- live per-frame drawing (fast path, drawn directly onto surface) ---

    def draw_line(self, pt1, pt2, color, thickness=STROKE_THICKNESS):
        cv2.line(self.surface, pt1, pt2, color, thickness)

    def erase_circle(self, center, radius=ERASE_RADIUS):
        cv2.circle(self.surface, center, radius, self.bg_color, -1)

    def reset_stroke(self):
        """Call whenever leaving DRAW state, so the next stroke doesn't
        connect to wherever the hand last was (Phase 3, step 13)."""
        self.prev_point = None

    def clear(self):
        self.surface[:] = self.bg_color
        self.items = []
        self._current_stroke = None
        self._was_drawing = False

    # --- stroke-object tracking (Phase 4) ---

    def update_stroke_tracking(self, is_drawing, point, color):
        """
        Call once per frame with whether we're currently in DRAW state and
        the current fingertip point (or None if not drawing).

        Starts a new stroke item whenever DRAW begins fresh (transition
        from a different state), appends points while DRAW continues, and
        finalizes (keeps as the "last completed stroke") when DRAW ends.
        """
        if is_drawing:
            if not self._was_drawing:
                # Fresh stroke starting
                self._current_stroke = {"type": "stroke", "points": [], "color": color}
                self.items.append(self._current_stroke)
            self._current_stroke["points"].append(point)
        else:
            if self._was_drawing:
                # Just finished a stroke — nothing more to do, it's already
                # in self.items as the most recent stroke item.
                pass
            self._current_stroke = None

        self._was_drawing = is_drawing

    def get_last_stroke_points(self):
        """Points of the most recently completed (or in-progress) stroke,
        or None if there isn't one yet."""
        for item in reversed(self.items):
            if item["type"] == "stroke":
                return item["points"], item["color"]
        return None, None

    def replace_last_stroke_with_shape(self, shape_result):
        """
        Find the most recent stroke item and swap it for a clean-shape item,
        then fully re-render the surface so the raw stroke pixels disappear
        and the clean shape appears in their place (Phase 4, step 22).
        """
        for i in range(len(self.items) - 1, -1, -1):
            if self.items[i]["type"] == "stroke":
                self.items[i] = {"type": "shape", "shape_result": shape_result}
                break
        self._rerender()

    def replace_last_stroke_with_text(self, text_result):
        """
        Same idea as replace_last_stroke_with_shape, but for a recognized
        character (Phase 5, step 28).
        """
        for i in range(len(self.items) - 1, -1, -1):
            if self.items[i]["type"] == "stroke":
                self.items[i] = {"type": "text", "text_result": text_result}
                break
        self._rerender()

    def _rerender(self):
        """Redraw the whole surface from scratch using self.items, in order.
        Used only on correction events (occasional), not every frame."""
        self.surface = np.full((self.height, self.width, 3), self.bg_color, dtype=np.uint8)
        for item in self.items:
            if item["type"] == "stroke":
                pts = item["points"]
                for p1, p2 in zip(pts, pts[1:]):
                    cv2.line(self.surface, p1, p2, item["color"], STROKE_THICKNESS)
            elif item["type"] == "shape":
                render_clean_shape(self.surface, item["shape_result"], thickness=STROKE_THICKNESS)
            elif item["type"] == "text":
                render_clean_text(self.surface, item["text_result"], thickness=STROKE_THICKNESS - 2)


def get_palette_rects(frame_width):
    rects = []
    x = SWATCH_MARGIN
    y = SWATCH_MARGIN
    for _ in PALETTE_COLORS:
        rects.append((x, y, x + SWATCH_SIZE, y + SWATCH_SIZE))
        x += SWATCH_SIZE + SWATCH_GAP
    return rects


def draw_palette(frame, rects, active_index, hover_index=None, hover_progress=0.0):
    for i, (x1, y1, x2, y2) in enumerate(rects):
        color = PALETTE_COLORS[i]
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, -1)
        border_color = (255, 255, 255) if i == active_index else (0, 0, 0)
        border_thickness = 3 if i == active_index else 1
        cv2.rectangle(frame, (x1, y1), (x2, y2), border_color, border_thickness)
        if hover_index == i and hover_progress > 0:
            bar_width = int((x2 - x1) * hover_progress)
            cv2.rectangle(frame, (x1, y2 + 4), (x1 + bar_width, y2 + 8), (0, 255, 255), -1)


def get_hovered_swatch(point, rects):
    px, py = point
    for i, (x1, y1, x2, y2) in enumerate(rects):
        if x1 <= px <= x2 and y1 <= py <= y2:
            return i
    return None


def get_palm_center(positions):
    xs = [positions[i][0] for i in PALM_LANDMARK_IDS]
    ys = [positions[i][1] for i in PALM_LANDMARK_IDS]
    return (int(sum(xs) / len(xs)), int(sum(ys) / len(ys)))


def overlay_canvas(frame, canvas_surface, bg_color=BACKGROUND_COLOR):
    mask = np.any(canvas_surface != bg_color, axis=-1)
    output = frame.copy()
    output[mask] = canvas_surface[mask]
    return output


class PaletteSelector:
    def __init__(self, hover_seconds=PALETTE_HOVER_SECONDS):
        self.hover_seconds = hover_seconds
        self._hover_index = None
        self._hover_start_time = None

    def update(self, hovered_index):
        if hovered_index is None:
            self._hover_index = None
            self._hover_start_time = None
            return None, 0.0, None

        if hovered_index != self._hover_index:
            self._hover_index = hovered_index
            self._hover_start_time = time.time()
            return hovered_index, 0.0, None

        elapsed = time.time() - self._hover_start_time
        progress = min(elapsed / self.hover_seconds, 1.0)

        if elapsed >= self.hover_seconds:
            self._hover_start_time = time.time()
            return hovered_index, 1.0, hovered_index

        return hovered_index, progress, None


def main():
    """Standalone test: draw/move/erase, plus pinky-up-held smart correction
    on shapes, falling through to handwriting OCR (Phase 5 checkpoint)."""
    landmarker = create_hand_landmarker()
    state_machine = GestureStateMachine()
    palette_selector = PaletteSelector()

    cap = cv2.VideoCapture(0)

    cv2.namedWindow("Phase 5 - Handwriting Correction Test", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Phase 5 - Handwriting Correction Test", 960, 540)

    start_time = time.time()

    canvas = None
    active_color_index = 0
    palette_rects = None
    status_message = ""
    status_message_until = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if canvas is None:
            h, w = frame.shape[:2]
            canvas = Canvas(h, w)
            palette_rects = get_palette_rects(w)

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        timestamp_ms = int((time.time() - start_time) * 1000)
        result = landmarker.detect_for_video(mp_image, timestamp_ms)

        gesture_label = "NO HAND"
        hover_index, hover_progress = None, 0.0

        if result.hand_landmarks:
            hand_landmarks = result.hand_landmarks[0]
            positions = get_landmark_positions(hand_landmarks, frame.shape)
            draw_hand_landmarks(frame, positions)

            finger_states = get_finger_states(positions)
            gesture_label = state_machine.update(finger_states)
            index_tip = positions[INDEX_TIP_ID]

            is_drawing = gesture_label == DRAW
            canvas.update_stroke_tracking(
                is_drawing, index_tip, PALETTE_COLORS[active_color_index]
            )

            if gesture_label == DRAW:
                if canvas.prev_point is not None:
                    canvas.draw_line(canvas.prev_point, index_tip, PALETTE_COLORS[active_color_index])
                canvas.prev_point = index_tip

            elif gesture_label == MOVE:
                canvas.reset_stroke()
                hovered = get_hovered_swatch(index_tip, palette_rects)
                hover_index, hover_progress, selected = palette_selector.update(hovered)
                if selected is not None:
                    active_color_index = selected

            elif gesture_label == ERASE:
                canvas.reset_stroke()
                palm_center = get_palm_center(positions)
                canvas.erase_circle(palm_center)

            elif gesture_label == CORRECT:
                canvas.reset_stroke()
                points, color = canvas.get_last_stroke_points()
                if points:
                    shape_result, shape_debug = classify_stroke_debug(points, color)
                else:
                    shape_result, shape_debug = None, {"reason": "no completed stroke found"}

                # Print full diagnostics to console every time — this is what
                # you use to tune the thresholds in shape_correction.py
                print(f"[CORRECT][shape] debug_info = {shape_debug}")

                if shape_result is not None:
                    canvas.replace_last_stroke_with_shape(shape_result)
                    status_message = f"Corrected to {shape_result.shape_type}! ({shape_debug['reason']})"

                elif points:
                    # Phase 4 found no shape — fall through to Phase 5 (handwriting)
                    text_result, text_debug = recognize_character_debug(canvas.surface, points, color)
                    print(f"[CORRECT][ocr] debug_info = {text_debug}")

                    if text_result is not None:
                        canvas.replace_last_stroke_with_text(text_result)
                        status_message = f"Corrected to letter '{text_result.character}'!"
                    else:
                        status_message = f"No shape or letter recognized: {text_debug['reason']}"

                else:
                    status_message = f"No shape recognized: {shape_debug['reason']}"

                status_message_until = time.time() + 3.0

            else:
                canvas.reset_stroke()
        else:
            canvas.reset_stroke()
            canvas.update_stroke_tracking(False, None, None)

        display_frame = overlay_canvas(frame, canvas.surface)
        draw_palette(display_frame, palette_rects, active_color_index, hover_index, hover_progress)

        # Gesture-name indicator, top-left, below the palette
        cv2.putText(
            display_frame, gesture_label, (20, GESTURE_LABEL_Y),
            cv2.FONT_HERSHEY_SIMPLEX, GESTURE_LABEL_FONT_SCALE, TEXT_COLOR, 2, cv2.LINE_AA
        )

        # Status / debug message for correction attempts, shown for a few seconds
        if status_message and time.time() < status_message_until:
            cv2.putText(
                display_frame, status_message, (20, frame.shape[0] - 20),
                cv2.FONT_HERSHEY_SIMPLEX, STATUS_MESSAGE_FONT_SCALE, TEXT_COLOR, 2, cv2.LINE_AA
            )

        cv2.imshow("Phase 5 - Handwriting Correction Test", display_frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('c'):
            canvas.clear()

    cap.release()
    cv2.destroyAllWindows()
    landmarker.close()


if __name__ == "__main__":
    main()