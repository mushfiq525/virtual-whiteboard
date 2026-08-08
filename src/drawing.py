"""
Phase 3 — Core drawing logic
Phase 4 — Strokes tracked as objects (point lists), so a completed
stroke can be erased and replaced with a clean shape after correction.

Phase 4.5 — Selection transform:
  SELECT (thumb+index+middle, drag) -> selects any items overlapping the
                                        drag box. Releasing the drag LOCKS
                                        the selection in — it is NOT
                                        cleared by DRAW/MOVE/ERASE or
                                        anything else afterward. It only
                                        changes when a new SELECT drag
                                        starts, or on canvas clear. Starting
                                        a new SELECT drag without meaningfully
                                        dragging (a quick tap) is how you
                                        deselect — see had_existing_selection
                                        below.
  GRAB (fist, held while a selection is locked)
                                     -> drags the whole selected group,
                                        tracked via palm-center delta, with
                                        a small deadzone so per-frame
                                        landmark jitter doesn't accumulate
                                        into slow visible drift.
  PINCH (thumb+index close together, other 3 fingers curled, index kept
         straight — held while a selection is locked)
                                     -> scales the group in real time:
                                        spreading thumb/index apart grows
                                        it, bringing them together shrinks
                                        it, anchored on the selection's own
                                        center so it doesn't drift while
                                        resizing.
"""

import time

import cv2
import numpy as np
import mediapipe as mp

from hand_tracking import (
    create_hand_landmarker,
    get_landmark_positions,
    get_finger_states,
    get_pinch_metrics,
    draw_hand_landmarks,
)
from gesture_state import (
    GestureStateMachine, DRAW, MOVE, ERASE, CORRECT, CORRECT_CANDIDATE,
    SELECT, GRAB, PINCH,
)
from shape_correction import classify_stroke_debug, render_clean_shape

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
GESTURE_LABEL_Y = 100

PALM_LANDMARK_IDS = (0, 5, 9, 13, 17)
INDEX_TIP_ID = 8

MIN_SELECT_SIZE = 15   # minimum drag box width/height (px) to count as a real selection
MIN_SHAPE_SIZE = 10    # floor on width/height when scaling down, so items can't vanish to zero

# Real-time pinch-scale tuning
PINCH_FRAME_FACTOR_MIN = 0.85
PINCH_FRAME_FACTOR_MAX = 1.15
# Per-frame scale multiplier is clamped to this range. Without a clamp, a
# single noisy/jittery landmark frame (fingertip briefly mis-tracked) can
# produce a wildly wrong distance ratio and cause a visible "pop" in size.
# The clamp caps how much damage one bad frame can do; smooth pinching
# still accumulates a large real change over many frames just fine.

GRAB_DEADZONE_PX = 3
# Minimum palm-center movement (px) between frames before a GRAB drag is
# applied. Landmark estimates wobble a couple pixels frame-to-frame even
# when your hand is genuinely still; without this, that jitter accumulates
# into a slow, visible drift. Deltas below this are simply ignored — the
# reference point still updates, so real motion above the deadzone isn't
# delayed or lost, it just doesn't compound noise.

# UI text styling
TEXT_COLOR = (0, 59, 6)  # BGR for #063B00 (dark green)
GESTURE_LABEL_FONT_SCALE = 0.8
STATUS_MESSAGE_FONT_SCALE = 0.6
SELECT_BOX_COLOR = (255, 140, 0)           # BGR, orange — live drag rectangle while selecting
SELECTION_HIGHLIGHT_COLOR = (255, 255, 0)  # BGR, cyan — locked selection outline


class Canvas:
    """
    Persistent drawing surface, PLUS a list of "items" (completed strokes
    or corrected shapes) that lets us fully re-render the surface whenever
    items need to change — erase-and-replace on smart correction, or
    translate/scale on a selection transform — without losing everything
    else that was drawn before it.
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
                self._current_stroke = {"type": "stroke", "points": [], "color": color}
                self.items.append(self._current_stroke)
            self._current_stroke["points"].append(point)
        else:
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

    # --- selection support (Phase 4.5) ---

    def _item_bbox(self, item):
        """Bounding box (x, y, w, h) of any item, stroke or shape."""
        if item["type"] == "stroke":
            pts = item["points"]
            if not pts:
                return None
            contour = np.array(pts, dtype=np.int32).reshape((-1, 1, 2))
            return cv2.boundingRect(contour)
        elif item["type"] == "shape":
            return item["shape_result"].bounding_box
        return None

    def select_items_in_bbox(self, selection_bbox):
        """
        Indices of items whose own bbox is FULLY CONTAINED within the
        given selection rectangle. Containment rather than mere overlap:
        with nested shapes (e.g. a small circle drawn inside a big
        triangle), the circle's bbox always sits inside the triangle's
        bbox, so a plain overlap test would grab the triangle any time
        you tried to drag-select just the circle. Containment fixes this:
        dragging a box around only the circle doesn't fully enclose the
        (bigger) triangle's bbox, so the triangle is correctly left out;
        dragging a box around the whole triangle DOES fully enclose the
        circle's bbox too (since it's nested inside), so both come along
        together as one group, same as Canva/Figma-style rubber-band
        select.
        """
        sx, sy, sw, sh = selection_bbox
        selected = []
        for i, item in enumerate(self.items):
            ib = self._item_bbox(item)
            if ib is None:
                continue
            ix, iy, iw, ih = ib
            contained = (ix >= sx and iy >= sy and ix + iw <= sx + sw and iy + ih <= sy + sh)
            if contained:
                selected.append(i)
        return selected

    def combined_bbox(self, indices):
        """Union bbox of a set of item indices — used both for drawing the
        selection highlight and as the pinch-scale anchor."""
        boxes = [self._item_bbox(self.items[i]) for i in indices]
        boxes = [b for b in boxes if b is not None]
        if not boxes:
            return None
        x1 = min(b[0] for b in boxes)
        y1 = min(b[1] for b in boxes)
        x2 = max(b[0] + b[2] for b in boxes)
        y2 = max(b[1] + b[3] for b in boxes)
        return (x1, y1, x2 - x1, y2 - y1)

    def _clamp_point(self, pt):
        x, y = pt
        return (max(0, min(self.width - 1, x)), max(0, min(self.height - 1, y)))

    def translate_items(self, indices, dx, dy):
        """Shift every selected item by (dx, dy) — the GRAB drag."""
        for i in indices:
            item = self.items[i]
            if item["type"] == "stroke":
                item["points"] = [self._clamp_point((p[0] + dx, p[1] + dy)) for p in item["points"]]
            elif item["type"] == "shape":
                x, y, w, h = item["shape_result"].bounding_box
                new_x, new_y = self._clamp_point((x + dx, y + dy))
                item["shape_result"].bounding_box = (new_x, new_y, w, h)
        self._rerender()

    def scale_items(self, indices, factor, anchor):
        """
        Scale every selected item by `factor` around a fixed `anchor`
        point (the selection's own center), so the group grows/shrinks in
        place instead of drifting — called every frame during a PINCH,
        each time with that frame's small incremental factor.
        """
        ax, ay = anchor
        for i in indices:
            item = self.items[i]
            if item["type"] == "stroke":
                new_pts = []
                for (px, py) in item["points"]:
                    nx = int(ax + (px - ax) * factor)
                    ny = int(ay + (py - ay) * factor)
                    new_pts.append(self._clamp_point((nx, ny)))
                item["points"] = new_pts
            elif item["type"] == "shape":
                x, y, w, h = item["shape_result"].bounding_box
                nx = int(ax + (x - ax) * factor)
                ny = int(ay + (y - ay) * factor)
                nw = max(MIN_SHAPE_SIZE, int(w * factor))
                nh = max(MIN_SHAPE_SIZE, int(h * factor))
                cx, cy = self._clamp_point((nx, ny))
                item["shape_result"].bounding_box = (cx, cy, nw, nh)
        self._rerender()

    def _rerender(self):
        """Redraw the whole surface from scratch using self.items, in order.
        Used only on correction/select-transform events (occasional), not
        every frame — except during an active PINCH/GRAB, where it does
        run every frame; fine at whiteboard-canvas resolution/item counts."""
        self.surface = np.full((self.height, self.width, 3), self.bg_color, dtype=np.uint8)
        for item in self.items:
            if item["type"] == "stroke":
                pts = item["points"]
                for p1, p2 in zip(pts, pts[1:]):
                    cv2.line(self.surface, p1, p2, item["color"], STROKE_THICKNESS)
            elif item["type"] == "shape":
                render_clean_shape(self.surface, item["shape_result"], thickness=STROKE_THICKNESS)


class Selection:
    """
    Tracks which canvas items are currently locked in as "selected" after
    a SELECT drag, plus their combined bounding box — kept in sync after
    every move/scale so the on-screen highlight and the next pinch's
    anchor point stay accurate. Persists across DRAW/MOVE/ERASE; only
    cleared by starting a fresh SELECT drag or by canvas.clear().
    """

    def __init__(self):
        self.indices = []
        self.bbox = None  # (x, y, w, h), or None when empty

    def has_items(self):
        return len(self.indices) > 0

    def clear(self):
        self.indices = []
        self.bbox = None


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


def selection_bbox(pt1, pt2):
    """Normalize two arbitrary drag corners into (x, y, w, h)."""
    x1, y1 = pt1
    x2, y2 = pt2
    x, y = min(x1, x2), min(y1, y2)
    w, h = abs(x2 - x1), abs(y2 - y1)
    return (x, y, w, h)


def draw_selection_highlight(frame, bbox, color=SELECTION_HIGHLIGHT_COLOR):
    x, y, w, h = bbox
    cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
    for cx, cy in [(x, y), (x + w, y), (x, y + h), (x + w, y + h)]:
        cv2.rectangle(frame, (cx - 4, cy - 4), (cx + 4, cy + 4), color, -1)


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
    """Standalone test: draw/move/erase, pinky-hold shape correction,
    thumb+index+middle drag-select (locks on release), fist-drag to move
    the locked selection, and a thumb-index pinch to scale it live. A
    quick SELECT tap (no meaningful drag) while something is already
    locked deselects it."""
    landmarker = create_hand_landmarker()
    state_machine = GestureStateMachine()
    palette_selector = PaletteSelector()

    cap = cv2.VideoCapture(0)

    cv2.namedWindow("Phase 4.5 - Selection Transform Test", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Phase 4.5 - Selection Transform Test", 960, 540)

    start_time = time.time()

    canvas = None
    active_color_index = 0
    palette_rects = None
    status_message = ""
    status_message_until = 0

    # Selection-drag state
    was_selecting = False
    select_start = None
    select_current = None
    had_existing_selection = False  # was something already locked when this SELECT drag began?

    # Locked selection + live transform state
    selection = Selection()
    grab_prev_point = None
    pinch_prev_ratio = None

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
        is_selecting = False

        if result.hand_landmarks:
            hand_landmarks = result.hand_landmarks[0]
            positions = get_landmark_positions(hand_landmarks, frame.shape)
            draw_hand_landmarks(frame, positions)

            finger_states = get_finger_states(positions)
            pinch_metrics = get_pinch_metrics(positions)
            gesture_label = state_machine.update(finger_states, pinch_metrics)
            index_tip = positions[INDEX_TIP_ID]

            # Only GRAB/PINCH track continuous per-frame deltas; any other
            # gesture drops that tracking so a fresh GRAB/PINCH doesn't
            # jump using a stale reference point from before.
            if gesture_label != GRAB:
                grab_prev_point = None
            if gesture_label != PINCH:
                pinch_prev_ratio = None

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

                print(f"[CORRECT][shape] debug_info = {shape_debug}")

                if shape_result is not None:
                    canvas.replace_last_stroke_with_shape(shape_result)
                    status_message = f"Corrected to {shape_result.shape_type}! ({shape_debug['reason']})"
                else:
                    status_message = f"No shape recognized, left as-is: {shape_debug['reason']}"
                status_message_until = time.time() + 3.0

            elif gesture_label == SELECT:
                is_selecting = True
                if not was_selecting:
                    select_start = index_tip
                    had_existing_selection = selection.has_items()
                    selection.clear()  # starting a fresh drag drops the old locked selection
                select_current = index_tip

            elif gesture_label == GRAB:
                canvas.reset_stroke()
                if selection.has_items():
                    palm_center = get_palm_center(positions)
                    if grab_prev_point is not None:
                        dx = palm_center[0] - grab_prev_point[0]
                        dy = palm_center[1] - grab_prev_point[1]
                        if abs(dx) >= GRAB_DEADZONE_PX or abs(dy) >= GRAB_DEADZONE_PX:
                            canvas.translate_items(selection.indices, dx, dy)
                            selection.bbox = canvas.combined_bbox(selection.indices)
                    grab_prev_point = palm_center

            elif gesture_label == PINCH:
                canvas.reset_stroke()
                if selection.has_items() and pinch_metrics is not None:
                    current_ratio = pinch_metrics["distance_ratio"]
                    if pinch_prev_ratio is not None and pinch_prev_ratio > 0:
                        raw_factor = current_ratio / pinch_prev_ratio
                        factor = max(PINCH_FRAME_FACTOR_MIN, min(PINCH_FRAME_FACTOR_MAX, raw_factor))
                        if abs(factor - 1.0) > 1e-3:
                            bx, by, bw, bh = selection.bbox
                            anchor = (bx + bw // 2, by + bh // 2)
                            canvas.scale_items(selection.indices, factor, anchor)
                            selection.bbox = canvas.combined_bbox(selection.indices)
                    pinch_prev_ratio = current_ratio

            else:
                canvas.reset_stroke()

        else:
            canvas.reset_stroke()
            canvas.update_stroke_tracking(False, None, None)
            grab_prev_point = None
            pinch_prev_ratio = None

        # Selection release — build the actual locked item list here.
        if was_selecting and not is_selecting:
            if select_start is not None and select_current is not None:
                bbox = selection_bbox(select_start, select_current)
                if bbox[2] >= MIN_SELECT_SIZE and bbox[3] >= MIN_SELECT_SIZE:
                    indices = canvas.select_items_in_bbox(bbox)
                    if indices:
                        selection.indices = indices
                        selection.bbox = canvas.combined_bbox(indices)
                        status_message = f"Locked {len(indices)} item(s) — fist to move, pinch to resize"
                    else:
                        selection.clear()
                        status_message = "Nothing to select in that region"
                else:
                    # Drag was too small to be a real selection — but the
                    # old locked selection was already dropped the moment
                    # this SELECT gesture started, so a quick tap like
                    # this doubles as "deselect."
                    status_message = "Selection cleared" if had_existing_selection else "Selection too small"
                status_message_until = time.time() + 2.5
            select_start = None
            select_current = None

        was_selecting = is_selecting

        display_frame = overlay_canvas(frame, canvas.surface)
        draw_palette(display_frame, palette_rects, active_color_index, hover_index, hover_progress)

        # Live selection-box preview while dragging
        if is_selecting and select_start is not None and select_current is not None:
            x, y, w, h = selection_bbox(select_start, select_current)
            cv2.rectangle(display_frame, (x, y), (x + w, y + h), SELECT_BOX_COLOR, 2)

        # Locked selection highlight
        if selection.has_items() and selection.bbox is not None:
            draw_selection_highlight(display_frame, selection.bbox)

        cv2.putText(
            display_frame, gesture_label, (20, GESTURE_LABEL_Y),
            cv2.FONT_HERSHEY_SIMPLEX, GESTURE_LABEL_FONT_SCALE, TEXT_COLOR, 2, cv2.LINE_AA
        )

        if status_message and time.time() < status_message_until:
            cv2.putText(
                display_frame, status_message, (20, frame.shape[0] - 20),
                cv2.FONT_HERSHEY_SIMPLEX, STATUS_MESSAGE_FONT_SCALE, TEXT_COLOR, 2, cv2.LINE_AA
            )

        cv2.imshow("Phase 4.5 - Selection Transform Test", display_frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('c'):
            canvas.clear()
            selection.clear()

    cap.release()
    cv2.destroyAllWindows()
    landmarker.close()


if __name__ == "__main__":
    main()