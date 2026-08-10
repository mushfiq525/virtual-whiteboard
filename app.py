"""
Phase 7 — Streamlit + streamlit-webrtc wrapper.

This is a thin adapter, not a reimplementation: every class and helper
function used below (Canvas, Selection, GestureStateMachine, FingertipSmoother,
PaletteSelector, HoldButton, perform_undo, draw_*, get_*, ...) is imported
straight from src/hand_tracking.py, src/gesture_state.py,
src/shape_correction.py, and src/drawing.py — nothing is duplicated here.

`WhiteboardProcessor.recv()` below is line-for-line the same pipeline as
the `while True` loop body in `src/drawing.py`'s `main()` (hand tracking ->
gesture state -> drawing/correction/selection/fill -> overlay), just with
every loop-local variable re-homed as a `self.` attribute, since
streamlit-webrtc calls `recv()` once per frame on its own background video
thread instead of us owning the loop.

Two differences from the local `src/drawing.py` test harness, both because
there's no OpenCV window/keyboard here:
  - The `'c'` (clear) and `'z'` (undo) keyboard shortcuts become the two
    Streamlit buttons below. Since Streamlit's button callbacks run on a
    different thread than `recv()`, a small lock + one-shot flag hands
    each click over safely instead of touching canvas/selection state
    directly from the wrong thread.
  - No cv2.imshow/waitKey loop — Streamlit owns rendering the video via
    `webrtc_streamer`.
"""

import os
import sys
import threading
import time
from collections import deque

# src/ files import each other as plain top-level modules (e.g.
# `from hand_tracking import ...`), so src/ needs to be on sys.path before
# any of them are imported — same requirement Phase 0's project-structure
# note describes for running `python src/drawing.py` directly.
SRC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

import av
import cv2
import mediapipe as mp
import streamlit as st
from streamlit_webrtc import RTCConfiguration, VideoProcessorBase, webrtc_streamer

from hand_tracking import (
    create_hand_landmarker,
    get_landmark_positions,
    get_finger_states,
    get_pinch_metrics,
    draw_hand_landmarks,
)
from gesture_state import (
    GestureStateMachine, DRAW, MOVE, ERASE, CORRECT,
    SELECT, GRAB, PINCH, FILL_CANDIDATE, FILL,
)
from shape_correction import classify_stroke_debug
from drawing import (
    Canvas, Selection, FingertipSmoother, PaletteSelector, HoldButton,
    perform_undo,
    get_palette_rects, get_undo_button_rect, draw_undo_button, draw_palette,
    get_hovered_swatch, get_palm_center, overlay_canvas, selection_bbox,
    draw_selection_highlight, point_in_rect,
    PALETTE_COLORS, INDEX_TIP_ID, GESTURE_LABEL_Y, TEXT_COLOR,
    GESTURE_LABEL_FONT_SCALE, STATUS_MESSAGE_FONT_SCALE, SELECT_BOX_COLOR,
    MIN_SELECT_SIZE, GRAB_DEADZONE_PX, PINCH_FRAME_FACTOR_MIN,
    PINCH_FRAME_FACTOR_MAX, FPS_CHECK_WINDOW, FPS_SKIP_THRESHOLD,
    FPS_RESUME_THRESHOLD,
)

DETECTION_MAX_WIDTH = 640
# Cap the width of the frame actually fed to MediaPipe's detector.
# Landmark output is normalized (0-1) regardless of input resolution, so
# detection can run on a smaller downscaled copy while every position is
# still mapped back onto the full-resolution frame for drawing — this
# just cuts detection cost when the webcam negotiates a high-res stream
# (a real contributor to the app stalling under load), with no loss of
# overlay/drawing quality.


class WhiteboardProcessor(VideoProcessorBase):
    """Owns one MediaPipe landmarker + one whiteboard Canvas for the
    lifetime of a single browser session's video stream."""

    def __init__(self):
        self.landmarker = create_hand_landmarker()
        self.state_machine = GestureStateMachine()
        self.palette_selector = PaletteSelector()
        self.undo_button = HoldButton()

        self.start_time = time.time()

        # Canvas/palette are sized off the first real frame, same as
        # drawing.py's `if canvas is None:` lazy-init.
        self.canvas = None
        self.active_color_index = 0
        self.palette_rects = None
        self.undo_rect = None
        self.status_message = ""
        self.status_message_until = 0

        # Selection-drag state
        self.was_selecting = False
        self.select_start = None
        self.select_current = None
        self.had_existing_selection = False

        # Locked selection + live transform state
        self.selection = Selection()
        self.grab_prev_point = None
        self.pinch_prev_ratio = None

        # Live fingertip smoothing while DRAW is active (Phase 6, step 37)
        self.draw_smoother = FingertipSmoother()

        # FPS-adaptive tracking (Phase 6, step 38)
        self.frame_times = deque(maxlen=FPS_CHECK_WINDOW)
        self.skip_tracking = False
        self.current_fps = 0.0
        self.frame_index = 0
        self.last_landmark_result = None

        # Cross-thread bridge for the Streamlit Clear/Undo buttons — see
        # module docstring. `recv()` consumes and clears these flags at
        # the top of every frame.
        self._lock = threading.Lock()
        self._pending_clear = False
        self._pending_undo = False

        # Whether to horizontally flip the incoming frame before any
        # processing. Some browsers/OSes mirror a phone's front-facing
        # camera at the capture level (so the raw track itself is already
        # flipped), others don't — Iriun on desktop never does. Rather
        # than guess per-device, this is a plain user-facing toggle (see
        # the sidebar checkbox below) driven from the main Streamlit
        # thread the same way Clear/Undo are.
        self.mirror = False

    # --- called from Streamlit's main thread, by the buttons below ---
    def request_clear(self):
        with self._lock:
            self._pending_clear = True

    def request_undo(self):
        with self._lock:
            self._pending_undo = True

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        try:
            return self._process_frame(img)
        except Exception as exc:
            # A crash inside frame processing must never take the whole
            # video track down with it — without this, streamlit-webrtc
            # simply stops delivering frames after the first unhandled
            # exception, which looks exactly like a freeze from the
            # browser side. Log it and fall back to the plain camera
            # frame for this one tick; the next frame gets a fresh try.
            print(f"[WhiteboardProcessor.recv] frame processing error: {exc!r}")
            return av.VideoFrame.from_ndarray(img, format="bgr24")

    def _process_frame(self, img):
        loop_start = time.time()

        if self.mirror:
            # Flip once, up front, before anything downstream (hand
            # tracking, drawing, canvas) ever sees the frame — so the
            # mirrored view stays internally consistent instead of only
            # being mirrored cosmetically after the fact.
            img = cv2.flip(img, 1)

        h, w = img.shape[:2]
        if self.canvas is None or (self.canvas.height, self.canvas.width) != (h, w):
            # (Re)build the canvas whenever the incoming frame's resolution
            # doesn't match what it was last sized for — not just on the
            # very first frame. The browser can renegotiate the webcam to
            # a different resolution (e.g. entering/exiting fullscreen),
            # and every array-indexing operation on the old canvas would
            # otherwise be wrong for the new frame size, throwing on every
            # subsequent frame — which is what was freezing the app.
            # Existing drawing is lost on a resolution change (there's no
            # good way to keep it perfectly aligned across a resize), but
            # that's a far better trade than the whole stream locking up.
            self.canvas = Canvas(h, w)
            self.palette_rects = get_palette_rects(w)
            self.undo_rect = get_undo_button_rect(self.palette_rects)
            self.selection.clear()

        with self._lock:
            do_clear, self._pending_clear = self._pending_clear, False
            do_undo, self._pending_undo = self._pending_undo, False
        if do_clear:
            self.canvas.clear()
            self.selection.clear()
        if do_undo:
            self.status_message = perform_undo(self.canvas, self.selection)
            self.status_message_until = time.time() + 2.0

        self.frame_index += 1
        run_tracking = True
        if (self.skip_tracking and self.frame_index % 2 == 0
                and self.last_landmark_result is not None):
            run_tracking = False

        if run_tracking:
            # MediaPipe detection cost scales with input resolution, but
            # its landmark output is normalized (0-1) regardless of input
            # size — so detection can run on a smaller downscaled copy
            # while positions are still mapped back onto the full-size
            # `img` for drawing, with no loss of overlay quality.
            detect_img = img
            if w > DETECTION_MAX_WIDTH:
                scale = DETECTION_MAX_WIDTH / w
                detect_img = cv2.resize(img, (DETECTION_MAX_WIDTH, int(h * scale)))
            rgb_frame = cv2.cvtColor(detect_img, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            timestamp_ms = int((time.time() - self.start_time) * 1000)
            result = self.landmarker.detect_for_video(mp_image, timestamp_ms)
            self.last_landmark_result = result
        else:
            result = self.last_landmark_result

        gesture_label = "NO HAND"
        hover_index, hover_progress = None, 0.0
        undo_hover_progress = 0.0
        is_selecting = False
        hand_positions = None

        if result is not None and result.hand_landmarks:
            hand_landmarks = result.hand_landmarks[0]
            positions = get_landmark_positions(hand_landmarks, img.shape)
            hand_positions = positions

            finger_states = get_finger_states(positions)
            pinch_metrics = get_pinch_metrics(positions)
            gesture_label = self.state_machine.update(finger_states, pinch_metrics)
            index_tip = positions[INDEX_TIP_ID]

            if gesture_label != GRAB:
                self.grab_prev_point = None
            if gesture_label != PINCH:
                self.pinch_prev_ratio = None

            is_drawing = gesture_label == DRAW
            if is_drawing:
                draw_point = self.draw_smoother.smooth(index_tip)
            else:
                self.draw_smoother.reset()
                draw_point = index_tip
            self.canvas.update_stroke_tracking(
                is_drawing, draw_point, PALETTE_COLORS[self.active_color_index]
            )

            if gesture_label == DRAW:
                if self.canvas.prev_point is not None:
                    self.canvas.draw_line(
                        self.canvas.prev_point, draw_point,
                        PALETTE_COLORS[self.active_color_index]
                    )
                self.canvas.prev_point = draw_point

            elif gesture_label == MOVE:
                self.canvas.reset_stroke()
                hovered = get_hovered_swatch(index_tip, self.palette_rects)
                hover_index, hover_progress, selected = self.palette_selector.update(hovered)
                if selected is not None:
                    self.active_color_index = selected

                is_hovering_undo = point_in_rect(index_tip, self.undo_rect)
                undo_hover_progress, undo_triggered = self.undo_button.update(is_hovering_undo)
                if undo_triggered:
                    self.status_message = perform_undo(self.canvas, self.selection)
                    self.status_message_until = time.time() + 2.0

            elif gesture_label == ERASE:
                self.canvas.reset_stroke()
                palm_center = get_palm_center(positions)
                self.canvas.erase_circle(palm_center)

            elif gesture_label == CORRECT:
                self.canvas.reset_stroke()
                points, color = self.canvas.get_last_stroke_points()
                if points:
                    shape_result, shape_debug = classify_stroke_debug(points, color)
                else:
                    shape_result, shape_debug = None, {"reason": "no completed stroke found"}

                if shape_result is not None:
                    self.canvas.replace_last_stroke_with_shape(shape_result)
                    self.status_message = (
                        f"Corrected to {shape_result.shape_type}! ({shape_debug['reason']})"
                    )
                else:
                    self.status_message = f"No shape recognized, left as-is: {shape_debug['reason']}"
                self.status_message_until = time.time() + 3.0

            elif gesture_label == SELECT:
                is_selecting = True
                if not self.was_selecting:
                    self.select_start = index_tip
                    self.had_existing_selection = self.selection.has_items()
                    self.selection.clear()
                self.select_current = index_tip

            elif gesture_label == GRAB:
                self.canvas.reset_stroke()
                if self.selection.has_items():
                    palm_center = get_palm_center(positions)
                    if self.grab_prev_point is not None:
                        dx = palm_center[0] - self.grab_prev_point[0]
                        dy = palm_center[1] - self.grab_prev_point[1]
                        if abs(dx) >= GRAB_DEADZONE_PX or abs(dy) >= GRAB_DEADZONE_PX:
                            self.canvas.translate_items(self.selection.indices, dx, dy)
                            self.selection.bbox = self.canvas.combined_bbox(self.selection.indices)
                    self.grab_prev_point = palm_center

            elif gesture_label == PINCH:
                self.canvas.reset_stroke()
                if self.selection.has_items() and pinch_metrics is not None:
                    current_ratio = pinch_metrics["distance_ratio"]
                    if self.pinch_prev_ratio is not None and self.pinch_prev_ratio > 0:
                        raw_factor = current_ratio / self.pinch_prev_ratio
                        factor = max(PINCH_FRAME_FACTOR_MIN, min(PINCH_FRAME_FACTOR_MAX, raw_factor))
                        if abs(factor - 1.0) > 1e-3:
                            bx, by, bw, bh = self.selection.bbox
                            anchor = (bx + bw // 2, by + bh // 2)
                            self.canvas.scale_items(self.selection.indices, factor, anchor)
                            self.selection.bbox = self.canvas.combined_bbox(self.selection.indices)
                    self.pinch_prev_ratio = current_ratio

            elif gesture_label == FILL_CANDIDATE:
                self.canvas.reset_stroke()

            elif gesture_label == FILL:
                self.canvas.reset_stroke()
                filled, fill_area = self.canvas.fill_at(
                    index_tip, PALETTE_COLORS[self.active_color_index]
                )
                if filled:
                    self.status_message = "Filled!"
                else:
                    self.status_message = "Can't fill here — area not fully closed, or already filled"
                self.status_message_until = time.time() + 2.5

            else:
                self.canvas.reset_stroke()

        else:
            self.canvas.reset_stroke()
            self.canvas.update_stroke_tracking(False, None, None)
            self.draw_smoother.reset()
            self.grab_prev_point = None
            self.pinch_prev_ratio = None

        # Selection release — build the actual locked item list here.
        if self.was_selecting and not is_selecting:
            if self.select_start is not None and self.select_current is not None:
                bbox = selection_bbox(self.select_start, self.select_current)
                if bbox[2] >= MIN_SELECT_SIZE and bbox[3] >= MIN_SELECT_SIZE:
                    indices = self.canvas.select_items_in_bbox(bbox)
                    if indices:
                        self.selection.indices = indices
                        self.selection.bbox = self.canvas.combined_bbox(indices)
                        self.status_message = f"Locked {len(indices)} item(s) — fist to move, pinch to resize"
                    else:
                        self.selection.clear()
                        self.status_message = "Nothing to select in that region"
                else:
                    self.status_message = (
                        "Selection cleared" if self.had_existing_selection else "Selection too small"
                    )
                self.status_message_until = time.time() + 2.5
            self.select_start = None
            self.select_current = None

        self.was_selecting = is_selecting

        display_frame = overlay_canvas(img, self.canvas.surface)
        draw_palette(display_frame, self.palette_rects, self.active_color_index, hover_index, hover_progress)
        if self.undo_rect is not None:
            draw_undo_button(display_frame, self.undo_rect, undo_hover_progress)

        if is_selecting and self.select_start is not None and self.select_current is not None:
            x, y, w, h = selection_bbox(self.select_start, self.select_current)
            cv2.rectangle(display_frame, (x, y), (x + w, y + h), SELECT_BOX_COLOR, 2)

        if self.selection.has_items() and self.selection.bbox is not None:
            draw_selection_highlight(display_frame, self.selection.bbox)

        if hand_positions is not None:
            draw_hand_landmarks(display_frame, hand_positions)

        cv2.putText(
            display_frame, gesture_label, (20, GESTURE_LABEL_Y),
            cv2.FONT_HERSHEY_SIMPLEX, GESTURE_LABEL_FONT_SCALE, TEXT_COLOR, 2, cv2.LINE_AA
        )

        if self.status_message and time.time() < self.status_message_until:
            cv2.putText(
                display_frame, self.status_message, (20, img.shape[0] - 20),
                cv2.FONT_HERSHEY_SIMPLEX, STATUS_MESSAGE_FONT_SCALE, TEXT_COLOR, 2, cv2.LINE_AA
            )

        fps_text = f"FPS: {self.current_fps:.0f}" + (" (tracking every 2nd frame)" if self.skip_tracking else "")
        fps_text_size = cv2.getTextSize(fps_text, cv2.FONT_HERSHEY_SIMPLEX, STATUS_MESSAGE_FONT_SCALE, 1)[0]
        cv2.putText(
            display_frame, fps_text, (img.shape[1] - fps_text_size[0] - 20, 30),
            cv2.FONT_HERSHEY_SIMPLEX, STATUS_MESSAGE_FONT_SCALE, TEXT_COLOR, 1, cv2.LINE_AA
        )

        self.frame_times.append(time.time() - loop_start)
        if len(self.frame_times) == self.frame_times.maxlen:
            self.current_fps = len(self.frame_times) / sum(self.frame_times)
            if self.current_fps < FPS_SKIP_THRESHOLD:
                self.skip_tracking = True
            elif self.current_fps > FPS_RESUME_THRESHOLD:
                self.skip_tracking = False

        return av.VideoFrame.from_ndarray(display_frame, format="bgr24")


# ---------------------------------------------------------------------------
# Streamlit page
# ---------------------------------------------------------------------------
st.set_page_config(page_title="Gesture Whiteboard", layout="wide")
st.title("Gesture-Controlled Smart Whiteboard")
st.caption(
    "Draw, auto-correct shapes, select, move, scale, and fill — all with "
    "hand gestures. Click **START** below and allow camera access."
)

def _build_rtc_configuration():
    """STUN alone only gets a direct peer-to-peer connection when both
    sides are behind simple NAT — that's true on a home WiFi/PC setup,
    but mobile data (carrier-grade NAT) and many other WiFi networks
    block it outright, and Streamlit Community Cloud's own network can
    block direct WebRTC media too. A TURN server relays the media
    instead of negotiating a direct link, so it works in those cases.

    TURN credentials are pulled from st.secrets so nothing is hardcoded
    here. Add a [turn] section to .streamlit/secrets.toml (locally) or
    the app's "Secrets" settings (on Streamlit Community Cloud):

        TURN_URL = "turn:your-turn-host:3478"
        TURN_USERNAME = "..."
        TURN_CREDENTIAL = "..."

    e.g. from a free provider like Open Relay Project / metered.ca, or
    Twilio's Network Traversal Service. If secrets aren't configured,
    this quietly falls back to STUN-only so local development still
    works — but mobile-data/other-network connections will keep
    timing out until TURN is added.
    """
    ice_servers = [{"urls": ["stun:stun.l.google.com:19302"]}]

    try:
        turn_url = st.secrets.get("TURN_URL")
        turn_username = st.secrets.get("TURN_USERNAME")
        turn_credential = st.secrets.get("TURN_CREDENTIAL")
    except Exception:
        turn_url = turn_username = turn_credential = None

    if turn_url and turn_username and turn_credential:
        ice_servers.append({
            "urls": [turn_url],
            "username": turn_username,
            "credential": turn_credential,
        })

    return RTCConfiguration({"iceServers": ice_servers})


RTC_CONFIGURATION = _build_rtc_configuration()

webrtc_ctx = webrtc_streamer(
    key="gesture-whiteboard",
    video_processor_factory=WhiteboardProcessor,
    rtc_configuration=RTC_CONFIGURATION,
    media_stream_constraints={
        # An explicit, moderate ideal resolution rather than bare `True` —
        # `True` lets the browser negotiate whatever the webcam offers by
        # default, which for a phone-as-webcam (Iriun) can be a high-res
        # stream that's both slow to process and, once stretched to fill
        # a wide page column, makes every fixed-pixel overlay (palette,
        # legend) look oversized and blurry. This keeps the frame (and
        # therefore the whiteboard UI) at a predictable, lighter size.
        "video": {"width": {"ideal": 960}, "height": {"ideal": 540}},
        "audio": False,
    },
    video_html_attrs={
        # Keep the video letterboxed within its box instead of being
        # cropped/stretched — the crop artifact after exiting fullscreen
        # was CSS scaling the video element to a fixed box while the
        # frame itself kept its own aspect ratio underneath.
        "style": {"width": "100%", "max-width": "960px", "object-fit": "contain"},
        "autoPlay": True,
        "muted": True,
    },
    async_processing=True,
)

mirror = st.checkbox(
    "🪞 Mirror camera",
    value=False,
    help=(
        "Some devices/browsers already mirror the front camera before it "
        "reaches this app, others don't — flip this on if drawing feels "
        "left/right reversed."
    ),
)
if webrtc_ctx.video_processor:
    webrtc_ctx.video_processor.mirror = mirror

st.write("")
col1, col2 = st.columns(2)
with col1:
    if st.button("🧹 Clear canvas", use_container_width=True):
        if webrtc_ctx.video_processor:
            webrtc_ctx.video_processor.request_clear()
with col2:
    if st.button("↩️ Undo last action", use_container_width=True):
        if webrtc_ctx.video_processor:
            webrtc_ctx.video_processor.request_undo()

with st.expander("Gesture legend", expanded=False):
    st.markdown(
        """
| Gesture | Action |
|---|---|
| Index finger up only | Draw |
| Index + middle finger up | Move / navigate (no drawing) |
| Open palm (all 5 up) | Erase (area under palm) |
| Pinky only, held ~1s | Smart shape correction on last stroke |
| Thumb + index + middle, drag | Select — draws a box, locks on release |
| Fist | Move the locked selection |
| Pinch (thumb + index, others curled) | Scale the locked selection live |
| Index + middle + ring, held ~1s | Paint-bucket fill |
        """
    )
