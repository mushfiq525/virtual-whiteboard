"""
Phase 5 — Handwriting correction
Crops the bounding box of a stroke that Phase 4 failed to classify as a
shape, binarizes it for OCR, and asks Tesseract for a single character.
If Tesseract returns nothing usable, the caller should leave the original
stroke untouched rather than guess wrong (per the workflow doc, step 28).
"""

import re

import cv2
import numpy as np
import pytesseract

# ---------------------------------------------------------------------------
# Point this at your actual Tesseract install path. Only needed on Windows —
# on Mac/Linux the binary is already on PATH after brew/apt install.
# ---------------------------------------------------------------------------
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

MIN_STROKE_POINTS = 3          # even a simple letter needs a few points to crop meaningfully
CROP_PADDING = 25              # pixels of margin added around the raw stroke bounding box
TARGET_GLYPH_HEIGHT = 120      # upscale small strokes to roughly this height before OCR
BORDER_PADDING = 25            # extra white border added after binarizing (Tesseract likes margin)
OCR_CONFIG = "--psm 10"        # page-segmentation mode 10 = treat image as a single character

# Only accept a single letter or digit back — filters out OCR noise like
# stray punctuation, which is common on single-character recognition.
VALID_CHAR_PATTERN = re.compile(r"[A-Za-z0-9]")

FONT = cv2.FONT_HERSHEY_SIMPLEX


class TextResult:
    """Holds everything drawing.py needs to erase-and-redraw clean text,
    same role as ShapeResult in shape_correction.py."""

    def __init__(self, character, bounding_box, color):
        self.character = character        # the recognized single character
        self.bounding_box = bounding_box   # (x, y, w, h) of the ORIGINAL stroke
        self.color = color


def _stroke_bounding_box(stroke_points, padding=CROP_PADDING):
    xs = [p[0] for p in stroke_points]
    ys = [p[1] for p in stroke_points]
    x_min, x_max = min(xs) - padding, max(xs) + padding
    y_min, y_max = min(ys) - padding, max(ys) + padding
    return x_min, y_min, x_max, y_max


def _binarize_for_ocr(gray_crop):
    """
    Otsu-threshold the crop, then make sure the result is black strokes on
    a white background (Tesseract works best this way). Otsu doesn't know
    which class is "ink" vs "background," so we check the border pixels —
    the background should dominate the border — and invert if the border
    came out black instead of white.
    """
    _, binary = cv2.threshold(gray_crop, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    border_pixels = np.concatenate([
        binary[0, :], binary[-1, :], binary[:, 0], binary[:, -1]
    ])
    border_is_mostly_black = np.mean(border_pixels) < 127
    if border_is_mostly_black:
        binary = cv2.bitwise_not(binary)

    return binary


def _prepare_crop(canvas_surface, bbox):
    """Crop from the canvas, convert to grayscale, binarize, upscale small
    strokes, and pad with a white border — each step makes OCR more reliable
    on a single hand-drawn character."""
    x_min, y_min, x_max, y_max = bbox
    h, w = canvas_surface.shape[:2]
    x_min, y_min = max(0, x_min), max(0, y_min)
    x_max, y_max = min(w, x_max), min(h, y_max)

    if x_max <= x_min or y_max <= y_min:
        return None

    crop = canvas_surface[y_min:y_max, x_min:x_max]
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    binary = _binarize_for_ocr(gray)

    # Upscale small glyphs — Tesseract does noticeably better on larger text
    crop_h = binary.shape[0]
    if crop_h < TARGET_GLYPH_HEIGHT:
        scale = TARGET_GLYPH_HEIGHT / crop_h
        binary = cv2.resize(
            binary, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC
        )

    # White border margin around the glyph
    binary = cv2.copyMakeBorder(
        binary, BORDER_PADDING, BORDER_PADDING, BORDER_PADDING, BORDER_PADDING,
        cv2.BORDER_CONSTANT, value=255
    )
    return binary


def recognize_character_debug(canvas_surface, stroke_points, color):
    """
    Attempt to OCR a completed stroke as a single character.

    Returns:
        (TextResult | None, debug_info_dict)
    """
    debug_info = {
        "num_points": 0 if stroke_points is None else len(stroke_points),
        "raw_ocr_output": None,
        "reason": None,
    }

    if stroke_points is None or len(stroke_points) < MIN_STROKE_POINTS:
        debug_info["reason"] = f"too few points (<{MIN_STROKE_POINTS})"
        return None, debug_info

    x_min, y_min, x_max, y_max = _stroke_bounding_box(stroke_points)
    ocr_crop = _prepare_crop(canvas_surface, (x_min, y_min, x_max, y_max))

    if ocr_crop is None:
        debug_info["reason"] = "empty/invalid crop region"
        return None, debug_info

    raw_output = pytesseract.image_to_string(ocr_crop, config=OCR_CONFIG)
    debug_info["raw_ocr_output"] = repr(raw_output)

    cleaned = raw_output.strip()
    match = VALID_CHAR_PATTERN.search(cleaned)

    if not match:
        debug_info["reason"] = f"no valid single character in OCR output {repr(raw_output)}"
        return None, debug_info

    character = match.group(0)
    debug_info["reason"] = f"recognized '{character}'"

    original_bbox = (x_min, y_min, x_max - x_min, y_max - y_min)
    return TextResult(character, original_bbox, color), debug_info


def render_clean_text(surface, text_result, thickness=3):
    """
    Draw the recognized character cleanly onto `surface`, sized to roughly
    match the original stroke's bounding box (Phase 5, step 28).
    """
    x, y, w, h = text_result.bounding_box
    color = text_result.color
    char = text_result.character

    # Find a font scale whose rendered height roughly matches the stroke's
    # original bounding-box height, so the corrected letter isn't jarringly
    # different in size from what the user actually drew.
    target_height = max(h, 10)
    font_scale = 1.0
    (_, text_h), _ = cv2.getTextSize(char, FONT, font_scale, thickness)
    if text_h > 0:
        font_scale = target_height / text_h

    (text_w, text_h), baseline = cv2.getTextSize(char, FONT, font_scale, thickness)
    origin = (x + (w - text_w) // 2, y + text_h)  # baseline position, roughly centered

    cv2.putText(surface, char, origin, FONT, font_scale, color, thickness, cv2.LINE_AA)