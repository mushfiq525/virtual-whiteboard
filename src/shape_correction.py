"""
Phase 4 — Smart shape correction
Takes a completed freehand stroke (list of (x, y) points), smooths it to
remove hand-tremor jitter, computes its convex hull (this is what makes
circularity/vertex detection robust to jittery real-world strokes —
inward zigzags get discarded instead of inflating the perimeter), then
classifies the hull as triangle / rectangle / circle. If the stroke
doesn't look like a closed, convex shape at all, returns None so the
caller can fall through to Phase 5 (handwriting correction).
"""

import math

import cv2
import numpy as np

MIN_STROKE_POINTS = 5             # strokes shorter than this can't be meaningfully classified
APPROX_POLY_EPSILON_RATIO = 0.02  # fraction of hull perimeter, standard starting value
RECT_ANGLE_TOLERANCE_DEG = 20     # how far from 90 degrees an angle can be and still count
CIRCULARITY_THRESHOLD = 0.75      # 4*pi*Area / Perimeter^2, 1.0 = perfect circle.
                                   # A hull-based square lands around 0.78-0.85, but squares
                                   # are caught earlier by the 4-vertex/rectangle check, so
                                   # this only needs to separate "circle" from "everything
                                   # else that fell through" — 0.75 is safely below real
                                   # circle hulls (~0.9+) while still being a real bar.
SMOOTHING_WINDOW = 7               # moving-average window size for de-jittering strokes

# cv2.convexHull() always treats the point list as a CLOSED loop when
# computing perimeter/area/vertex-count, even if the stroke was never
# actually closed. Real shapes (circle/triangle/rectangle) are almost
# always drawn as a closed loop: the fingertip ends up back near where it
# started. Most letters don't. So if start and end points are far apart
# relative to the stroke's own size, treat it as an open path — a
# handwriting candidate — before it ever reaches shape classification.
CLOSURE_RATIO_THRESHOLD = 0.35    # start-to-end gap as a fraction of the
                                   # bounding-box diagonal; above this, the
                                   # stroke is considered "open"

# CLOSURE alone isn't enough: letters like P, R, B, D have a loop that
# *does* close near the starting point, but also have a tail or stem the
# convex hull bridges over with a big chunk of area that was never
# actually drawn. SOLIDITY catches this: it's the ratio of the real
# (non-convex) traced area to the hull's area. A genuine circle/triangle/
# rectangle is essentially convex already, so its traced area nearly
# fills its own hull. A loopy letter leaves a large gap between what was
# drawn and what the hull bridges over, so solidity drops well below this.
SOLIDITY_THRESHOLD = 0.80


class ShapeResult:
    """Holds everything drawing.py needs to erase-and-redraw a clean shape."""

    def __init__(self, shape_type, bounding_box, color):
        self.shape_type = shape_type      # "triangle" | "rectangle" | "circle"
        self.bounding_box = bounding_box  # (x, y, w, h)
        self.color = color


def _smooth_points(points, window=SMOOTHING_WINDOW):
    """
    Moving-average smoothing over the raw fingertip trail. Reduces high-
    frequency jitter before we even get to the convex hull step, so single
    wild outlier points don't distort the hull shape.
    """
    if len(points) < window:
        return points
    smoothed = []
    half = window // 2
    for i in range(len(points)):
        lo = max(0, i - half)
        hi = min(len(points), i + half + 1)
        chunk = points[lo:hi]
        avg_x = sum(p[0] for p in chunk) / len(chunk)
        avg_y = sum(p[1] for p in chunk) / len(chunk)
        smoothed.append((int(avg_x), int(avg_y)))
    return smoothed


def _stroke_to_contour(stroke_points):
    """Wrap a raw point list as the Nx1x2 int32 array cv2 contour functions expect."""
    return np.array(stroke_points, dtype=np.int32).reshape((-1, 1, 2))


def _angle_deg(p0, p1, p2):
    """Interior angle at p1, formed by segments p1->p0 and p1->p2, in degrees."""
    v1 = np.array(p0) - np.array(p1)
    v2 = np.array(p2) - np.array(p1)
    denom = (np.linalg.norm(v1) * np.linalg.norm(v2))
    if denom == 0:
        return 0.0
    cos_angle = np.dot(v1, v2) / denom
    cos_angle = np.clip(cos_angle, -1.0, 1.0)
    return math.degrees(math.acos(cos_angle))


def _looks_like_rectangle(approx_points):
    """True if all 4 interior angles of a 4-point polygon are close to 90 degrees."""
    if len(approx_points) != 4:
        return False
    pts = [tuple(p) for p in approx_points]
    for i in range(4):
        p0 = pts[i - 1]
        p1 = pts[i]
        p2 = pts[(i + 1) % 4]
        angle = _angle_deg(p0, p1, p2)
        if abs(angle - 90) > RECT_ANGLE_TOLERANCE_DEG:
            return False
    return True


def _closure_ratio(stroke_points, w, h):
    """
    How far apart the stroke's start and end points are, relative to the
    stroke's own bounding-box diagonal. Near 0 = stroke was closed (loop
    ends where it began). Large = open path, like most handwriting.
    """
    diagonal = math.hypot(w, h)
    if diagonal == 0:
        return 1.0
    start_point = stroke_points[0]
    end_point = stroke_points[-1]
    gap = math.hypot(end_point[0] - start_point[0], end_point[1] - start_point[1])
    return gap / diagonal


def classify_stroke_debug(stroke_points, color):
    """
    Classify a completed stroke's point list as a recognized shape, and
    also return a dict of the raw numbers used to make the decision
    (vertex count, circularity, closure ratio, solidity, etc.) so you can
    tune the thresholds at the top of this file against what your own
    strokes actually produce.

    Returns:
        (ShapeResult | None, debug_info_dict)
    """
    debug_info = {
        "num_points": 0 if stroke_points is None else len(stroke_points),
        "num_vertices": None,
        "circularity": None,
        "closure_ratio": None,
        "solidity": None,
        "rect_check_passed": None,
        "reason": None,
    }

    if stroke_points is None or len(stroke_points) < MIN_STROKE_POINTS:
        debug_info["reason"] = f"too few points (<{MIN_STROKE_POINTS})"
        return None, debug_info

    smoothed_points = _smooth_points(stroke_points)
    contour = _stroke_to_contour(smoothed_points)

    # Bounding box comes from the raw contour (same result as from the hull,
    # since both share the same extreme points) — used for sizing the
    # redrawn clean shape, and for the closure check below.
    x, y, w, h = cv2.boundingRect(contour)
    bounding_box = (x, y, w, h)

    # --- Closure check: reject clearly open paths (most letters) before
    # they ever get a chance to coincidentally hull down into something
    # shape-like ---
    closure_ratio = _closure_ratio(smoothed_points, w, h)
    debug_info["closure_ratio"] = round(closure_ratio, 3)
    if closure_ratio > CLOSURE_RATIO_THRESHOLD:
        debug_info["reason"] = (
            f"stroke not closed (closure_ratio={closure_ratio:.3f} > "
            f"{CLOSURE_RATIO_THRESHOLD}) -> looks like handwriting, not a shape"
        )
        return None, debug_info

    # Convex hull: this is the key fix for jitter. A jittery hand-drawn
    # stroke zigzags inward and outward around the "true" shape it was
    # meant to trace. Those inward zigzags add real distance to the
    # perimeter without meaningfully changing the enclosed area, which
    # tanks circularity and adds phantom vertices. The convex hull discards
    # all the inward noise and keeps just the outer boundary — much closer
    # to what the user actually intended to draw.
    hull = cv2.convexHull(contour)

    perimeter = cv2.arcLength(hull, closed=True)
    if perimeter == 0:
        debug_info["reason"] = "zero perimeter"
        return None, debug_info

    # --- Solidity check: reject strokes that closed their loop (passed
    # the check above) but still aren't actually convex — e.g. a letter
    # like P, R, B, D, whose tail/stem makes the convex hull bridge over a
    # large chunk of area that was never actually drawn. Real shapes are
    # already close to convex, so their traced area nearly fills their
    # own hull; loopy letters leave a big gap. ---
    hull_area = cv2.contourArea(hull)
    raw_area = abs(cv2.contourArea(contour))
    solidity = (raw_area / hull_area) if hull_area > 0 else 0.0
    debug_info["solidity"] = round(solidity, 3)
    if solidity < SOLIDITY_THRESHOLD:
        debug_info["reason"] = (
            f"low solidity ({solidity:.3f} < {SOLIDITY_THRESHOLD}) -> "
            f"traced area doesn't fill its hull, likely handwriting not a shape"
        )
        return None, debug_info

    epsilon = APPROX_POLY_EPSILON_RATIO * perimeter
    approx = cv2.approxPolyDP(hull, epsilon, closed=True)
    approx_points = approx.reshape(-1, 2)
    num_vertices = len(approx_points)
    debug_info["num_vertices"] = num_vertices

    circularity = (4 * math.pi * hull_area) / (perimeter ** 2)
    debug_info["circularity"] = round(circularity, 3)

    # --- Triangle ---
    if num_vertices == 3:
        debug_info["reason"] = "3 vertices -> triangle"
        return ShapeResult("triangle", bounding_box, color), debug_info

    # --- Rectangle / square ---
    if num_vertices == 4:
        rect_ok = _looks_like_rectangle(approx_points)
        debug_info["rect_check_passed"] = rect_ok
        if rect_ok:
            debug_info["reason"] = "4 vertices + ~90deg angles -> rectangle"
            return ShapeResult("rectangle", bounding_box, color), debug_info

    # --- Circle ---
    if circularity > CIRCULARITY_THRESHOLD:
        debug_info["reason"] = f"circularity {circularity:.3f} > {CIRCULARITY_THRESHOLD} -> circle"
        return ShapeResult("circle", bounding_box, color), debug_info

    # Not a recognized shape — Phase 5 will try handwriting correction instead
    debug_info["reason"] = (
        f"no match (verts={num_vertices}, circ={circularity:.3f}, "
        f"rect_ok={debug_info['rect_check_passed']})"
    )
    return None, debug_info


def classify_stroke(stroke_points, color):
    """
    Classify a completed stroke's point list as a recognized shape.
    Thin wrapper around classify_stroke_debug() for callers that don't
    need the diagnostic info.

    Returns:
        ShapeResult if recognized, otherwise None (caller should fall through
        to handwriting correction).
    """
    result, _ = classify_stroke_debug(stroke_points, color)
    return result


def render_clean_shape(surface, shape_result, thickness=4):
    """
    Draw a clean version of the classified shape onto `surface`, sized to
    the original stroke's bounding box (Phase 4, step 22).
    """
    x, y, w, h = shape_result.bounding_box
    color = shape_result.color

    if shape_result.shape_type == "rectangle":
        cv2.rectangle(surface, (x, y), (x + w, y + h), color, thickness)

    elif shape_result.shape_type == "circle":
        center = (x + w // 2, y + h // 2)
        radius = max(w, h) // 2
        cv2.circle(surface, center, radius, color, thickness)

    elif shape_result.shape_type == "triangle":
        # Isosceles triangle inscribed in the bounding box: apex centered on
        # top edge, base spans the bottom edge.
        apex = (x + w // 2, y)
        bottom_left = (x, y + h)
        bottom_right = (x + w, y + h)
        pts = np.array([apex, bottom_left, bottom_right], dtype=np.int32).reshape((-1, 1, 2))
        cv2.polylines(surface, [pts], isClosed=True, color=color, thickness=thickness)