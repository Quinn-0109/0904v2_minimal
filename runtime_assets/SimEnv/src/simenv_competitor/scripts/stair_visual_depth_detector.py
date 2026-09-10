#!/usr/bin/env python3
"""Side-band stair detector using RGB structure + depth geometry + temporal vote.

This node is deliberately decoupled from motion:
  - subscribes RGB/depth/camera_info
  - publishes a debug image and JSON detection
  - optionally saves debug frames
  - never publishes /cmd_vel, /joy, or navigation goals

Detection logic:
  1) RGB: stair-like parallel/repeated edges in right/front image ROI.
  2) Depth: ROI must have enough valid depth and multiple depth layers / strong
     non-planar variation. This rejects flat walls/doors/ground markings.
  3) Temporal: final "stable" detection requires repeated positive detections
     with similar bearing over a sliding window.
"""

import argparse
import json
import math
import os
import time
from collections import deque

import cv2
import numpy as np


def _clip(v, lo, hi):
    return max(lo, min(hi, v))


def _angle_deg(x1, y1, x2, y2):
    return math.degrees(math.atan2(y2 - y1, x2 - x1))


def _median_depth(depth, x, y, radius=5):
    if depth is None:
        return None
    h, w = depth.shape[:2]
    x1 = _clip(int(x) - radius, 0, w - 1)
    x2 = _clip(int(x) + radius + 1, 0, w)
    y1 = _clip(int(y) - radius, 0, h - 1)
    y2 = _clip(int(y) + radius + 1, 0, h)
    patch = depth[y1:y2, x1:x2].astype(np.float32)
    vals = patch[np.isfinite(patch) & (patch > 0.05) & (patch < 20.0)]
    if vals.size < 5:
        return None
    return float(np.median(vals))


def normalize_depth_image(depth_msg_arr):
    if depth_msg_arr is None:
        return None
    depth = depth_msg_arr.astype(np.float32)
    # Common ROS depth encodings: 16UC1 is millimeters, 32FC1 is meters.
    finite = depth[np.isfinite(depth)]
    # Gazebo can publish a transient all-NaN/empty depth frame while rendering
    # catches up. Keep that frame as unavailable depth instead of raising from
    # nanmax and starving the RGB stair-context stream.
    if finite.size == 0:
        return depth
    if np.nanmax(finite) > 50.0:
        depth *= 0.001
    return depth


def rgb_candidates(img_bgr):
    h, w = img_bgr.shape[:2]
    b, g, r = cv2.split(img_bgr)
    # Remove hand-drawn red marks and red debug overlays if screenshots/recorded
    # video are used as inputs.
    red_mask = ((r > 130) & (r > g * 1.55) & (r > b * 1.55)).astype(np.uint8) * 255
    red_mask = cv2.dilate(red_mask, np.ones((9, 9), np.uint8), iterations=2)
    cleaned = img_bgr.copy()
    cleaned[red_mask > 0] = (80, 80, 80)
    gray = cv2.cvtColor(cleaned, cv2.COLOR_BGR2GRAY)

    # The staircase body is usually front/right, but its *first riser* is at
    # the lower-left end of that body in this view.  Starting at 36% cropped
    # out that first riser entirely and forced the selector onto a far-right
    # tread/edge.  Keep a small left margin while retaining the vertical ROI.
    x0 = int(0.05 * w)
    y0 = int(0.08 * h)
    x1 = w
    y1 = int(0.88 * h)
    roi = gray[y0:y1, x0:x1]
    roi_eq = cv2.equalizeHist(roi)
    blur = cv2.GaussianBlur(roi_eq, (5, 5), 0)
    edges = cv2.Canny(blur, 45, 120, apertureSize=3)
    edges[red_mask[y0:y1, x0:x1] > 0] = 0

    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180.0,
        threshold=32,
        minLineLength=max(22, int(0.035 * w)),
        maxLineGap=12,
    )
    out = []
    if lines is not None:
        for l in lines[:, 0, :]:
            lx1, ly1, lx2, ly2 = [int(v) for v in l]
            gx1, gy1 = lx1 + x0, ly1 + y0
            gx2, gy2 = lx2 + x0, ly2 + y0
            length = math.hypot(gx2 - gx1, gy2 - gy1)
            if length < 20:
                continue
            ang = _angle_deg(gx1, gy1, gx2, gy2)
            absang = abs(ang)
            tread_like = absang <= 30 or abs(absang - 180) <= 30
            side_like = 24 <= absang <= 66
            if not (tread_like or side_like):
                continue
            if min(gy1, gy2) < int(0.08 * h) or max(gy1, gy2) > int(0.88 * h):
                continue
            # Reject long low perspective/floor edges.
            if length > 0.45 * w and min(gy1, gy2) > int(0.55 * h):
                continue
            out.append({
                "p1": [gx1, gy1],
                "p2": [gx2, gy2],
                "mid": [(gx1 + gx2) * 0.5, (gy1 + gy2) * 0.5],
                "angle_deg": ang,
                "length": length,
                "kind": "tread" if tread_like else "side",
            })
    return out, [x0, y0, x1 - x0, y1 - y0], edges


def group_bbox(lines, img_shape):
    h, w = img_shape[:2]
    if not lines:
        return None
    pts = []
    for c in lines:
        pts.append(c["p1"])
        pts.append(c["p2"])
    pts = np.array(pts, dtype=np.float32)
    xs, ys = pts[:, 0], pts[:, 1]
    bx1 = int(np.percentile(xs, 5)) - 10
    bx2 = int(np.percentile(xs, 95)) + 10
    by1 = int(np.percentile(ys, 5)) - 10
    by2 = int(np.percentile(ys, 95)) + 10
    bx1 = _clip(bx1, 0, w - 1)
    bx2 = _clip(bx2, 0, w - 1)
    by1 = _clip(by1, 0, h - 1)
    by2 = _clip(by2, 0, h - 1)
    if bx2 <= bx1 + 20 or by2 <= by1 + 20:
        return None
    return [bx1, by1, bx2, by2]


def depth_metrics(depth, bbox, first_step_pixel=None):
    if depth is None or bbox is None:
        return {"available": False, "reason": "no_depth"}
    h, w = depth.shape[:2]
    bx1, by1, bx2, by2 = bbox
    bx1, bx2 = _clip(bx1, 0, w - 1), _clip(bx2, 0, w - 1)
    by1, by2 = _clip(by1, 0, h - 1), _clip(by2, 0, h - 1)
    roi = depth[by1:by2, bx1:bx2].astype(np.float32)
    vals = roi[np.isfinite(roi) & (roi > 0.10) & (roi < 15.0)]
    area = max(1, roi.size)
    if vals.size < 80:
        return {
            "available": True,
            "valid_ratio": float(vals.size / area),
            "reason": "too_few_valid_depth",
        }
    p10, p25, p50, p75, p90 = [float(np.percentile(vals, p)) for p in (10, 25, 50, 75, 90)]
    iqr = p75 - p25
    spread = p90 - p10
    # Layer count by depth histogram. Stairs/side structure is non-planar; a
    # flat wall usually collapses to one/two dominant bins.
    bins = np.arange(max(0.05, p10), p90 + 0.12, 0.12)
    hist, _ = np.histogram(vals, bins=bins) if len(bins) >= 2 else (np.array([]), [])
    strong_bins = int(np.sum(hist > max(10, vals.size * 0.035))) if hist.size else 0

    lower = depth[int(by1 + 0.55 * (by2 - by1)):by2, bx1:bx2].astype(np.float32)
    upper = depth[by1:int(by1 + 0.45 * (by2 - by1)), bx1:bx2].astype(np.float32)
    lv = lower[np.isfinite(lower) & (lower > 0.10) & (lower < 15.0)]
    uv = upper[np.isfinite(upper) & (upper > 0.10) & (upper < 15.0)]
    vertical_delta = None
    if lv.size > 20 and uv.size > 20:
        vertical_delta = float(np.median(uv) - np.median(lv))

    first_depth = None
    if first_step_pixel is not None:
        first_depth = _median_depth(depth, first_step_pixel[0], first_step_pixel[1], radius=7)

    # Do not demand huge spread; simulated Realsense on flat-colored stairs can
    # be smooth. But require more than a perfectly flat wall.
    ok = (
        vals.size / area > 0.08
        and (spread > 0.18 or abs(vertical_delta or 0.0) > 0.12 or strong_bins >= 3)
    )
    return {
        "available": True,
        "ok": bool(ok),
        "valid_ratio": float(vals.size / area),
        "median": p50,
        "spread_p90_p10": spread,
        "iqr": iqr,
        "strong_depth_bins": strong_bins,
        "vertical_depth_delta": vertical_delta,
        "first_step_depth": first_depth,
        "reason": "ok" if ok else "depth_not_stair_like",
    }


def _tread_riser_contrast(gray, line):
    """Return the intensity step across a finite horizontal tread edge.

    A real visible stair tread in this simulator has a coherent riser band on
    one side of the edge.  Floor seams can also be horizontal, so this value is
    used only together with repeated, monotonic tread-chain support.
    """
    h, w = gray.shape[:2]
    x1, y1 = line["p1"]
    x2, y2 = line["p2"]
    xa = max(0, min(x1, x2) + 3)
    xb = min(w, max(x1, x2) - 2)
    y = int(round(line["mid"][1]))
    if xb <= xa + 4 or y < 8 or y + 8 >= h:
        return None
    above = float(np.mean(gray[y - 7:y - 2, xa:xb]))
    below = float(np.mean(gray[y + 2:y + 7, xa:xb]))
    return below - above


def choose_first_step(lines, bbox, img_shape, depth=None, img_bgr=None):
    h, w = img_shape[:2]
    if bbox is None:
        return None, None
    bx1, by1, bx2, by2 = bbox
    bh = max(1, by2 - by1)
    # First try to associate a candidate with a *repeated stair chain*.
    # Merely choosing the lowest/nearest horizontal line is unsafe: floor seams
    # in front of the stairs satisfy exactly those two properties.
    gray = (cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
            if img_bgr is not None else None)
    chain_treads = []
    if gray is not None:
        for c in lines:
            # The first few riser/tread edges can be oblique (up to about
            # 30 deg) when the staircase is only partly visible at image
            # right.  The repeated centre progression below, rather than a
            # near-horizontal assumption, is the semantic stair constraint.
            if c["kind"] != "tread" or abs(c["angle_deg"]) > 32.0:
                continue
            mx, my = c["mid"]
            # Do not inherit group_bbox's right edge here.  In a partial view
            # that box can be dominated by floor seams and clip off the real
            # staircase.  Keep candidates inside the camera ROI instead.
            # This detector is enabled only after the verified right-turn
            # observation action.  In that camera pose the first stair is in
            # the centre/right middle band.  Perspective floor seams occupy
            # the lower band; excluding it prevents them from forming a fake
            # converging "chain".
            if not (0.50 * w <= mx <= 0.99 * w and
                    0.18 * h <= my <= 0.70 * h):
                continue
            if not (22.0 <= c["length"] <= 0.24 * w):
                continue
            contrast = _tread_riser_contrast(gray, c)
            if contrast is None or contrast < 8.0:
                continue
            chain_treads.append((c, contrast))

    chain_anchors = []
    for c, contrast in chain_treads:
        cx, cy = c["mid"]
        support = []
        for q, q_contrast in chain_treads:
            qx, qy = q["mid"]
            dx = qx - cx
            rise = cy - qy
            # In the verified post-turn view the staircase rises toward image
            # right.  Limit each association to neighbouring visible steps;
            # this prevents a remote floor seam from jumping directly into the
            # upper stair stack.
            if 15.0 <= dx <= 175.0 and 8.0 <= rise <= 85.0:
                support.append((q, q_contrast))
        if len(support) < 2:
            continue
        y_span = cy - min(q[0]["mid"][1] for q in support)
        x_span = max(q[0]["mid"][0] for q in support) - cx
        if y_span < 25.0 or x_span < 45.0:
            continue
        # Prefer the lowest member of a well-supported chain.  Contrast and
        # support break ties, while no global "left-most line" preference is
        # used anymore.
        score = (4.0 * len(support) + 0.035 * y_span +
                 0.012 * contrast + 1.2 * cy / h)
        chain_anchors.append((score, c))

    if chain_anchors:
        chain_anchors.sort(reverse=True, key=lambda x: x[0])
        c = chain_anchors[0][1]
        px = int(round(c["mid"][0]))
        py = int(round(c["mid"][1]))
        return [px, py], [c["p1"], c["p2"]]

    # No compatibility fallback is allowed here.  A single low horizontal
    # segment is indistinguishable from a floor/tile seam in this scene.  It is
    # safer to wait for the next camera frame than to publish a false stair
    # target to the motion sideband.
    return None, None


def choose_wall_step_midpoint(img_bgr, first_step_pixel, first_step_line, bbox,
                              stair_lines=None):
    """Locate free-floor centre between the left wall foot and first step.

    The navigation target is not the stair centre.  In the intended view the
    first riser is on the right and the wall/floor junction is on the left.
    Detect that high-contrast junction independently, evaluate it at a floor
    row just in front of the riser, then take the image-space midpoint of the
    two physical boundaries.  No layout or Gazebo truth is used.
    """
    if bbox is None:
        return None
    h, w = img_bgr.shape[:2]
    # Evaluate both free-space boundaries on one ground row.  This avoids
    # pairing a near floor seam with a tread at a different perspective depth.
    target_y = int(0.64 * h)

    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 35, 105, apertureSize=3)
    segments = cv2.HoughLinesP(
        edges, rho=1, theta=np.pi / 180.0, threshold=30,
        minLineLength=max(35, int(0.10 * w)), maxLineGap=20)
    left_candidates = []
    right_candidates = []
    structural_left_candidates = []
    if segments is not None:
        for raw in segments[:, 0, :]:
            x1, y1, x2, y2 = [float(v) for v in raw]
            if x2 < x1:
                x1, y1, x2, y2 = x2, y2, x1, y1
            dx = x2 - x1
            dy = y2 - y1
            length = math.hypot(dx, dy)
            if dx < 18.0 or length < 0.10 * w:
                continue
            slope = dy / dx
            # Horizontal Hough segments are floor/tread seams, not either
            # side boundary.  Reject them before any x-at-y calculation to
            # avoid division by zero and meaningless far extrapolation.
            if abs(slope) < 1e-6:
                continue
            if max(y1, y2) < 0.34 * h:
                continue
            line_y_min = min(y1, y2)
            line_y_max = max(y1, y2)
            # Candidate classification may inspect the preferred floor row,
            # but it must stay on the finite detected segment.  The final pair
            # is recomputed below on one shared row inside both segments.
            eval_y = _clip(target_y, line_y_min, line_y_max)
            x_at = x1 + (eval_y - y1) / slope
            mx = int(_clip(0.5 * (x1 + x2), 2, w - 3))
            my = int(_clip(0.5 * (y1 + y2), 10, h - 11))
            above = float(np.median(gray[max(0, my-12):max(1, my-4),
                                         max(0, mx-4):min(w, mx+5)]))
            below = float(np.median(gray[min(h-1, my+4):min(h, my+13),
                                         max(0, mx-4):min(w, mx+5)]))
            contrast = below - above
            line = [int(x1), int(y1), int(x2), int(y2)]
            if (-1.8 <= slope <= -0.10 and 0.02 * w <= x_at <= 0.52 * w
                    and contrast >= 5.0):
                score = (3.0 * contrast + 0.35 * length
                         - 0.12 * abs(target_y-my) + 0.08 * x_at)
                left_candidates.append((score, line, contrast, slope,
                                        line_y_min, line_y_max))
            # The wall-side boundary of the staircase recedes down/right.  It
            # is normally a long side edge, not one of the many horizontal
            # floor/tread seams that confused the previous selector.
            if (0.10 <= slope <= 1.8 and 0.52 * w <= x_at <= 0.98 * w
                    and length >= 0.13 * w):
                score = (0.55 * length - 0.10 * abs(target_y-my)
                         - 0.04 * x_at)
                right_candidates.append((score, line, contrast, slope,
                                         line_y_min, line_y_max))

    # The stair detector already supplies finite, structurally associated side
    # edges.  Reuse a negative-slope side edge as an inner wall-foot candidate
    # even when its grey-on-grey contrast is weaker than the outer wall.  It
    # still passes the same shared-row, gap, and midpoint checks below.
    for candidate in stair_lines or []:
        if candidate.get("kind") != "side":
            continue
        p1 = candidate.get("p1")
        p2 = candidate.get("p2")
        if not (isinstance(p1, list) and isinstance(p2, list) and
                len(p1) == 2 and len(p2) == 2):
            continue
        x1, y1 = [float(v) for v in p1]
        x2, y2 = [float(v) for v in p2]
        if x2 < x1:
            x1, y1, x2, y2 = x2, y2, x1, y1
        dx = x2 - x1
        dy = y2 - y1
        length = math.hypot(dx, dy)
        if dx < 18.0 or length < 0.08 * w:
            continue
        slope = dy / dx
        if not (-1.8 <= slope <= -0.10):
            continue
        line_y_min = min(y1, y2)
        line_y_max = max(y1, y2)
        eval_y = _clip(target_y, line_y_min, line_y_max)
        x_at = x1 + (eval_y - y1) / slope
        if not (0.18 * w <= x_at < first_step_pixel[0]):
            continue
        line = [int(x1), int(y1), int(x2), int(y2)]
        score = 0.35 * length - 0.12 * abs(target_y - 0.5 * (y1 + y2))
        structural_left_candidates.append((score, line, 0.0, slope,
                                           line_y_min, line_y_max))
    # Never fall back to a merely high-contrast outer wall.  If the current
    # frame does not expose a stair-associated inner edge, publish no midpoint
    # and let the temporal detector wait for the next frame.
    if not structural_left_candidates:
        return None
    left_candidates = structural_left_candidates
    if not left_candidates or not right_candidates:
        return None
    pairs = []
    for left in left_candidates:
        for right in right_candidates:
            left_line = left[1]
            right_line = right[1]
            overlap_low = max(left[4], right[4])
            overlap_high = min(left[5], right[5])
            overlap = overlap_high - overlap_low
            # Both physical boundary points must be interpolated at the same
            # image row and that row must lie inside both finite Hough
            # segments.  With no meaningful overlap there is no supported
            # wall--stair corridor cross-section, so reject the pair.
            if overlap < 12.0:
                continue
            margin = min(8.0, 0.20 * overlap)
            pair_y = _clip(target_y, overlap_low + margin,
                           overlap_high - margin)
            lx1, ly1, lx2, ly2 = [float(v) for v in left_line]
            rx1, ry1, rx2, ry2 = [float(v) for v in right_line]
            wall_x = lx1 + (pair_y - ly1) / left[3]
            step_x = rx1 + (pair_y - ry1) / right[3]
            gap = step_x - wall_x
            if not (0.18 * w <= gap <= 0.70 * w):
                continue
            midpoint_x = 0.5 * (wall_x + step_x)
            if not (0.25 * w <= midpoint_x <= 0.75 * w):
                continue
            # Prefer the nearest enclosing boundaries and a navigable central
            # gap over unrelated outer architecture edges.
            gap_preference = -abs(gap - 0.34 * w)
            pairs.append((left[0] + right[0] + 0.35 * gap_preference,
                          left, right, midpoint_x, pair_y, wall_x, step_x,
                          overlap))
    if not pairs:
        return None
    # The entrance wall is the nearest valid left boundary of the stair-side
    # aisle.  A more distant outer wall can have stronger contrast and used to
    # win the appearance score even though it encloses a wider, wrong gap.
    # Select the narrowest geometrically valid pair first; use the established
    # appearance score only as a tie-breaker.
    pairs.sort(key=lambda item: (item[6] - item[5], -item[0]))
    (_pair_score, left, right, _pair_midpoint_x, pair_y,
     _pair_wall_x, _pair_step_x, pair_overlap) = pairs[0]
    _left_score, wall_line, contrast, _left_slope, _ly0, _ly1 = left
    _right_score, step_line, _step_contrast, _right_slope, _ry0, _ry1 = right

    # Pairing above deliberately uses a shared in-segment row: it proves that
    # the two finite Hough segments really enclose one plausible passage.  Once
    # the pair is accepted, however, selecting points on that near row leaves
    # the approach goal too far outside the stair bay.  Use the visually far
    # endpoint (smaller image y) of each *finite* segment instead.  This moves
    # the target deeper without extrapolating either detected line.
    wall_p1 = (float(wall_line[0]), float(wall_line[1]))
    wall_p2 = (float(wall_line[2]), float(wall_line[3]))
    step_p1 = (float(step_line[0]), float(step_line[1]))
    step_p2 = (float(step_line[2]), float(step_line[3]))
    wall_far = min((wall_p1, wall_p2), key=lambda p: (p[1], -p[0]))
    step_far = min((step_p1, step_p2), key=lambda p: (p[1], p[0]))
    midpoint_x = 0.5 * (wall_far[0] + step_far[0])
    midpoint_y = 0.5 * (wall_far[1] + step_far[1])
    return {
        "wall_foot_pixel": [int(round(wall_far[0])), int(round(wall_far[1]))],
        "wall_foot_line": wall_line,
        "first_step_wall_edge_pixel": [int(round(step_far[0])), int(round(step_far[1]))],
        "first_step_wall_edge_line": step_line,
        "aisle_midpoint_pixel": [int(round(midpoint_x)), int(round(midpoint_y))],
        "gap_width_px": float(step_far[0] - wall_far[0]),
        "endpoint_distance_px": float(math.hypot(
            step_far[0] - wall_far[0], step_far[1] - wall_far[1])),
        "point_policy": "finite_segment_far_endpoints",
        "shared_segment_row": float(pair_y),
        "shared_segment_overlap_px": float(pair_overlap),
        "wall_floor_contrast": float(contrast),
    }


def stair_structure_metrics(lines, bbox, img_shape):
    """Return stricter structural checks for the visible stair block.

    False positives in this scene are often large wall/floor rectangles with
    many horizontal edges but few repeated diagonal riser/side edges.  A real
    stair view should include a cluster of short/medium diagonal edges in the
    upper or side part, or multiple tread lines distributed vertically.
    """
    h, w = img_shape[:2]
    if bbox is None:
        return {"ok": False, "reason": "no_bbox"}
    bx1, by1, bx2, by2 = bbox
    bw = max(1, bx2 - bx1)
    bh = max(1, by2 - by1)
    in_box = []
    for c in lines:
        mx, my = c["mid"]
        if bx1 <= mx <= bx2 and by1 <= my <= by2:
            in_box.append(c)
    treads = [c for c in in_box if c["kind"] == "tread"]
    sides = [c for c in in_box if c["kind"] == "side" and c["length"] < 0.40 * w]
    tread_y_bins = set()
    for c in treads:
        rel_y = (c["mid"][1] - by1) / bh
        if 0.05 <= rel_y <= 0.92:
            tread_y_bins.add(int(rel_y * 8))
    side_y_bins = set()
    side_x_bins = set()
    for c in sides:
        rel_y = (c["mid"][1] - by1) / bh
        rel_x = (c["mid"][0] - bx1) / bw
        if 0.02 <= rel_y <= 0.85:
            side_y_bins.add(int(rel_y * 8))
            side_x_bins.add(int(rel_x * 8))
    # At least one of:
    # - enough diagonal stair-riser/side edges spread in the structure
    # - many horizontal treads across multiple vertical bands plus some diagonal support
    diagonal_ok = len(sides) >= 3 and len(side_y_bins) >= 2
    distributed_treads_ok = len(treads) >= 6 and len(tread_y_bins) >= 3 and len(sides) >= 1
    # Once the robot faces the staircase, the visible geometry is often a
    # near-frontal stack of repeated horizontal risers.  In that valid view
    # there may be no diagonal side edge at all.  Require a stronger repeated
    # tread pattern (rather than simply dropping the side-edge requirement) so
    # ordinary wall/floor seams remain rejected.
    frontal_treads_ok = len(treads) >= 10 and len(tread_y_bins) >= 5
    return {
        "ok": bool(diagonal_ok or distributed_treads_ok or frontal_treads_ok),
        "reason": "ok" if (diagonal_ok or distributed_treads_ok or frontal_treads_ok) else "missing_stair_diagonal_or_vertical_repetition",
        "in_box_count": len(in_box),
        "tread_count_box": len(treads),
        "side_count_box": len(sides),
        "tread_y_bins": sorted(tread_y_bins),
        "side_y_bins": sorted(side_y_bins),
        "side_x_bins": sorted(side_x_bins),
        "frontal_treads_ok": frontal_treads_ok,
    }


def detect_side_stair_candidate(img_bgr, lines, depth, camera_info):
    """Detect the upper-right side profile of a staircase while in corridor.

    This is deliberately separate from the frontal first-riser detector.  A
    valid side profile must contain both repeated near-horizontal tread edges
    and repeated diagonal riser/underside edges in one compact upper-right
    cluster.  The lowest supported tread supplies a short-range RGB-D
    landmark; it is never used directly as a stair-entry navigation target.
    """
    h, w = img_bgr.shape[:2]
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    side_roi = (int(.58*w), int(.06*h), int(.99*w), int(.50*h))
    x0, y0, x1, y1 = side_roi
    treads = []
    sides = []
    for c in lines:
        mx, my = c["mid"]
        if not (x0 <= mx <= x1 and y0 <= my <= y1):
            continue
        if not (20.0 <= c["length"] <= .24*w):
            continue
        if c["kind"] == "tread" and abs(c["angle_deg"]) <= 25.0:
            contrast = _tread_riser_contrast(gray, c)
            # Hough endpoint order is arbitrary in the side view, so the
            # bright-to-dark polarity may be reversed.  Repetition plus the
            # diagonal cluster supplies semantics; use contrast magnitude.
            if contrast is not None and abs(contrast) >= 7.0:
                treads.append(c)
        elif c["kind"] == "side" and 35.0 <= abs(c["angle_deg"]) <= 66.0:
            sides.append(c)

    result = {
        "detected_raw": False,
        # RGB-only semantic context.  The mission may use this to establish
        # that the planar range target ahead belongs to the stair area, but it
        # must never use this flag as a metric stair-entry goal.
        "structure_ok": False,
        "reason": "insufficient_side_profile",
        "roi": list(side_roi),
        "tread_count": len(treads),
        "side_count": len(sides),
        "anchor_pixel": None,
        "anchor_depth_m": None,
        "camera_point": None,
    }
    if len(treads) < 2 or len(sides) < 2:
        return result

    # Reject a single wall/floor corner: a stair profile occupies several
    # rows and has diagonal support spatially close to the horizontal chain.
    tread_ys = [c["mid"][1] for c in treads]
    all_xs = [c["mid"][0] for c in treads+sides]
    all_ys = [c["mid"][1] for c in treads+sides]
    if max(tread_ys)-min(tread_ys) < 22.0 or max(all_ys)-min(all_ys) < 35.0:
        result["reason"] = "side_profile_not_repeated"
        return result
    if max(all_xs)-min(all_xs) > .38*w:
        result["reason"] = "side_profile_not_compact"
        return result

    result["structure_ok"] = True

    # The lowest tread in this upper image band is the closest visible member
    # of the side profile.  Use its left endpoint (the lower-flight side) and
    # sample depth around both endpoints and the midpoint to survive edge
    # pixels/no-return values.
    anchor_line = max(treads, key=lambda c: c["mid"][1])
    p1, p2 = anchor_line["p1"], anchor_line["p2"]
    left = p1 if p1[0] <= p2[0] else p2
    sample_pixels = [left, anchor_line["mid"], p1, p2]
    depths = []
    if depth is not None:
        for px, py in sample_pixels:
            value = _median_depth(depth, int(round(px)), int(round(py)), radius=7)
            if value is not None and .6 <= value <= 8.0:
                depths.append(float(value))
    ax, ay = int(round(left[0])), int(round(left[1]))
    # Publish the structural anchor even without depth so recorded RGB frames
    # can audit the decision. It is not navigation-valid until RGB-D supplies
    # a metric camera point, so ``detected_raw`` remains false in that case.
    result.update({
        "anchor_pixel": [ax, ay],
        "anchor_line": [list(p1), list(p2)],
        "vertical_span_px": float(max(all_ys)-min(all_ys)),
        "horizontal_span_px": float(max(all_xs)-min(all_xs)),
    })
    if not depths:
        result["reason"] = "side_profile_no_depth"
        return result
    anchor_depth = float(np.median(depths))
    camera_point = None
    if camera_info is not None:
        fx, fy = camera_info.get("fx"), camera_info.get("fy")
        cx, cy = camera_info.get("cx"), camera_info.get("cy")
        if fx and fy:
            camera_point = [
                float((ax-cx)*anchor_depth/fx),
                float((ay-cy)*anchor_depth/fy),
                anchor_depth,
            ]
    result.update({
        "detected_raw": camera_point is not None,
        "reason": "ok" if camera_point is not None else "no_camera_model",
        "anchor_depth_m": anchor_depth,
        "camera_point": camera_point,
        "confidence": float(min(1.0, .18*len(treads)+.16*len(sides))),
    })
    return result


def detect_frame(img_bgr, depth=None, camera_info=None, history=None):
    h, w = img_bgr.shape[:2]
    lines, roi, _edges = rgb_candidates(img_bgr)
    side_stair = detect_side_stair_candidate(
        img_bgr, lines, depth, camera_info)
    bbox = group_bbox(lines, img_bgr.shape)
    if bbox is None:
        det = {
            "detected_raw": False,
            "detected_stable": False,
            "reason": "no_rgb_bbox",
            "roi": roi,
            "line_count": len(lines),
            "side_stair_candidate": side_stair,
            "lines": lines[:80],
        }
        return det
    first_pixel, first_line = choose_first_step(
        lines, bbox, img_bgr.shape, depth=depth, img_bgr=img_bgr)
    if first_pixel is None:
        if history is not None:
            history.append({
                "ok": False,
                "bearing": 0.0,
                "t": time.time(),
                "confidence": 0.0,
            })
        sm = stair_structure_metrics(lines, bbox, img_bgr.shape)
        return {
            "detected_raw": False,
            "detected_stable": False,
            "reason": "no_associated_tread_chain",
            "confidence": 0.0,
            "stable_count": 0,
            "bearing_std": None,
            "roi": roi,
            "bbox": bbox,
            "first_step_pixel": None,
            "first_step_line": None,
            "normalized_bearing": None,
            "line_count": len(lines),
            "tread_count": sum(1 for c in lines if c["kind"] == "tread"),
            "side_count": sum(1 for c in lines if c["kind"] == "side"),
            "depth": {"available": depth is not None,
                      "ok": False,
                      "reason": "no_associated_tread_chain"},
            "structure": sm,
            "entry_target_relative": None,
            "wall_step_midpoint": None,
            "aisle_midpoint_camera_point": None,
            "stair_boundary_camera_point": None,
            "side_stair_candidate": side_stair,
            "lines": lines[:80],
        }
    aisle = choose_wall_step_midpoint(img_bgr, first_pixel, first_line, bbox,
                                      stair_lines=lines)
    dm = depth_metrics(depth, bbox, first_pixel)
    sm = stair_structure_metrics(lines, bbox, img_bgr.shape)

    tread_count = sum(1 for c in lines if c["kind"] == "tread")
    side_count = sum(1 for c in lines if c["kind"] == "side")
    bx1, by1, bx2, by2 = bbox
    area_ratio = ((bx2 - bx1) * (by2 - by1)) / float(w * h)
    bearing = (first_pixel[0] - 0.5 * w) / (0.5 * w)
    # Penalize huge all-image boxes and boxes stuck at very bottom.
    rgb_score = min(1.0, 0.11 * min(tread_count, 6) + 0.08 * min(side_count, 4) + 0.55 * min(area_ratio / 0.22, 1.0))
    # A genuine first riser can be at the left edge of the expanded stair ROI.
    # Keep only a narrow image-border rejection; the structural/depth checks
    # remain the false-positive guard.
    point_not_on_edge = first_pixel[0] < int(0.95 * w) and first_pixel[0] > int(0.08 * w)
    shape_ok = (
        len(lines) >= 5
        and tread_count >= 3
        and 0.025 <= area_ratio <= 0.55
        and by2 < int(0.92 * h)
        and point_not_on_edge
        and sm.get("ok", False)
    )
    # ROI-wide depth on Gazebo's flat grey stair can look deceptively planar.
    # If the RGB stair structure is strong and the selected first-step point has
    # a plausible depth, allow a candidate even when the whole ROI histogram is
    # conservative.  This still requires the structural stair check above.
    first_depth_for_gate = dm.get("first_step_depth")
    point_depth_ok = first_depth_for_gate is not None and 0.6 <= first_depth_for_gate <= 6.0
    depth_ok = (not dm.get("available")) or dm.get("ok", False) or (sm.get("ok", False) and point_depth_ok)
    raw_ok = bool(shape_ok and rgb_score >= 0.42 and depth_ok)
    # A historical stable latch is useful for diagnostics, but a failed
    # current frame must not expose a fresh navigation midpoint.  Keep the
    # geometric proposal private unless this frame itself passes all gates.
    usable_aisle = aisle if raw_ok else None

    stable = False
    stable_count = 0
    bearing_std = None
    if history is not None:
        history.append({
            "ok": raw_ok,
            "bearing": bearing,
            "t": time.time(),
            "confidence": rgb_score,
        })
        recent = [x for x in history if x["ok"]]
        stable_count = len(recent)
        if len(recent) >= 2:
            bs = np.array([x["bearing"] for x in recent], dtype=np.float32)
            bearing_std = float(np.std(bs))
        stable = stable_count >= max(3, int(0.55 * len(history))) and (bearing_std is None or bearing_std < 0.42)

    first_depth = dm.get("first_step_depth")
    camera_point = None
    entry_pixel = None
    entry_depth = None
    entry_camera_point = None
    entry_target_relative = None
    entry_anchor_pixel = None
    aisle_midpoint_camera_point = None
    stair_boundary_camera_point = None
    if first_depth is not None and camera_info is not None:
        fx = camera_info.get("fx")
        fy = camera_info.get("fy")
        cx = camera_info.get("cx")
        cy = camera_info.get("cy")
        if fx and fy:
            z = first_depth
            x = (first_pixel[0] - cx) * z / fx
            y = (first_pixel[1] - cy) * z / fy
            camera_point = [float(x), float(y), float(z)]

            # Detection and navigation must be separated:
            #   - detected stair lines/bbox tell us where the stair is;
            #   - the navigation target should be the left wall-foot / floor
            #     patch in front of the first step, outside the stair body.
            #
            # The approach floor is directly below the *detected first riser*,
            # not an arbitrary corner of the complete stair bounding box.  A
            # bbox corner includes the long upper flight and can point through
            # the nearby wall.  Keep the first-step x coordinate and sample a
            # lower image row on the visible floor before that riser.
            if bbox is not None:
                bx1, by1, bx2, by2 = bbox
                bw = max(1, bx2 - bx1)
                bh = max(1, by2 - by1)
                entry_anchor_pixel = [
                    int(_clip(first_pixel[0], 0, w - 1)),
                    int(_clip(first_pixel[1] + 0.16 * bh, 0, h - 1)),
                ]
                entry_pixel = [
                    int(_clip(first_pixel[0], 0, w - 1)),
                    int(_clip(first_pixel[1] + 0.30 * bh, 0, h - 1)),
                ]
            else:
                entry_anchor_pixel = list(first_pixel)
                entry_pixel = [
                    int(_clip(first_pixel[0] - 0.20 * w, 0, w - 1)),
                    int(_clip(first_pixel[1] + 0.08 * h, 0, h - 1)),
                ]

            entry_depth = _median_depth(depth, entry_pixel[0], entry_pixel[1], radius=11)
            anchor_depth = _median_depth(depth, entry_anchor_pixel[0], entry_anchor_pixel[1], radius=9)
            target_depth = entry_depth if entry_depth is not None else anchor_depth
            if target_depth is None:
                target_depth = first_depth

            tx = (entry_pixel[0] - cx) * target_depth / fx
            # Use the actual left-wall-foot floor pixel as target.  Clamp only
            # to keep it reachable/safe; do not bias it back to the right.
            forward = _clip(target_depth, 0.55, 1.15)
            right = _clip(tx, -0.65, 0.05)
            bearing = math.atan2(right, forward)
            entry_target_relative = {
                "frame": "camera_forward_right",
                "forward_m": float(forward),
                "right_m": float(right),
                "range_m": float(math.hypot(forward, right)),
                "bearing_rad": float(bearing),
                "approach_offset_m": 0.0,
                "lateral_center_bias_m": 0.0,
                "anchor_pixel": entry_anchor_pixel,
                "anchor_depth_m": float(anchor_depth) if anchor_depth is not None else None,
                "target_pixel": entry_pixel,
                "target_depth_m": float(target_depth),
                "target_policy": "floor_pixel_directly_below_detected_first_riser",
            }
            if entry_depth is not None:
                ex = (entry_pixel[0] - cx) * entry_depth / fx
                ey = (entry_pixel[1] - cy) * entry_depth / fy
                entry_camera_point = [float(ex), float(ey), float(entry_depth)]

            # Navigation target requested by the mission: the free-space
            # midpoint between the left wall foot and the wall-side edge of
            # the first stair, held at a standoff in front of the riser.
            if usable_aisle is not None:
                mx, my = usable_aisle["aisle_midpoint_pixel"]
                sx, sy = usable_aisle["first_step_wall_edge_pixel"]
                stair_boundary_depth = _median_depth(depth, sx, sy, radius=8)
                midpoint_depth = _median_depth(depth, mx, my, radius=10)
                reference_depth = (stair_boundary_depth
                                   if stair_boundary_depth is not None
                                   else first_depth)
                # Keep the strict stair-chain and finite-segment checks above,
                # but do not drive all the way to a distant boundary sample.
                # Keep the verified 0.35 m stand-off before the first step, but
                # allow up to 4.00 m so a valid detected entrance is not clipped.
                midpoint_forward = _clip(reference_depth - 0.35, 0.25, 4.00)
                midpoint_right = (mx - cx) * midpoint_forward / fx
                midpoint_down = (my - cy) * midpoint_forward / fy
                aisle_midpoint_camera_point = [
                    float(midpoint_right), float(midpoint_down),
                    float(midpoint_forward)]
                boundary_right = (sx - cx) * reference_depth / fx
                boundary_down = (sy - cy) * reference_depth / fy
                stair_boundary_camera_point = [
                    float(boundary_right), float(boundary_down),
                    float(reference_depth)]
                usable_aisle["midpoint_depth_m"] = (float(midpoint_depth)
                                                      if midpoint_depth is not None
                                                      else None)
                usable_aisle["stair_boundary_depth_m"] = float(reference_depth)

    return {
        "detected_raw": raw_ok,
        "detected_stable": stable,
        "reason": "ok" if raw_ok else ("depth_reject" if not depth_ok else "rgb_shape_reject"),
        "confidence": rgb_score,
        "stable_count": stable_count,
        "bearing_std": bearing_std,
        "roi": roi,
        "bbox": bbox,
        "first_step_pixel": first_pixel,
        "first_step_line": first_line,
        "normalized_bearing": bearing,
        "line_count": len(lines),
        "tread_count": tread_count,
        "side_count": side_count,
        "area_ratio": area_ratio,
        "depth": dm,
        "structure": sm,
        "first_step_camera_point": camera_point,
        "entry_pixel": entry_pixel,
        "entry_anchor_pixel": entry_anchor_pixel,
        "entry_depth": entry_depth,
        "entry_camera_point": entry_camera_point,
        "entry_target_relative": entry_target_relative,
        "wall_step_midpoint": usable_aisle,
        "aisle_midpoint_camera_point": aisle_midpoint_camera_point,
        "stair_boundary_camera_point": stair_boundary_camera_point,
        "camera_info_source": camera_info.get("source") if camera_info else None,
        "side_stair_candidate": side_stair,
        "lines": lines[:80],
    }


def draw_debug(img_bgr, det):
    out = img_bgr.copy()
    h, w = out.shape[:2]
    x, y, rw, rh = det.get("roi", [0, 0, w, h])
    cv2.rectangle(out, (x, y), (x + rw, y + rh), (80, 80, 80), 1)
    for c in det.get("lines", []):
        color = (0, 220, 255) if c.get("kind") == "tread" else (255, 160, 0)
        cv2.line(out, tuple(c["p1"]), tuple(c["p2"]), color, 2)
    if "bbox" in det:
        color = (0, 255, 0) if det.get("detected_raw") else (0, 180, 255)
        cv2.rectangle(out, tuple(det["bbox"][:2]), tuple(det["bbox"][2:]), color, 3)
    if det.get("first_step_line"):
        p1, p2 = det["first_step_line"]
        cv2.line(out, tuple(p1), tuple(p2), (255, 0, 255), 4)
    if det.get("first_step_pixel"):
        px, py = det["first_step_pixel"]
        cv2.circle(out, (px, py), 10, (0, 0, 255), -1)
        cv2.circle(out, (px, py), 15, (255, 255, 255), 2)
        cv2.arrowedLine(out, (w // 2, h - 25), (px, py), (0, 0, 255), 3, tipLength=0.08)
    if det.get("entry_pixel"):
        ex, ey = det["entry_pixel"]
        cv2.circle(out, (ex, ey), 10, (0, 255, 0), -1)
        cv2.circle(out, (ex, ey), 15, (255, 255, 255), 2)
        cv2.putText(out, "entry", (ex + 12, ey - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2, cv2.LINE_AA)
    midpoint = det.get("wall_step_midpoint")
    if midpoint:
        wx, wy = midpoint["wall_foot_pixel"]
        sx, sy = midpoint["first_step_wall_edge_pixel"]
        mx, my = midpoint["aisle_midpoint_pixel"]
        if midpoint.get("wall_foot_line"):
            x1, y1, x2, y2 = midpoint["wall_foot_line"]
            cv2.line(out, (x1, y1), (x2, y2), (255, 0, 0), 3)
        if midpoint.get("first_step_wall_edge_line"):
            x1, y1, x2, y2 = midpoint["first_step_wall_edge_line"]
            cv2.line(out, (x1, y1), (x2, y2), (255, 0, 255), 3)
        cv2.line(out, (wx, wy), (sx, sy), (255, 255, 0), 2)
        cv2.circle(out, (wx, wy), 8, (255, 0, 0), -1)
        cv2.circle(out, (sx, sy), 8, (0, 0, 255), -1)
        cv2.circle(out, (mx, my), 12, (0, 255, 0), -1)
        cv2.putText(out, "wall-step midpoint", (mx + 12, my - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.50, (0, 255, 0), 2,
                    cv2.LINE_AA)
    label = "raw={} stable={} conf={:.2f} lines={} depth={}".format(
        det.get("detected_raw"),
        det.get("detected_stable"),
        float(det.get("confidence", 0.0)),
        det.get("line_count", 0),
        det.get("depth", {}).get("reason", "n/a"),
    )
    cv2.putText(out, label, (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.68, (0, 0, 255), 2, cv2.LINE_AA)
    if det.get("first_step_camera_point"):
        cv2.putText(out, "cam_xyz={}".format([round(v, 2) for v in det["first_step_camera_point"]]), (12, 58),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.58, (0, 0, 255), 2, cv2.LINE_AA)
    if det.get("entry_target_relative"):
        rel = det["entry_target_relative"]
        cv2.putText(out, "entry f={:.2f} r={:.2f} brg={:.2f}".format(
            rel["forward_m"], rel["right_m"], rel["bearing_rad"]), (12, 84),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.58, (0, 255, 0), 2, cv2.LINE_AA)
    side = det.get("side_stair_candidate", {})
    if isinstance(side, dict):
        roi = side.get("roi")
        if isinstance(roi, list) and len(roi) == 4:
            cv2.rectangle(out, (int(roi[0]), int(roi[1])),
                          (int(roi[2]), int(roi[3])), (255, 180, 0), 1)
        anchor = side.get("anchor_pixel")
        if isinstance(anchor, list) and len(anchor) == 2:
            cv2.circle(out, (int(anchor[0]), int(anchor[1])), 7,
                       (255, 255, 0), 2)
        depth_text = ("n/a" if side.get("anchor_depth_m") is None else
                      "{:.2f}".format(float(side["anchor_depth_m"])))
        cv2.putText(
            out,
            "side={} t={} s={} z={}".format(
                side.get("reason", "n/a"), side.get("tread_count", 0),
                side.get("side_count", 0), depth_text),
            (12, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.50,
            (255, 180, 0), 2, cv2.LINE_AA)
    return out


def offline_main(args):
    img = cv2.imread(args.image, cv2.IMREAD_COLOR)
    if img is None:
        raise SystemExit("failed to read image: {}".format(args.image))
    det = detect_frame(img, depth=None, history=deque(maxlen=1))
    debug = draw_debug(img, det)
    if args.output:
        os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
        cv2.imwrite(args.output, debug)
    if args.json:
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(det, f, indent=2, sort_keys=True)
    print(json.dumps({k: v for k, v in det.items() if k != "lines"}, indent=2, sort_keys=True))


def ros_main():
    import rospy
    from cv_bridge import CvBridge
    from sensor_msgs.msg import CameraInfo, Image
    from std_msgs.msg import String

    rospy.init_node("simenv_stair_visual_depth_detector")
    image_topic = rospy.get_param("~image_topic", "/real_sense/rgb/image_raw")
    depth_topic = rospy.get_param("~depth_topic", "/real_sense/depth/image_raw")
    camera_info_topic = rospy.get_param("~camera_info_topic", "/real_sense/rgb/camera_info")
    debug_topic = rospy.get_param("~debug_image_topic", "/simenv/stair_visual_debug")
    detection_topic = rospy.get_param("~detection_topic", "/simenv/stair_visual_detection")
    save_dir = rospy.get_param("~save_dir", "")
    save_every = int(rospy.get_param("~save_every", 5))
    # Keep the unannotated sensor frame alongside the diagnostic overlay.  This
    # is evidence for visual-model tuning; it is never consumed by navigation.
    save_raw = bool(rospy.get_param("~save_raw", True))
    history_len = int(rospy.get_param("~history_len", 12))
    fallback_fov_deg = float(rospy.get_param("~fallback_horizontal_fov_deg", 60.0))
    bridge = CvBridge()
    debug_pub = rospy.Publisher(debug_topic, Image, queue_size=1)
    det_pub = rospy.Publisher(detection_topic, String, queue_size=5)
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
    state = {
        "depth": None,
        "camera": None,
        "count": 0,
        "history": deque(maxlen=max(3, history_len)),
        "first_stable_saved": False,
    }

    def on_depth(msg):
        try:
            arr = bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
            state["depth"] = normalize_depth_image(arr)
        except Exception as e:
            rospy.logwarn_throttle(2.0, "depth conversion failed: %s", e)

    def on_camera(msg):
        k = msg.K
        state["camera"] = {"fx": float(k[0]), "fy": float(k[4]), "cx": float(k[2]), "cy": float(k[5])}

    def on_rgb(msg):
        try:
            img = bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except Exception as e:
            rospy.logwarn_throttle(2.0, "rgb conversion failed: %s", e)
            return
        camera = state["camera"]
        if camera is None:
            h, w = img.shape[:2]
            fx = w / (2.0 * math.tan(math.radians(fallback_fov_deg) * 0.5))
            camera = {
                "fx": float(fx),
                "fy": float(fx),
                "cx": float(w * 0.5),
                "cy": float(h * 0.5),
                "source": "fallback_fov_{:.1f}deg".format(fallback_fov_deg),
            }
        det = detect_frame(img, depth=state["depth"], camera_info=camera, history=state["history"])
        det["stamp"] = msg.header.stamp.to_sec()
        debug = draw_debug(img, det)
        out_msg = bridge.cv2_to_imgmsg(debug, encoding="bgr8")
        out_msg.header = msg.header
        debug_pub.publish(out_msg)
        public_det = {k: v for k, v in det.items() if k != "lines"}
        det_pub.publish(String(data=json.dumps(public_det, sort_keys=True)))
        n = state["count"]
        if save_dir and det.get("detected_stable") and not state["first_stable_saved"]:
            # Preserve one unambiguous evidence bundle for the first stable
            # first-riser observation.  Navigation never reads these files.
            cv2.imwrite(os.path.join(save_dir, "first_stable_first_step_raw.png"), img)
            cv2.imwrite(os.path.join(save_dir, "first_stable_first_step_debug.png"), debug)
            with open(os.path.join(save_dir, "first_stable_first_step.json"), "w", encoding="utf-8") as f:
                json.dump(det, f, indent=2, sort_keys=True)
            state["first_stable_saved"] = True
        # The first stable evidence bundle is already preserved above.  Keep
        # later diagnostics periodic instead of writing three files on every
        # stable frame; navigation consumes the ROS topic, not these files.
        if save_dir and n % max(1, save_every) == 0:
            base = os.path.join(save_dir, "stair_v2_{:06d}".format(n))
            cv2.imwrite(base + ".png", debug)
            if save_raw:
                cv2.imwrite(base + "_raw.png", img)
            with open(base + ".json", "w", encoding="utf-8") as f:
                json.dump(det, f, indent=2, sort_keys=True)
        state["count"] += 1

    rospy.Subscriber(depth_topic, Image, on_depth, queue_size=1, buff_size=2**24)
    rospy.Subscriber(camera_info_topic, CameraInfo, on_camera, queue_size=1)
    rospy.Subscriber(image_topic, Image, on_rgb, queue_size=1, buff_size=2**24)
    rospy.loginfo("stair visual-depth detector: rgb=%s depth=%s debug=%s", image_topic, depth_topic, debug_topic)
    rospy.spin()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image")
    ap.add_argument("--output")
    ap.add_argument("--json")
    ap.add_argument("--ros", action="store_true")
    args, _ = ap.parse_known_args()
    if args.ros or not args.image:
        ros_main()
    else:
        offline_main(args)


if __name__ == "__main__":
    main()
