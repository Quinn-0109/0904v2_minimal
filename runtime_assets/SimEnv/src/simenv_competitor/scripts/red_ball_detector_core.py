#!/usr/bin/env python3
"""Pure image-geometry helpers for the RGB-D red-sphere detector."""

import math


def oblique_sphere_shape_gate(
        aspect, circularity, fill_ratio, minimum_aspect, maximum_aspect,
        minimum_circularity, minimum_fill, maximum_fill):
    """Accept sphere-like oblique blobs while excluding solid box faces."""
    values = (aspect, circularity, fill_ratio, minimum_aspect,
              maximum_aspect, minimum_circularity, minimum_fill,
              maximum_fill)
    if not all(math.isfinite(float(value)) for value in values):
        return False
    return bool(
        float(minimum_aspect) <= float(aspect) <= float(maximum_aspect) and
        float(circularity) >= float(minimum_circularity) and
        float(minimum_fill) <= float(fill_ratio) <= float(maximum_fill))


def oblique_sphere_metric_gate(
        diameter, minimum_diameter, maximum_diameter,
        oblique_maximum_diameter):
    """Apply the tighter metric-size envelope for oblique candidates."""
    values = (diameter, minimum_diameter, maximum_diameter,
              oblique_maximum_diameter)
    if not all(math.isfinite(float(value)) for value in values):
        return False
    lower = float(minimum_diameter)
    upper = min(float(maximum_diameter), float(oblique_maximum_diameter))
    return bool(lower <= float(diameter) <= upper)


def edge_clipped_partial_sphere_hint(
        area, aspect, circularity, fill_ratio, x, width,
        image_width, edge_margin):
    """Return a direction-only hint for a sphere arc clipped by an image edge.

    The hint is never published as a hazard observation. It only lets the
    existing camera fan turn toward the clipped red arc and obtain an
    ordinary strict RGB-D sphere measurement.
    """
    values = (area, aspect, circularity, fill_ratio, x, width,
              image_width, edge_margin)
    if not all(math.isfinite(float(value)) for value in values):
        return False
    area = float(area)
    aspect = float(aspect)
    circularity = float(circularity)
    fill_ratio = float(fill_ratio)
    x = int(x)
    width = int(width)
    image_width = int(image_width)
    edge_margin = max(0, int(edge_margin))
    clipped = (x <= edge_margin or
               x + width >= image_width - edge_margin)
    return bool(
        clipped and area >= 100.0 and
        0.24 <= aspect < 0.80 and
        circularity >= 0.45 and
        0.58 <= fill_ratio <= 0.83)
