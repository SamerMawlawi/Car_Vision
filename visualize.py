"""
Visual output for the inspection pipeline.
Draws part bounding boxes, damage MASKS (filled polygons with transparency),
and missing-part labels onto the original image.
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from src.pipeline import InspectionReport

# Color palette (BGR) — parts in muted tones, damages in vivid tones.
PART_COLOR   = (180, 180, 180)  # light grey border for parts
DAMAGE_COLORS = {
    "dent":          (255, 100, 0),    # blue
    "scratch":       (0, 220, 220),    # yellow/cyan
    "crack":         (0, 0, 255),      # red
    "glass shatter": (200, 0, 200),    # magenta
    "lamp broken":   (0, 165, 255),    # orange
    "tire flat":     (255, 0, 128),    # pink
}
MISSING_COLOR = (0, 0, 200)  # dark red text
MASK_ALPHA = 0.40  # transparency for filled mask overlay


def _draw_mask(img: np.ndarray, polygon: list[list[int]], color: tuple) -> None:
    """Draw a semi-transparent filled polygon mask onto the image."""
    pts = np.array(polygon, dtype=np.int32)
    overlay = img.copy()
    cv2.fillPoly(overlay, [pts], color)
    cv2.addWeighted(overlay, MASK_ALPHA, img, 1 - MASK_ALPHA, 0, dst=img)
    # Draw the polygon outline on top for crisp edges
    cv2.polylines(img, [pts], isClosed=True, color=color, thickness=2, lineType=cv2.LINE_AA)


def _draw_damage(img: np.ndarray, d, color: tuple, font_scale: float, thick: int, tag: str = "") -> None:
    """Draw a single damage detection — mask if available, box as fallback."""
    if d.mask_polygon is not None and len(d.mask_polygon) >= 3:
        _draw_mask(img, d.mask_polygon, color)
        # Place label near the top of the mask bounding rect
        xs = [pt[0] for pt in d.mask_polygon]
        ys = [pt[1] for pt in d.mask_polygon]
        lx, ly = min(xs), min(ys) - 4
    else:
        # Fallback to bbox if no mask
        dx1, dy1, dx2, dy2 = d.bbox
        cv2.rectangle(img, (dx1, dy1), (dx2, dy2), color, thick + 1)
        lx, ly = dx1, dy1 - 4

    label = f"{tag}{d.damage_type} {d.confidence:.0%}"
    _put_label(img, label, lx, ly, font_scale, color, thick)


def draw_report(image_path: str, report: InspectionReport) -> np.ndarray:
    """Draw all pipeline results onto the image. Returns the annotated image."""
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(image_path)

    H, W = img.shape[:2]
    scale = max(W, H) / 1200
    thick = max(1, int(scale * 2))
    font_scale = max(0.4, scale * 0.5)

    # Draw part bounding boxes (thin grey)
    for part in report.parts:
        x1, y1, x2, y2 = part.bbox
        cv2.rectangle(img, (x1, y1), (x2, y2), PART_COLOR, thick)
        label = f"{part.name}"
        _put_label(img, label, x1, y1 - 4, font_scale * 0.8, PART_COLOR, max(1, thick - 1))

        # Draw damage masks inside this part
        for d in part.damages:
            color = DAMAGE_COLORS.get(d.damage_type, (255, 255, 255))
            _draw_damage(img, d, color, font_scale, thick)

    # Draw full-image fallback damages
    for d in report.full_image_damages:
        color = DAMAGE_COLORS.get(d.damage_type, (255, 255, 255))
        _draw_damage(img, d, color, font_scale * 0.8, max(1, thick - 1), tag="[F] ")

    # View + missing parts text block (top-left)
    y_text = int(30 * scale)
    view_text = f"View: {', '.join(report.view_angles)}"
    _put_label(img, view_text, 10, y_text, font_scale, (255, 255, 255), thick)

    if report.missing_parts:
        y_text += int(28 * scale)
        miss_text = f"MISSING: {', '.join(report.missing_parts)}"
        _put_label(img, miss_text, 10, y_text, font_scale, MISSING_COLOR, thick)

    return img


def _put_label(
    img: np.ndarray, text: str, x: int, y: int,
    font_scale: float, color: tuple, thickness: int,
) -> None:
    """Put text with a dark background strip for readability."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    (tw, th), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    cv2.rectangle(img, (x, y - th - 4), (x + tw + 4, y + 4), (0, 0, 0), cv2.FILLED)
    cv2.putText(img, text, (x + 2, y), font, font_scale, color, thickness, cv2.LINE_AA)


def save_pipeline_results(
    reports: list[InspectionReport],
    output_dir: str | Path,
    max_images: int | None = None,
) -> Path:
    """Save annotated images for a batch of reports."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    count = 0
    for r in reports:
        if max_images and count >= max_images:
            break
        img = draw_report(r.image_path, r)
        name = Path(r.image_path).stem + ".jpg"
        cv2.imwrite(str(out / name), img)
        count += 1
    print(f"Saved {count} annotated images to {out}")
    return out
