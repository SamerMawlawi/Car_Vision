"""
Two-stage vehicle inspection pipeline.

Stage 1: Parts Detector  — locates car parts, crops each one.
Stage 2: Damage Detector — detects damages on each crop.
Result Merger             — maps coords back, infers missing parts.

Usage:
    from src.pipeline import InspectionPipeline
    pipe = InspectionPipeline()
    report = pipe.inspect("path/to/car_photo.jpg")
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from ultralytics import YOLO


ROOT = Path(__file__).resolve().parent.parent

# Default model paths (relative to project root)
DEFAULT_PARTS_MODEL  = ROOT / "models" / "Parts Detector (YOLO26)" / "weights" / "best.pt"
DEFAULT_DAMAGE_MODEL = ROOT / "models" / "CSK_Model" / "weights" / "epoch60.pt"

# Parts model class IDs 8..28 are the 21 car parts; 0..7 are damage (ignored).
PART_CLASS_OFFSET = 8

# Expected parts per view angle — used for missing-part inference.
# Keys are view tags; values are sets of part class names.
VIEW_INDICATORS: dict[str, set[str]] = {
    "front": {"Front-bumper", "Hood", "Grille", "Windshield"},
    "rear":  {"Back-bumper", "Trunk", "Back-windshield"},
    "side":  {"Front-door", "Back-door", "Fender", "Quarter-panel", "Rocker-panel"},
}
EXPECTED_PARTS: dict[str, set[str]] = {
    "front": {
        "Front-bumper", "Hood", "Headlight", "Grille", "Windshield",
        "Front-wheel", "Fender", "Mirror",
    },
    "rear": {
        "Back-bumper", "Trunk", "Tail-light", "Back-windshield",
        "Back-wheel", "Quarter-panel",
    },
    "side": {
        "Front-door", "Back-door", "Front-wheel", "Back-wheel",
        "Fender", "Quarter-panel", "Rocker-panel", "Mirror",
    },
}

# Crop padding factor (fraction of bbox size added on each side).
CROP_PAD = 0.15

# Per-class minimum confidence for the damage detector.
# Classes that hallucinate on crops get a higher bar.
DAMAGE_CONF_OVERRIDE: dict[str, float] = {
    "glass shatter": 0.75,  # fires on intact windshield texture
    "tire flat":     0.65,  # fires on normal tire close-ups
    "lamp broken":   0.55,  # fires on non-lamp parts
}

# Part-damage compatibility. If a damage type is listed here, it's ONLY
# allowed on the specified part types. Anything not listed is allowed everywhere.
DAMAGE_PART_FILTER: dict[str, set[str]] = {
    "tire flat":   {"Front-wheel", "Back-wheel"},
    "lamp broken": {"Headlight", "Tail-light"},
}

# IoU threshold for cross-crop deduplication.
DEDUP_IOU = 0.40


# ── Data classes ──────────────────────────────────────────────

@dataclass
class DamageDetection:
    damage_type: str
    confidence: float
    # Bbox in original image coords [x1, y1, x2, y2]
    bbox: list[int]
    # Segmentation mask as polygon points [[x,y], ...] in original image coords.
    # None if no mask available.
    mask_polygon: list[list[int]] | None = None

@dataclass
class PartResult:
    name: str
    bbox: list[int]
    confidence: float
    damages: list[DamageDetection] = field(default_factory=list)

@dataclass
class InspectionReport:
    image_path: str
    parts: list[PartResult]
    missing_parts: list[str]
    view_angles: list[str]
    full_image_damages: list[DamageDetection]

    def summary(self) -> str:
        """Human-readable summary."""
        lines = [f"Inspection: {Path(self.image_path).name}"]
        lines.append(f"View: {', '.join(self.view_angles) or 'unknown'}")
        lines.append(f"Parts detected: {len(self.parts)}")
        for p in self.parts:
            if p.damages:
                dmg = ", ".join(f"{d.damage_type} ({d.confidence:.0%})" for d in p.damages)
                lines.append(f"  {p.name}: {dmg}")
            else:
                lines.append(f"  {p.name}: OK")
        if self.missing_parts:
            lines.append(f"Missing parts: {', '.join(self.missing_parts)}")
        if self.full_image_damages:
            dmg = ", ".join(f"{d.damage_type} ({d.confidence:.0%})" for d in self.full_image_damages)
            lines.append(f"Full-image fallback damages: {dmg}")
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        """JSON-serializable dict for WebUI consumption."""
        return {
            "image": self.image_path,
            "view_angles": self.view_angles,
            "parts": [
                {
                    "name": p.name,
                    "bbox": p.bbox,
                    "confidence": round(p.confidence, 4),
                    "damages": [
                        {
                            "type": d.damage_type,
                            "confidence": round(d.confidence, 4),
                            "bbox": d.bbox,
                            "mask": d.mask_polygon,
                        }
                        for d in p.damages
                    ],
                }
                for p in self.parts
            ],
            "missing_parts": self.missing_parts,
            "full_image_damages": [
                {
                    "type": d.damage_type,
                    "confidence": round(d.confidence, 4),
                    "bbox": d.bbox,
                    "mask": d.mask_polygon,
                }
                for d in self.full_image_damages
            ],
        }


# ── Pipeline ──────────────────────────────────────────────────

class InspectionPipeline:
    """Orchestrates the two-stage inspection."""

    def __init__(
        self,
        parts_model: str | Path = DEFAULT_PARTS_MODEL,
        damage_model: str | Path = DEFAULT_DAMAGE_MODEL,
        parts_conf: float = 0.35,
        damage_conf: float = 0.30,
        imgsz_parts: int = 1024,
        imgsz_damage_crop: int = 640,   # damage detector on part crops (small input)
        imgsz_damage_full: int = 1280,  # damage detector on full image fallback (matches CSK training res)
        crop_pad: float = CROP_PAD,
        run_full_image_fallback: bool = True,
    ):
        self.parts_detector = YOLO(str(parts_model))
        self.damage_detector = YOLO(str(damage_model))
        self.parts_conf = parts_conf
        self.damage_conf = damage_conf
        self.imgsz_parts = imgsz_parts
        self.imgsz_damage_crop = imgsz_damage_crop
        self.imgsz_damage_full = imgsz_damage_full
        self.crop_pad = crop_pad
        self.run_fallback = run_full_image_fallback

        # Build a name→id lookup for parts (classes 8..28).
        self._part_names: dict[int, str] = {
            cid: name
            for cid, name in self.parts_detector.names.items()
            if cid >= PART_CLASS_OFFSET
        }
        self._damage_names: dict[int, str] = self.damage_detector.names

    # ── public API ────────────────────────────────────────────

    def inspect(self, image_path: str | Path) -> InspectionReport:
        """Run full inspection on one image. Returns a structured report."""
        img = cv2.imread(str(image_path))
        if img is None:
            raise FileNotFoundError(f"Cannot read image: {image_path}")
        H, W = img.shape[:2]

        # Stage 1: detect parts
        part_results = self._detect_parts(img)

        # Stage 2: detect damage on each cropped part
        for part in part_results:
            crop, offx, offy = self._crop(img, part.bbox, W, H)
            detections = self._detect_damage(crop, self.imgsz_damage_crop)
            # Map crop-local coords back to original image coords.
            for d in detections:
                d.bbox = [
                    d.bbox[0] + offx, d.bbox[1] + offy,
                    d.bbox[2] + offx, d.bbox[3] + offy,
                ]
                if d.mask_polygon is not None:
                    d.mask_polygon = [
                        [pt[0] + offx, pt[1] + offy] for pt in d.mask_polygon
                    ]
            # Filter: per-class confidence override
            detections = [
                d for d in detections
                if d.confidence >= DAMAGE_CONF_OVERRIDE.get(d.damage_type, 0.0)
            ]
            # Filter: part-damage compatibility (e.g. tire_flat only on wheels)
            detections = [
                d for d in detections
                if d.damage_type not in DAMAGE_PART_FILTER
                or part.name in DAMAGE_PART_FILTER[d.damage_type]
            ]
            part.damages = detections

        # Full-image fallback (catches damage between parts or on missed parts)
        full_damages: list[DamageDetection] = []
        if self.run_fallback:
            full_damages = self._detect_damage(img, self.imgsz_damage_full)
            # Apply same per-class conf override to fallback detections
            full_damages = [
                d for d in full_damages
                if d.confidence >= DAMAGE_CONF_OVERRIDE.get(d.damage_type, 0.0)
            ]

        # Deduplicate: remove fallback detections that overlap with crop detections
        crop_dets = [d for p in part_results for d in p.damages]
        full_damages = self._dedup_fallback(crop_dets, full_damages)

        # Infer view angle and missing parts
        detected_names = {p.name for p in part_results}
        views = self._infer_views(detected_names)
        missing = self._infer_missing(detected_names, views)

        return InspectionReport(
            image_path=str(image_path),
            parts=part_results,
            missing_parts=missing,
            view_angles=views,
            full_image_damages=full_damages,
        )

    def inspect_batch(
        self, image_paths: list[str | Path], verbose: bool = True,
    ) -> list[InspectionReport]:
        """Run inspection on multiple images."""
        reports: list[InspectionReport] = []
        for i, p in enumerate(image_paths):
            if verbose:
                print(f"[{i+1}/{len(image_paths)}] {Path(p).name}")
            reports.append(self.inspect(p))
        return reports

    # ── Stage 1 ───────────────────────────────────────────────

    def _detect_parts(self, img: np.ndarray) -> list[PartResult]:
        results = self.parts_detector.predict(
            img, imgsz=self.imgsz_parts, conf=self.parts_conf, verbose=False,
        )
        parts: list[PartResult] = []
        if not results or results[0].boxes is None:
            return parts

        r = results[0]
        for box, cls_id, conf in zip(
            r.boxes.xyxy.cpu().numpy(),
            r.boxes.cls.cpu().numpy().astype(int),
            r.boxes.conf.cpu().numpy(),
        ):
            if cls_id not in self._part_names:
                continue  # skip damage classes from the 29-class model
            parts.append(PartResult(
                name=self._part_names[cls_id],
                bbox=[int(v) for v in box],
                confidence=float(conf),
            ))
        return parts

    # ── Stage 2 ───────────────────────────────────────────────

    def _detect_damage(self, img: np.ndarray, imgsz: int = 640) -> list[DamageDetection]:
        results = self.damage_detector.predict(
            img, imgsz=imgsz, conf=self.damage_conf, verbose=False,
        )
        detections: list[DamageDetection] = []
        if not results or results[0].boxes is None:
            return detections

        r = results[0]
        # r.masks.xy is a list of numpy arrays, each shape (N, 2) in pixel coords.
        has_masks = r.masks is not None and r.masks.xy is not None
        masks_xy = r.masks.xy if has_masks else [None] * len(r.boxes)

        for box, cls_id, conf, mask in zip(
            r.boxes.xyxy.cpu().numpy(),
            r.boxes.cls.cpu().numpy().astype(int),
            r.boxes.conf.cpu().numpy(),
            masks_xy,
        ):
            poly = None
            if mask is not None and len(mask) >= 3:
                poly = [[int(x), int(y)] for x, y in mask]
            detections.append(DamageDetection(
                damage_type=self._damage_names[cls_id],
                confidence=float(conf),
                bbox=[int(v) for v in box],
                mask_polygon=poly,
            ))
        return detections

    # ── Cropping ──────────────────────────────────────────────

    def _crop(
        self, img: np.ndarray, bbox: list[int], W: int, H: int,
    ) -> tuple[np.ndarray, int, int]:
        """Crop bbox from image with padding. Returns (crop, offset_x, offset_y)."""
        x1, y1, x2, y2 = bbox
        bw, bh = x2 - x1, y2 - y1
        pad_x = int(bw * self.crop_pad)
        pad_y = int(bh * self.crop_pad)

        # Clamp to image bounds
        cx1 = max(x1 - pad_x, 0)
        cy1 = max(y1 - pad_y, 0)
        cx2 = min(x2 + pad_x, W)
        cy2 = min(y2 + pad_y, H)

        crop = img[cy1:cy2, cx1:cx2]
        return crop, cx1, cy1

    # ── Deduplication ────────────────────────────────────────

    @staticmethod
    def _iou(a: list[int], b: list[int]) -> float:
        """Intersection-over-union for two [x1,y1,x2,y2] boxes."""
        ix1 = max(a[0], b[0]); iy1 = max(a[1], b[1])
        ix2 = min(a[2], b[2]); iy2 = min(a[3], b[3])
        inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
        area_a = (a[2] - a[0]) * (a[3] - a[1])
        area_b = (b[2] - b[0]) * (b[3] - b[1])
        union = area_a + area_b - inter
        return inter / union if union > 0 else 0.0

    @staticmethod
    def _dedup_fallback(
        crop_dets: list[DamageDetection], fallback_dets: list[DamageDetection],
    ) -> list[DamageDetection]:
        """Remove fallback detections that overlap with any crop detection."""
        kept: list[DamageDetection] = []
        for fd in fallback_dets:
            overlaps = any(
                InspectionPipeline._iou(fd.bbox, cd.bbox) > DEDUP_IOU
                for cd in crop_dets
            )
            if not overlaps:
                kept.append(fd)
        return kept

    # ── Missing-part logic ────────────────────────────────────

    @staticmethod
    def _infer_views(detected: set[str]) -> list[str]:
        """Determine which view angles are present based on detected parts."""
        views: list[str] = []
        for view, indicators in VIEW_INDICATORS.items():
            # Require >=3 indicator parts to confidently declare a view.
            # (2 was too aggressive — led to false missing-part flags.)
            if len(detected & indicators) >= 3:
                views.append(view)
        return views or ["unknown"]

    @staticmethod
    def _infer_missing(detected: set[str], views: list[str]) -> list[str]:
        """Parts expected for the detected views but not found.
        Only flag large, obvious parts. Small/easy-to-occlude parts
        (Mirror, License-plate, Grille) are excluded — their absence
        is more likely a detection miss than an actually missing part."""
        if "unknown" in views:
            return []  # don't guess when we can't determine the view

        HIGH_CONF_PARTS: dict[str, set[str]] = {
            "front": {"Front-bumper", "Hood", "Headlight", "Windshield"},
            "rear":  {"Back-bumper", "Trunk", "Tail-light"},
            "side":  {"Front-door", "Front-wheel", "Back-wheel"},
        }
        expected: set[str] = set()
        for v in views:
            expected |= HIGH_CONF_PARTS.get(v, set())
        missing = sorted(expected - detected)
        return missing
