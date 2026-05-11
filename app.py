"""
MiniVision Flask server.
Serves the frontend and exposes /api/analyze for the two-stage damage pipeline.

Model loading strategy (in priority order):
  1. Local paths (development) — models/Parts Detector .../best.pt + models/CSK_Model/.../epoch60.pt
  2. HF Hub download (production on HF Spaces) — set HF_MODEL_REPO env var to your model repo id
     e.g.  HF_MODEL_REPO=YourUsername/minivision-models
"""
from __future__ import annotations

import os
import sys
import uuid
from pathlib import Path

import cv2
import numpy as np
from flask import Flask, jsonify, request, send_file, send_from_directory

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from pipeline import InspectionPipeline  # noqa: E402


def _resolve_models() -> tuple[Path, Path]:
    """Return (parts_weights, damage_weights) — local first, HF Hub fallback."""
    local_parts  = ROOT / "models" / "Parts Detector (YOLO26)" / "weights" / "best.pt"
    local_damage = ROOT / "models" / "CSK_Model" / "weights" / "epoch60.pt"

    if local_parts.exists() and local_damage.exists():
        print("Using local model weights.")
        return local_parts, local_damage

    # HF Hub fallback
    repo_id = os.environ.get("HF_MODEL_REPO", "")
    if not repo_id:
        raise RuntimeError(
            "Model weights not found locally and HF_MODEL_REPO env var is not set.\n"
            "Set HF_MODEL_REPO=YourUsername/minivision-models"
        )

    print(f"Downloading weights from HF Hub: {repo_id} …")
    from huggingface_hub import hf_hub_download
    cache = ROOT / "model_cache"
    cache.mkdir(exist_ok=True)

    parts_pt  = hf_hub_download(repo_id=repo_id, filename="parts_best.pt",  local_dir=str(cache))
    damage_pt = hf_hub_download(repo_id=repo_id, filename="epoch60.pt",      local_dir=str(cache))
    print("Weights downloaded.")
    return Path(parts_pt), Path(damage_pt)


app = Flask(__name__, static_folder="static", static_url_path="/static")

# ── Load pipeline once at startup (keeps models in VRAM / RAM) ───────────────
print("Loading inspection pipeline…")
parts_weights, damage_weights = _resolve_models()
pipe = InspectionPipeline(parts_model=parts_weights, damage_model=damage_weights)
print("Pipeline ready.")

# ── Colour palette per damage type (BGR for OpenCV) ──────────────────────────
DAMAGE_COLORS: dict[str, tuple[int, int, int]] = {
    "dent":          (50,  100, 255),   # orange
    "scratch":       (255, 200,  50),   # cyan
    "crack":         (80,  255,  80),   # green
    "glass shatter": (255,  80, 200),   # purple
    "lamp broken":   (50,  255, 255),   # yellow
    "tire flat":     (100,  50, 255),   # red
}
FONT = cv2.FONT_HERSHEY_SIMPLEX


# ── Helpers ──────────────────────────────────────────────────────────────────

def compute_severity(mask_polygon: list | None, img_area: int) -> str:
    """Area-based severity: ratio of mask to full image."""
    if not mask_polygon or len(mask_polygon) < 3:
        return "minor"
    area = cv2.contourArea(np.array(mask_polygon, dtype=np.float32))
    ratio = area / max(img_area, 1)
    if ratio > 0.015:
        return "critical"
    elif ratio > 0.004:
        return "medium"
    return "minor"


def draw_annotated(img_path: Path, report, out_path: Path) -> None:
    """Draw part boxes + damage masks onto the image and save."""
    img = cv2.imread(str(img_path))
    if img is None:
        # Try reading with imdecode for webp/unusual formats
        img = cv2.imdecode(np.fromfile(str(img_path), dtype=np.uint8), cv2.IMREAD_COLOR)
    overlay = img.copy()
    H, W = img.shape[:2]

    # Part bounding boxes (grey, thin)
    for part in report.parts:
        x1, y1, x2, y2 = part.bbox
        cv2.rectangle(img, (x1, y1), (x2, y2), (180, 180, 180), 1)
        cv2.putText(img, part.name, (x1 + 3, y1 + 13),
                    FONT, 0.36, (180, 180, 180), 1, cv2.LINE_AA)

    # Damage polygons
    all_damages = [(d, p.name) for p in report.parts for d in p.damages]
    all_damages += [(d, "scene") for d in report.full_image_damages]

    for d, part_name in all_damages:
        color = DAMAGE_COLORS.get(d.damage_type, (255, 255, 255))
        if d.mask_polygon and len(d.mask_polygon) >= 3:
            pts = np.array(d.mask_polygon, dtype=np.int32)
            cv2.fillPoly(overlay, [pts], color)
            cv2.polylines(img, [pts], True, color, 2, cv2.LINE_AA)
        else:
            x1, y1, x2, y2 = d.bbox
            cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

        # Label
        x1l, y1l = d.bbox[0], d.bbox[1]
        label = f"{d.damage_type} {d.confidence:.0%}"
        (tw, th), _ = cv2.getTextSize(label, FONT, 0.42, 1)
        cv2.rectangle(img, (x1l, y1l - th - 6), (x1l + tw + 4, y1l), color, -1)
        cv2.putText(img, label, (x1l + 2, y1l - 3),
                    FONT, 0.42, (0, 0, 0), 1, cv2.LINE_AA)

    cv2.addWeighted(overlay, 0.38, img, 0.62, 0, img)

    # Top info bar
    view_str = "View: " + ", ".join(report.view_angles)
    miss_str = "Missing: " + (", ".join(report.missing_parts) if report.missing_parts else "none")
    cv2.rectangle(img, (0, 0), (W, 28), (20, 20, 20), -1)
    cv2.putText(img, f"{view_str}   |   {miss_str}",
                (6, 18), FONT, 0.46, (240, 240, 240), 1, cv2.LINE_AA)

    cv2.imwrite(str(out_path), img, [cv2.IMWRITE_JPEG_QUALITY, 92])


def report_to_json(report, img_area: int, annotated_url: str, input_url: str) -> dict:
    """Convert InspectionReport to a JSON-serialisable dict with severity scores."""
    parts_out = []
    for p in report.parts:
        damages_out = []
        for d in p.damages:
            damages_out.append({
                "damage_type": d.damage_type,
                "confidence":  round(d.confidence, 4),
                "bbox":        d.bbox,
                "severity":    compute_severity(d.mask_polygon, img_area),
            })
        parts_out.append({
            "name":       p.name,
            "bbox":       p.bbox,
            "confidence": round(p.confidence, 4),
            "damages":    damages_out,
        })

    full_out = []
    for d in report.full_image_damages:
        full_out.append({
            "damage_type": d.damage_type,
            "confidence":  round(d.confidence, 4),
            "bbox":        d.bbox,
            "severity":    compute_severity(d.mask_polygon, img_area),
        })

    return {
        "annotated_image":    annotated_url,
        "input_image":        input_url,
        "view_angles":        report.view_angles,
        "missing_parts":      report.missing_parts,
        "parts":              parts_out,
        "full_image_damages": full_out,
    }


# ── Routes ───────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return send_file(ROOT / "index.html")


@app.route("/pages/<path:page>")
def pages(page: str):
    return send_from_directory(ROOT / "pages", page)


@app.route("/api/analyze", methods=["POST"])
def analyze():
    results_dir = ROOT / "static" / "results"
    results_dir.mkdir(exist_ok=True)
    uid = uuid.uuid4().hex[:12]

    # ── Determine input image path ────────────────────────────────────────
    if request.content_type and "application/json" in request.content_type:
        data = request.get_json(force=True)
        example_name = data.get("example", "")
        img_path = ROOT / "static" / "examples" / Path(example_name).name
        if not img_path.exists():
            return jsonify({"error": f"Example not found: {example_name}"}), 404
        # For examples we don't need to copy — serve the original
        input_url = f"/static/examples/{img_path.name}"
    else:
        file = request.files.get("image")
        if not file:
            return jsonify({"error": "No image provided"}), 400
        suffix = Path(file.filename).suffix or ".jpg"
        input_path = results_dir / f"{uid}_input{suffix}"
        file.save(str(input_path))
        img_path = input_path
        input_url = f"/static/results/{input_path.name}"

    # ── Run pipeline ──────────────────────────────────────────────────────
    try:
        report = pipe.inspect(img_path)
    except Exception as e:
        return jsonify({"error": f"Pipeline error: {e}"}), 500

    # ── Draw annotated image ──────────────────────────────────────────────
    annotated_path = results_dir / f"{uid}_annotated.jpg"
    draw_annotated(img_path, report, annotated_path)
    annotated_url = f"/static/results/{annotated_path.name}"

    # ── Image area for severity ───────────────────────────────────────────
    raw = cv2.imread(str(img_path))
    if raw is None:
        raw = cv2.imdecode(np.fromfile(str(img_path), dtype=np.uint8), cv2.IMREAD_COLOR)
    img_area = raw.shape[0] * raw.shape[1] if raw is not None else 1_000_000

    return jsonify(report_to_json(report, img_area, annotated_url, input_url))


# ── Run ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # HF Spaces requires port 7860; local dev uses 5000
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
