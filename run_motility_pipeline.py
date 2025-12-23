"""
Standalone tracking + motility + visualization pipeline for HSTLI.

Paper:
"HSTLI: A Dataset of Human Semen Time-Lapse Images for Detection, Tracking,
and Motility Parameter Analysis"

This script reproduces detection, tracking, motility computation,
and visualization (spider plots + density heatmaps).

------------------------------------------------------------
USER INSTRUCTIONS
------------------------------------------------------------
Only modify the parameters in the section:
    >>> USER CONFIGURATION (EDIT ONLY THIS SECTION) <<<

Everything else should be left unchanged.
"""

# ============================================================
# Matplotlib backend (headless-safe for GitHub / CI)
# ============================================================
import matplotlib
matplotlib.use("Agg")

# ============================================================
# Imports
# ============================================================
import os
import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
from math import atan2, degrees
from scipy.stats import gaussian_kde

from sort import Sort


# ============================================================
# >>> USER CONFIGURATION (EDIT ONLY THIS SECTION) <<<
# ============================================================

# ------------------------------------------------------------
# Tracking mode
# ------------------------------------------------------------
# Options: "yolo" or "ground_truth"
MODE = "yolo"   # <<< CHANGE ME IF NEEDED >>>

# ------------------------------------------------------------
# YOLO model configuration (used ONLY if MODE == "yolo")
# ------------------------------------------------------------
DETECTION_DOMAIN = "sys-opt"   # "sys-opt" or "sys-casa"
YOLO_SIZE = "n"               # "n", "s", "m", "l", or "x"

# ------------------------------------------------------------
# PATHS (ABSOLUTE OR RELATIVE)
# ------------------------------------------------------------

# Path to input video file
VIDEO_PATH = r"/path/to/video.mp4"   # <<< CHANGE ME >>>

# Path to ground-truth label directory (YOLO txt format)
GT_LABEL_DIR = r"/path/to/gt_labels" # <<< CHANGE ME (only if MODE == ground_truth) >>>

# Path to local YOLOv5 repository
YOLOV5_REPO = r"/path/to/yolov5"     # <<< CHANGE ME (only if MODE == yolo) >>>

# ------------------------------------------------------------
# Runtime parameters
# ------------------------------------------------------------
MAX_FRAMES = 10          # Number of frames to process (increase for full runs)
YOLO_CONF = 0.05         # YOLO confidence threshold
YOLO_IOU = 0.5           # YOLO NMS IoU threshold

WINDOW_SIZE = 10         # Sliding window size (frames)
WINDOW_OVERLAP = 0.2     # Overlap fraction between windows


# ============================================================
# ---------------- INTERNAL SETUP (DO NOT EDIT) ---------------
# ============================================================

if MODE not in {"yolo", "ground_truth"}:
    raise ValueError("MODE must be 'yolo' or 'ground_truth'")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

if MODE == "yolo":
    YOLO_WEIGHTS = os.path.join(
        SCRIPT_DIR,
        "weights",
        DETECTION_DOMAIN,
        f"{DETECTION_DOMAIN}_yolov5{YOLO_SIZE}.pt"
    )
    if not os.path.exists(YOLO_WEIGHTS):
        raise FileNotFoundError(f"YOLO weights not found: {YOLO_WEIGHTS}")

tracker = Sort(max_age=3, min_hits=3, iou_threshold=0.3)

if MODE == "yolo":
    yolo_model = torch.hub.load(
        YOLOV5_REPO,
        "custom",
        path=YOLO_WEIGHTS,
        source="local",
        verbose=False
    )
    yolo_model.conf = YOLO_CONF
    yolo_model.iou = YOLO_IOU
    yolo_model.eval()


# ============================================================
# ---------------- LOAD VIDEO --------------------------------
# ============================================================

cap = cv2.VideoCapture(VIDEO_PATH)
fps = cap.get(cv2.CAP_PROP_FPS)

frames = []
while cap.isOpened() and len(frames) < MAX_FRAMES:
    ret, frame = cap.read()
    if not ret:
        break
    frames.append(frame)
cap.release()

if not frames:
    raise RuntimeError("No frames loaded from video.")

height, width = frames[0].shape[:2]


# ============================================================
# ---------------- GROUND TRUTH LOADER ------------------------
# ============================================================

def load_ground_truth_boxes(path, w, h):
    boxes = []
    if not os.path.exists(path):
        return np.empty((0, 5))
    with open(path) as f:
        for line in f:
            _, cx, cy, bw, bh = map(float, line.split())
            x1 = (cx - bw / 2) * w
            y1 = (cy - bh / 2) * h
            x2 = (cx + bw / 2) * w
            y2 = (cy + bh / 2) * h
            boxes.append([x1, y1, x2, y2, 1.0])
    return np.array(boxes)


# ============================================================
# ---------------- TRACKING ----------------------------------
# ============================================================

tracks = {}

for frame_idx, frame in enumerate(frames):

    if MODE == "yolo":
        with torch.no_grad():
            results = yolo_model(frame)
        det = results.xyxy[0].cpu().numpy() if len(results.xyxy[0]) else np.empty((0, 6))
        detections = det[:, :5]
    else:
        gt_file = os.path.join(GT_LABEL_DIR, f"frame-{frame_idx}.txt")
        detections = load_ground_truth_boxes(gt_file, width, height)

    tracked = tracker.update(detections)

    for x1, y1, x2, y2, tid in tracked:
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        tracks.setdefault(str(int(tid)), {})
        tracks[str(int(tid))][frame_idx] = (cx, cy)


# ============================================================
# ---------------- MOTILITY COMPUTATION -----------------------
# ============================================================

def smooth_path(p, w=5):
    if len(p) < w:
        return p
    k = np.ones(w) / w
    return np.column_stack((
        np.convolve(p[:, 0], k, "same"),
        np.convolve(p[:, 1], k, "same")
    ))


def compute_segment_motility(p):
    diffs = np.diff(p, axis=0)
    if len(diffs) == 0:
        return None

    dt = 1.0 / fps
    dists = np.linalg.norm(diffs, axis=1)

    VCL = np.sum(dists) / (len(dists) * dt)
    VSL = np.linalg.norm(p[-1] - p[0]) / (len(dists) * dt)

    smooth = smooth_path(p)
    diffs_ap = np.diff(smooth, axis=0)
    VAP = np.sum(np.linalg.norm(diffs_ap, axis=1)) / (len(diffs_ap) * dt)

    LIN = VSL / VCL if VCL > 0 else np.nan
    WOB = VAP / VCL if VCL > 0 else np.nan
    STR = VSL / VAP if VAP > 0 else np.nan
    ALH = np.mean(np.linalg.norm(p - smooth, axis=1))

    ang = [degrees(atan2(d[1], d[0])) for d in diffs]
    MAD = np.mean(np.abs(np.diff(ang))) if len(ang) > 1 else np.nan

    return dict(VCL=VCL, VSL=VSL, VAP=VAP, LIN=LIN, ALH=ALH, WOB=WOB, STR=STR, MAD=MAD)


def compute_motility(tracks):
    step = max(1, int(WINDOW_SIZE * (1 - WINDOW_OVERLAP)))
    out = {}
    for tid, tr in tracks.items():
        frames = sorted(tr)
        if len(frames) < WINDOW_SIZE:
            continue
        pts = np.array([tr[f] for f in frames])
        out[tid] = []
        for i in range(0, len(pts) - WINDOW_SIZE + 1, step):
            m = compute_segment_motility(pts[i:i + WINDOW_SIZE])
            if m:
                out[tid].append(m)
    return out


motility = compute_motility(tracks)


# ============================================================
# ---------------- VISUALIZATION ------------------------------
# ============================================================

PARAMS = ["VCL", "VSL", "VAP", "LIN", "ALH", "WOB", "STR", "MAD"]
MAXVAL = dict(VCL=250, VSL=150, VAP=150, LIN=1, ALH=20, WOB=1, STR=1, MAD=180)

# ---- Spider Plot ----
agg = {k: [] for k in PARAMS}
for w in motility.values():
    for m in w:
        for k in PARAMS:
            if not np.isnan(m[k]):
                agg[k].append(m[k])

means = {k: np.mean(v) if v else 0 for k, v in agg.items()}
vals = [means[p] / MAXVAL[p] for p in PARAMS]

angles = np.linspace(0, 2*np.pi, len(PARAMS), endpoint=False)
angles = np.append(angles, angles[0])
vals.append(vals[0])

plt.figure(figsize=(8, 8))
ax = plt.subplot(111, polar=True)
ax.plot(angles, vals, linewidth=3)
ax.fill(angles, vals, alpha=0.25)
ax.set_xticks(angles[:-1])
ax.set_xticklabels(PARAMS)
ax.set_yticks([0.25, 0.5, 0.75, 1.0])
plt.title(f"Motility Spider Plot ({MODE.upper()})", pad=20)
plt.savefig("motility_spider_plot.png", dpi=300, bbox_inches="tight")
plt.close()

# ---- Density Heatmaps ----
PAIRS = [
    ("VSL", "VCL", (0, 150), (0, 250)),
    ("ALH", "LIN", (0, 20), (0, 1)),
    ("VSL", "WOB", (0, 150), (0, 1)),
    ("MAD", "LIN", (0, 180), (0, 1)),
]

fig, axes = plt.subplots(1, 4, figsize=(20, 5))
for ax, (xk, yk, xlim, ylim) in zip(axes, PAIRS):
    xs, ys = [], []
    for w in motility.values():
        for m in w:
            if not np.isnan(m[xk]) and not np.isnan(m[yk]):
                xs.append(m[xk])
                ys.append(m[yk])
    if not xs:
        continue
    dens = gaussian_kde(np.vstack([xs, ys]))(np.vstack([xs, ys]))
    sc = ax.scatter(xs, ys, c=dens, cmap="viridis", s=8)
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_xlabel(xk)
    ax.set_ylabel(yk)
    plt.colorbar(sc, ax=ax)

plt.suptitle(f"Motility Density Heatmaps ({MODE.upper()})", fontsize=16)
plt.savefig("motility_heatmaps.png", dpi=300, bbox_inches="tight")
plt.close()

print("Saved output figures:")
print(" - motility_spider_plot.png")
print(" - motility_heatmaps.png")
