"""
au4_landmark_viewer.py
----------------------
Streamlit UI for visualizing MediaPipe Face Landmarker results
focused on AU4 (Brow Lowerer) analysis on a child frontal-face image.

Run:
    streamlit run au4_landmark_viewer.py

Install:
    pip install streamlit mediapipe Pillow numpy
"""

import os
import io
import numpy as np
import streamlit as st
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
from PIL import Image, ImageDraw, ImageFont

# ---------------------------------------------------------------------------
# Landmark index groups  (MediaPipe 478-point canonical face mesh)
# ---------------------------------------------------------------------------
RIGHT_EYEBROW    = [46, 53, 52, 65, 55, 70, 63, 105, 66, 107]
LEFT_EYEBROW     = [276, 283, 282, 295, 285, 300, 293, 334, 296, 336]

RIGHT_EYE        = [33, 7, 163, 144, 145, 153, 154, 155,
                    133, 246, 161, 160, 159, 158, 157, 173]
LEFT_EYE         = [263, 249, 390, 373, 374, 380, 381, 382,
                    362, 466, 388, 387, 386, 385, 384, 398]

RIGHT_INNER_BROW = [52, 55, 65]
LEFT_INNER_BROW  = [282, 285, 295]

RIGHT_OUTER_EYE_CORNER = 33
LEFT_OUTER_EYE_CORNER  = 263

# AU4 crop region: all brow indices that define the crop bounding box
AU4_CROP_INDICES = RIGHT_EYEBROW + LEFT_EYEBROW + RIGHT_INNER_BROW + LEFT_INNER_BROW

# ---------------------------------------------------------------------------
# Drawing constants  (RGB colour palette for Pillow)
# ---------------------------------------------------------------------------
C_RIGHT_BROW = (255, 255,   0)   # yellow
C_LEFT_BROW  = (255, 165,   0)   # orange
C_RIGHT_EYE  = (  0,   0, 255)   # blue
C_LEFT_EYE   = (200,   0, 180)   # purple
C_INNER_BROW = (255,   0,   0)   # red  (AU4 core)
C_OUTLINE    = (  0,   0,   0)   # black
C_WHITE      = (255, 255, 255)
C_DARK_BG    = ( 25,  25,  35)

DOT_RADIUS      = 5
DOT_RADIUS_CORE = 8
LINE_THICKNESS  = 2

MODEL_DOWNLOAD_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "face_landmarker/face_landmarker/float16/1/face_landmarker.task"
)

# Load fonts (Pillow 10+ supports load_default(size=N))
try:
    FONT_LABEL  = ImageFont.load_default(size=12)
    FONT_LEGEND = ImageFont.load_default(size=14)
    FONT_INFO   = ImageFont.load_default(size=15)
    FONT_TITLE  = ImageFont.load_default(size=17)
except TypeError:
    _f = ImageFont.load_default()
    FONT_LABEL = FONT_LEGEND = FONT_INFO = FONT_TITLE = _f

# ---------------------------------------------------------------------------
# Helper – run MediaPipe Face Landmarker (Tasks API)
# ---------------------------------------------------------------------------
def run_face_landmarker(image_rgb: np.ndarray, model_path: str):
    """
    Run MediaPipe Face Landmarker on an RGB uint8 ndarray.
    Returns a FaceLandmarkerResult.
    Raises FileNotFoundError if the model file is missing.
    """
    if not os.path.isfile(model_path):
        raise FileNotFoundError(
            f"Model file not found: '{model_path}'\n"
            f"Download it from:\n{MODEL_DOWNLOAD_URL}"
        )

    base_opts = mp_python.BaseOptions(
        model_asset_path=model_path,
        delegate=mp_python.BaseOptions.Delegate.CPU,
    )
    options   = mp_vision.FaceLandmarkerOptions(
        base_options=base_opts,
        output_face_blendshapes=False,
        output_facial_transformation_matrixes=False,
        num_faces=5,
        min_face_detection_confidence=0.5,
        min_face_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    with mp_vision.FaceLandmarker.create_from_options(options) as detector:
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
        result   = detector.detect(mp_image)

    return result


# ---------------------------------------------------------------------------
# Helper – convert normalised landmark → pixel (x, y)
# ---------------------------------------------------------------------------
def lm_px(lm, w: int, h: int):
    """Return (x_px, y_px) as integers."""
    return int(lm.x * w), int(lm.y * h)


# ---------------------------------------------------------------------------
# Helper – draw a landmark group with connecting polyline + optional labels
# ---------------------------------------------------------------------------
def _draw_group(draw: ImageDraw.ImageDraw, img_w: int, img_h: int,
                landmarks, indices, color,
                radius: int = DOT_RADIUS, label: bool = False,
                connect: bool = True):
    pts = [lm_px(landmarks[i], img_w, img_h) for i in indices]

    if connect and len(pts) > 1:
        for i in range(len(pts) - 1):
            draw.line([pts[i], pts[i + 1]], fill=color, width=LINE_THICKNESS)

    for i, idx in enumerate(indices):
        px, py = pts[i]
        r = radius
        draw.ellipse([(px-r-2, py-r-2), (px+r+2, py+r+2)], fill=C_OUTLINE)
        draw.ellipse([(px-r,   py-r),   (px+r,   py+r)],   fill=color)

        if label:
            text = str(idx)
            bbox = draw.textbbox((0, 0), text, font=FONT_LABEL)
            tw   = bbox[2] - bbox[0]
            th   = bbox[3] - bbox[1]
            tx   = px + r + 3
            ty   = py - r - th - 3
            draw.rectangle([(tx-1, ty-1), (tx+tw+1, ty+th+1)], fill=C_OUTLINE)
            draw.text((tx, ty), text, fill=C_WHITE, font=FONT_LABEL)


# ---------------------------------------------------------------------------
# Helper – draw legend with a filled background box
# ---------------------------------------------------------------------------
def _draw_legend(image_rgb: np.ndarray) -> np.ndarray:
    items = [
        (C_RIGHT_BROW, "Right eyebrow"),
        (C_LEFT_BROW,  "Left eyebrow"),
        (C_RIGHT_EYE,  "Right eye (ref)"),
        (C_LEFT_EYE,   "Left eye (ref)"),
        (C_INNER_BROW, "Inner brow – AU4 core"),
    ]
    ih, iw = image_rgb.shape[:2]
    pil_img = Image.fromarray(image_rgb)

    # Measure widest label first
    tmp_draw = ImageDraw.Draw(pil_img)
    max_tw   = max(tmp_draw.textbbox((0, 0), t, font=FONT_LEGEND)[2] for _, t in items)

    row_h = 24
    pad   = 8
    sw    = 16
    sh    = 12
    box_w = pad + sw + pad + max_tw + pad
    box_h = pad + len(items) * row_h + pad
    x0    = 10
    y0    = ih - box_h - 10

    # Semi-transparent dark background via blend
    overlay      = pil_img.copy()
    overlay_draw = ImageDraw.Draw(overlay)
    overlay_draw.rectangle([(x0, y0), (x0 + box_w, y0 + box_h)], fill=(30, 30, 30))
    pil_img = Image.blend(pil_img, overlay, alpha=0.75)

    draw = ImageDraw.Draw(pil_img)
    for i, (color, text) in enumerate(items):
        row_y = y0 + pad + i * row_h
        draw.rectangle([(x0+pad, row_y+2), (x0+pad+sw, row_y+2+sh)],
                       fill=color, outline=C_OUTLINE)
        draw.text((x0 + pad + sw + pad, row_y + 2), text,
                  fill=C_WHITE, font=FONT_LEGEND)

    return np.array(pil_img)


# ---------------------------------------------------------------------------
# Core – draw all AU4-relevant landmarks on a copy of image_rgb
# ---------------------------------------------------------------------------
def draw_au4_landmarks(image_rgb: np.ndarray, landmarks) -> np.ndarray:
    """Return an annotated RGB copy."""
    pil_img = Image.fromarray(image_rgb.copy())
    draw    = ImageDraw.Draw(pil_img)
    ih, iw  = image_rgb.shape[:2]

    _draw_group(draw, iw, ih, landmarks, RIGHT_EYE, C_RIGHT_EYE,
                radius=3, label=False, connect=True)
    _draw_group(draw, iw, ih, landmarks, LEFT_EYE, C_LEFT_EYE,
                radius=3, label=False, connect=True)
    _draw_group(draw, iw, ih, landmarks, RIGHT_EYEBROW, C_RIGHT_BROW,
                radius=DOT_RADIUS, label=True, connect=True)
    _draw_group(draw, iw, ih, landmarks, LEFT_EYEBROW, C_LEFT_BROW,
                radius=DOT_RADIUS, label=True, connect=True)
    _draw_group(draw, iw, ih, landmarks, RIGHT_INNER_BROW, C_INNER_BROW,
                radius=DOT_RADIUS_CORE, label=True, connect=True)
    _draw_group(draw, iw, ih, landmarks, LEFT_INNER_BROW, C_INNER_BROW,
                radius=DOT_RADIUS_CORE, label=True, connect=True)

    annotated = np.array(pil_img)
    return _draw_legend(annotated)


# ---------------------------------------------------------------------------
# Core – crop the AU4 region (eyebrow + glabella) with padding
# ---------------------------------------------------------------------------
def crop_au4_region(image_rgb: np.ndarray, landmarks,
                    pad_frac: float = 0.35) -> np.ndarray:
    """
    Return a tightly-padded RGB crop of the brow/glabella region.
    pad_frac: fractional padding relative to bounding-box size.
    """
    h, w = image_rgb.shape[:2]
    xs = [int(landmarks[i].x * w) for i in AU4_CROP_INDICES]
    ys = [int(landmarks[i].y * h) for i in AU4_CROP_INDICES]

    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)

    bw = x_max - x_min
    bh = y_max - y_min

    pad_x = max(int(bw * pad_frac), 10)
    pad_y = max(int(bh * pad_frac), 10)

    cx1 = max(x_min - pad_x, 0)
    cy1 = max(y_min - pad_y, 0)
    cx2 = min(x_max + pad_x, w)
    cy2 = min(y_max + pad_y, h)

    return image_rgb[cy1:cy2, cx1:cx2].copy()


# ---------------------------------------------------------------------------
# Core – compute AU4 geometric measurements
# ---------------------------------------------------------------------------
def compute_au4_measurements(landmarks, width: int, height: int) -> dict:
    """
    Return a dict of AU4-related geometric measurements (pixels + normalised).

    Brow–eye distance = mean(eye_y) – mean(brow_y).
    Smaller value = brow pulled toward eye = stronger AU4.
    """
    def mean_y(indices):
        return float(np.mean([landmarks[i].y * height for i in indices]))

    def mean_x(indices):
        return float(np.mean([landmarks[i].x * width for i in indices]))

    right_brow_y  = mean_y(RIGHT_EYEBROW)
    left_brow_y   = mean_y(LEFT_EYEBROW)
    right_eye_y   = mean_y(RIGHT_EYE)
    left_eye_y    = mean_y(LEFT_EYE)

    right_dist    = right_eye_y - right_brow_y
    left_dist     = left_eye_y  - left_brow_y
    mean_dist     = (right_dist + left_dist) / 2.0

    right_inner_x = mean_x(RIGHT_INNER_BROW)
    left_inner_x  = mean_x(LEFT_INNER_BROW)
    inner_brow_gap = abs(left_inner_x - right_inner_x)

    r_cx       = landmarks[RIGHT_OUTER_EYE_CORNER].x * width
    l_cx       = landmarks[LEFT_OUTER_EYE_CORNER].x  * width
    face_width = abs(l_cx - r_cx)
    safe_fw    = face_width if face_width > 0 else 1.0

    return {
        "right_brow_eye_dist_px":   right_dist,
        "left_brow_eye_dist_px":    left_dist,
        "mean_brow_eye_dist_px":    mean_dist,
        "inner_brow_gap_px":        inner_brow_gap,
        "face_width_px":            face_width,
        "right_brow_eye_dist_norm": right_dist     / safe_fw,
        "left_brow_eye_dist_norm":  left_dist      / safe_fw,
        "mean_brow_eye_dist_norm":  mean_dist      / safe_fw,
        "inner_brow_gap_norm":      inner_brow_gap / safe_fw,
    }


# ---------------------------------------------------------------------------
# Helper – resize image to a fixed height, preserving aspect ratio
# ---------------------------------------------------------------------------
def _resize_to_height(img: np.ndarray, target_h: int) -> np.ndarray:
    h, w = img.shape[:2]
    if h == 0:
        return img
    scale = target_h / h
    new_w = max(1, int(w * scale))
    return np.array(Image.fromarray(img).resize((new_w, target_h), Image.LANCZOS))


# ---------------------------------------------------------------------------
# Helper – render a measurement text block onto a dark panel
# ---------------------------------------------------------------------------
def _make_text_panel(measurements: dict, img_w: int, img_h: int,
                     num_faces: int, panel_h: int, panel_w: int) -> np.ndarray:
    panel = Image.new("RGB", (panel_w, panel_h), color=C_DARK_BG)
    draw  = ImageDraw.Draw(panel)

    m     = measurements
    lines = [
        ("AU4 Geometric Measurements",                                               FONT_TITLE, (220, 220, 100)),
        ("",                                                                          FONT_INFO,  C_WHITE),
        (f"Image size          : {img_w} x {img_h} px",                             FONT_INFO,  C_WHITE),
        (f"Faces detected      : {num_faces}",                                       FONT_INFO,  C_WHITE),
        ("",                                                                          FONT_INFO,  C_WHITE),
        (f"Right brow-eye dist : {m['right_brow_eye_dist_px']:6.1f} px  ({m['right_brow_eye_dist_norm']:.4f} norm)", FONT_INFO, C_WHITE),
        (f"Left  brow-eye dist : {m['left_brow_eye_dist_px']:6.1f} px  ({m['left_brow_eye_dist_norm']:.4f} norm)",  FONT_INFO, C_WHITE),
        (f"Mean  brow-eye dist : {m['mean_brow_eye_dist_px']:6.1f} px  ({m['mean_brow_eye_dist_norm']:.4f} norm)",  FONT_INFO, C_WHITE),
        (f"Inner brow gap      : {m['inner_brow_gap_px']:6.1f} px  ({m['inner_brow_gap_norm']:.4f} norm)",          FONT_INFO, C_WHITE),
        ("",                                                                          FONT_INFO,  C_WHITE),
        (f"Face width ref.     : {m['face_width_px']:6.1f} px",                     FONT_INFO,  C_WHITE),
        ("",                                                                          FONT_INFO,  C_WHITE),
        ("Lower brow-eye dist = stronger AU4 activation.",                           FONT_INFO,  (180, 220, 255)),
    ]

    y  = 25
    dy = 28
    for text, font, color in lines:
        if not text:
            y += dy // 2
            continue
        draw.text((20, y), text, fill=color, font=font)
        y += dy

    return np.array(panel)


# ---------------------------------------------------------------------------
# Helper – assemble the final combined output panel
# ---------------------------------------------------------------------------
def make_combined_panel(image_rgb: np.ndarray,
                        annotated_rgb: np.ndarray,
                        crop_rgb: np.ndarray,
                        measurements: dict,
                        img_h: int, img_w: int,
                        num_faces: int,
                        target_row_h: int = 400) -> np.ndarray:
    """
    Layout:
      Row 1 (side-by-side): original | overlay
      Row 2 (side-by-side): AU4 crop | measurement text panel
    Returns a single BGR image.
    """
    orig_r  = _resize_to_height(image_rgb,     target_row_h)
    anno_r  = _resize_to_height(annotated_rgb, target_row_h)

    # Row 1: original + overlay, same height
    divider_v = np.full((target_row_h, 6, 3), 180, dtype=np.uint8)
    row1 = np.hstack([orig_r, divider_v, anno_r])

    total_w = row1.shape[1]

    # Row 2: crop + text panel
    crop_h       = target_row_h
    crop_resized = _resize_to_height(crop_rgb, crop_h)
    crop_w_actual = crop_resized.shape[1]

    text_w = max(total_w - crop_w_actual - 6, 200)
    text_panel = _make_text_panel(measurements, img_w, img_h, num_faces,
                                  panel_h=crop_h, panel_w=text_w)

    row2 = np.hstack([crop_resized, divider_v[:crop_h, :], text_panel])

    # Pad rows to same width
    def _pad_to_width(img, target_w):
        h_img, w_img = img.shape[:2]
        if w_img >= target_w:
            return img[:, :target_w]
        pad = np.zeros((h_img, target_w - w_img, 3), dtype=np.uint8)
        pad[:] = (25, 25, 35)
        return np.hstack([img, pad])

    max_w  = max(row1.shape[1], row2.shape[1])
    row1   = _pad_to_width(row1, max_w)
    row2   = _pad_to_width(row2, max_w)

    divider_h = np.full((6, max_w, 3), 180, dtype=np.uint8)
    return np.vstack([row1, divider_h, row2])


# ---------------------------------------------------------------------------
# Streamlit application
# ---------------------------------------------------------------------------
def main():
    st.set_page_config(page_title="AU4 Landmark Viewer", layout="wide")
    st.title("AU4 (Brow Lowerer) Landmark Viewer")
    st.markdown(
        "Upload a **frontal child face image** to inspect MediaPipe Face Landmarker "
        "results for AU4 (brow lowerer). Eyebrow and inner brow landmarks are "
        "highlighted with index labels; AU4-related distances are shown below."
    )

    # ------------------------------------------------------------------
    # Sidebar – configuration
    # ------------------------------------------------------------------
    with st.sidebar:
        st.header("Configuration")

        model_path = st.text_input(
            "Path to `face_landmarker.task`",
            value="models/face_landmarker.task",
            help=f"Download the model from:\n{MODEL_DOWNLOAD_URL}",
        )

        output_path = st.text_input(
            "Output PNG path",
            value="output_au4_analysis_panel.png",
        )

        st.markdown("---")
        st.markdown("**Landmark colour key**")
        st.markdown("- 🟡 Right eyebrow")
        st.markdown("- 🟠 Left eyebrow")
        st.markdown("- 🔵 Right eye (reference)")
        st.markdown("- 🟣 Left eye (reference)")
        st.markdown("- 🔴 Inner brow – AU4 core")
        st.markdown("---")
        st.markdown(
            "**Interpretation:** A *smaller* brow–eye distance means the brow "
            "has moved toward the eye — stronger AU4 activation."
        )

    # ------------------------------------------------------------------
    # Main area – uploader + run button
    # ------------------------------------------------------------------
    uploaded_file = st.file_uploader(
        "Upload a frontal face image (JPG / PNG)",
        type=["jpg", "jpeg", "png"],
    )

    run_btn = st.button("▶  Run AU4 Detection", type="primary")

    if uploaded_file is None:
        st.info("Please upload an image to begin.")
        return

    if not run_btn:
        st.info("Image loaded. Press **Run AU4 Detection** to start.")
        return

    # ------------------------------------------------------------------
    # Decode image
    # ------------------------------------------------------------------
    try:
        pil_img   = Image.open(io.BytesIO(uploaded_file.read())).convert("RGB")
        image_rgb = np.array(pil_img)
    except Exception as exc:
        st.error(f"Failed to decode the uploaded image: {exc}")
        return

    h, w = image_rgb.shape[:2]

    # ------------------------------------------------------------------
    # Validate model path before running
    # ------------------------------------------------------------------
    if not os.path.isfile(model_path):
        st.error(
            f"Model file not found: `{model_path}`\n\n"
            f"Please download it from:\n{MODEL_DOWNLOAD_URL}\n\n"
            "Then set the correct path in the sidebar."
        )
        return

    # ------------------------------------------------------------------
    # Run MediaPipe
    # ------------------------------------------------------------------
    with st.spinner("Running MediaPipe Face Landmarker …"):
        try:
            result = run_face_landmarker(image_rgb, model_path)
        except FileNotFoundError as exc:
            st.error(str(exc))
            return
        except Exception as exc:
            st.error(f"MediaPipe error: {exc}")
            return

    num_faces = len(result.face_landmarks) if result.face_landmarks else 0

    # ------------------------------------------------------------------
    # Detection status banner
    # ------------------------------------------------------------------
    det_col1, det_col2, det_col3 = st.columns(3)
    det_col1.metric("Face Detected", "Yes" if num_faces > 0 else "No")
    det_col2.metric("Faces Found", num_faces)
    det_col3.metric("Image Size", f"{w} × {h} px")

    if num_faces == 0:
        st.warning(
            "No face detected in the uploaded image. "
            "Please use a clear, well-lit frontal face photo."
        )
        return

    if num_faces > 1:
        st.warning(
            f"{num_faces} faces detected — using only the first detected face."
        )

    landmarks = result.face_landmarks[0]

    # ------------------------------------------------------------------
    # Annotate + crop + measure
    # ------------------------------------------------------------------
    annotated_rgb = draw_au4_landmarks(image_rgb, landmarks)
    au4_crop_rgb  = crop_au4_region(annotated_rgb, landmarks)
    measurements  = compute_au4_measurements(landmarks, w, h)

    # ------------------------------------------------------------------
    # Row 1: Original | Full overlay  (top two columns)
    # ------------------------------------------------------------------
    st.markdown("---")
    col_orig, col_anno = st.columns(2)
    with col_orig:
        st.subheader("Original Image")
        st.image(image_rgb, use_column_width=True)
    with col_anno:
        st.subheader("AU4 Landmarks Overlay")
        st.image(annotated_rgb, use_column_width=True)

    # ------------------------------------------------------------------
    # Row 2: AU4 brow/glabella crop (below the two columns)
    # ------------------------------------------------------------------
    st.markdown("---")
    st.subheader("AU4 Region – Eyebrow & Glabella Zoom")
    st.image(au4_crop_rgb,
             caption="Zoomed crop: eyebrows + inner brow (AU4 core) region",
             use_column_width=False)

    # ------------------------------------------------------------------
    # Measurements section
    # ------------------------------------------------------------------
    st.markdown("---")
    st.subheader("AU4 Geometric Measurements")

    m = measurements
    c1, c2, c3, c4 = st.columns(4)
    c1.metric(
        label="Right Brow–Eye Dist",
        value=f"{m['right_brow_eye_dist_px']:.1f} px",
        delta=f"{m['right_brow_eye_dist_norm']:.4f} norm",
        delta_color="off",
    )
    c2.metric(
        label="Left Brow–Eye Dist",
        value=f"{m['left_brow_eye_dist_px']:.1f} px",
        delta=f"{m['left_brow_eye_dist_norm']:.4f} norm",
        delta_color="off",
    )
    c3.metric(
        label="Mean Brow–Eye Dist",
        value=f"{m['mean_brow_eye_dist_px']:.1f} px",
        delta=f"{m['mean_brow_eye_dist_norm']:.4f} norm",
        delta_color="off",
    )
    c4.metric(
        label="Inner Brow Gap",
        value=f"{m['inner_brow_gap_px']:.1f} px",
        delta=f"{m['inner_brow_gap_norm']:.4f} norm",
        delta_color="off",
    )

    st.caption(
        f"Face-width reference (outer eye corners): **{m['face_width_px']:.1f} px** · "
        "Normalised values are divided by face width. · "
        "Lower brow–eye distance = stronger AU4 activation."
    )

    with st.expander("Show all raw measurement values"):
        st.table([{"Measurement": k, "Value": f"{v:.4f}"} for k, v in m.items()])

    # ------------------------------------------------------------------
    # Save combined output panel
    # ------------------------------------------------------------------
    panel = make_combined_panel(
        image_rgb, annotated_rgb, au4_crop_rgb,
        measurements, h, w, num_faces,
    )
    try:
        Image.fromarray(panel).save(output_path)
        st.success(f"Combined analysis panel saved → `{output_path}`")
    except Exception as exc:
        st.warning(f"Could not save output image: {exc}")

    # Console log
    print("\n--- AU4 Measurements ---")
    for key, val in m.items():
        print(f"  {key:40s}: {val:.4f}")
    print("------------------------\n")


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    main()
