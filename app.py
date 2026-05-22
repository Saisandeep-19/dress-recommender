import os
import cv2
import numpy as np
import mediapipe as mp
import streamlit as st
from PIL import Image
import tempfile
import traceback
import onnxruntime as ort

st.set_page_config(page_title="Skin Tone Analyzer", page_icon="🎨", layout="centered")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display&family=DM+Sans:wght@300;400;500&display=swap');
html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; background-color: #fdf6f0; color: #1a1a1a; }
h1, h2, h3 { font-family: 'DM Serif Display', serif; }
.stApp { background-color: #fdf6f0; }
.hero { text-align: center; padding: 2.5rem 1rem 1rem; }
.hero h1 { font-size: 2.8rem; color: #2b1a0e; margin-bottom: 0.3rem; }
.hero p  { color: #7a5c44; font-size: 1.05rem; font-weight: 300; }
.card { background: #fff8f3; border: 1px solid #e8d5c4; border-radius: 16px; padding: 1.6rem; margin: 1rem 0; }
.tone-badge { display: inline-block; padding: 0.35rem 1.1rem; border-radius: 999px; font-weight: 500; font-size: 0.95rem; background: #2b1a0e; color: #fdf6f0; margin-bottom: 0.5rem; }
.undertone-badge { display: inline-block; padding: 0.3rem 1rem; border-radius: 999px; font-size: 0.9rem; font-weight: 500; margin-left: 0.5rem; }
.warm    { background: #f5e0c3; color: #7a3e00; }
.cool    { background: #c3d9f5; color: #003d7a; }
.neutral { background: #e2e2e2; color: #333; }
.prob-bar-wrap { margin: 0.4rem 0; }
.prob-label { display: flex; justify-content: space-between; font-size: 0.85rem; color: #5a3e2b; margin-bottom: 2px; }
.prob-bar-bg { background: #ecddd3; border-radius: 999px; height: 8px; width: 100%; }
.prob-bar-fill { height: 8px; border-radius: 999px; background: linear-gradient(90deg, #c87941, #e8a070); }
.swatch-row { display: flex; flex-wrap: wrap; gap: 10px; margin-top: 0.8rem; }
.swatch { display: flex; flex-direction: column; align-items: center; gap: 5px; }
.swatch-circle { width: 52px; height: 52px; border-radius: 50%; border: 2px solid #e8d5c4; box-shadow: 0 2px 8px rgba(0,0,0,0.1); }
.swatch-name { font-size: 0.7rem; color: #5a3e2b; text-align: center; max-width: 60px; text-transform: capitalize; }
.swatch-score { font-size: 0.65rem; color: #9a7a64; }
.divider { border: none; border-top: 1px solid #e8d5c4; margin: 1.2rem 0; }
.section-title { font-family: 'DM Serif Display', serif; font-size: 1.1rem; color: #2b1a0e; margin-bottom: 0.8rem; }
footer { display: none; }
</style>
""", unsafe_allow_html=True)

# ── Constants ─────────────────────────────────────────────────────────────────
MODEL_PATH = "skintone_model.onnx"
cls_names  = ['dark', 'fair', 'light']
INPUT_NAME = "keras_tensor_352"

COLOR_MAP = {
    "white": "#FFFFFF", "cream": "#FFFDD0", "yellow": "#FFD700",
    "cyan": "#00CED1", "mint": "#98FF98", "sky blue": "#87CEEB",
    "bright red": "#FF2400", "lavender": "#E6E6FA", "pastel pink": "#FFD1DC",
    "light gray": "#D3D3D3", "maroon": "#800000", "teal": "#008080",
    "olive": "#808000", "charcoal": "#36454F", "mustard": "#FFDB58",
    "denim blue": "#1560BD", "forest green": "#228B22", "deep purple": "#673AB7",
    "navy": "#001F5B", "black": "#1a1a1a", "emerald": "#50C878",
    "burgundy": "#800020", "royal blue": "#4169E1", "deep green": "#006400",
    "rust": "#B7410E", "gold": "#FFD700", "brown": "#8B4513",
    "beige": "#F5F5DC", "coral": "#FF7F50", "silver": "#C0C0C0",
    "purple": "#800080", "gray": "#808080", "taupe": "#483C32",
}

L_CHK = [234, 93, 132, 58, 172]
R_CHK = [454, 323, 361, 288, 397]
FORH  = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397]

POOLS = {
    "dark": ["white","cream","yellow","cyan","mint","sky blue","bright red","lavender","pastel pink","light gray"],
    "mid":  ["maroon","teal","olive","charcoal","mustard","denim blue","forest green","deep purple"],
    "fair": ["navy","black","emerald","burgundy","royal blue","charcoal","maroon","teal","deep green"],
}
UNDERTONE_BONUS = {
    "warm":    ["olive","mustard","rust","gold","brown","beige","coral"],
    "cool":    ["navy","silver","emerald","purple","teal","charcoal"],
    "neutral": ["black","white","gray","denim blue","cream","taupe"],
}

# ── Model loader ──────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    return ort.InferenceSession(MODEL_PATH)

# ── Core functions ────────────────────────────────────────────────────────────
def pred_tone(path, sess):
    img = np.array(Image.open(path).resize((224, 224))).astype(np.float32)
    img = np.expand_dims(img, axis=0)
    probs = sess.run(None, {INPUT_NAME: img})[0][0]
    return cls_names[int(np.argmax(probs))], probs

def tone_value(probs):
    weights = {"dark": 0.0, "light": 0.5, "fair": 1.0}
    return sum(float(probs[i]) * weights[cls_names[i]] for i in range(len(cls_names)))

def extract_lab_and_mask(img_path):
    img_bgr = cv2.imread(img_path)
    if img_bgr is None:
        return None, None
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    h, w    = img_rgb.shape[:2]
    with mp.solutions.face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, refine_landmarks=True) as fm:
        res = fm.process(img_rgb)
        if not res.multi_face_landmarks:
            return None, None
        lms = res.multi_face_landmarks[0].landmark
        def roi_mask(ids):
            pts = np.array([[int(lms[i].x * w), int(lms[i].y * h)] for i in ids], dtype=np.int32)
            m   = np.zeros((h, w), dtype=np.uint8)
            cv2.fillPoly(m, [pts], 255)
            return m
        mask     = roi_mask(L_CHK) | roi_mask(R_CHK) | roi_mask(FORH)
        skin_rgb = img_rgb[mask == 255]
        if len(skin_rgb) < 500:
            return None, None
        skin_lab = cv2.cvtColor(skin_rgb.reshape(-1,1,3).astype(np.uint8), cv2.COLOR_RGB2LAB).reshape(-1,3)
        overlay  = img_rgb.copy()
        overlay[mask == 255] = (overlay[mask == 255] * 0.5 + np.array([255,160,80]) * 0.5).astype(np.uint8)
        return skin_lab.mean(axis=0), overlay

def get_undertone(lab_mean):
    _, A, B = lab_mean
    if B - A >= 8: return "warm"
    if A - B >= 8: return "cool"
    return "neutral"

def recommend(tv, undertone=None, top_k=5):
    wd = max(0.0, 1.0 - 2.0 * tv)
    wm = 1.0 - abs(2.0 * tv - 1.0)
    wf = max(0.0, 2.0 * tv - 1.0)
    scores = {}
    for c in POOLS["dark"]: scores[c] = scores.get(c, 0) + wd
    for c in POOLS["mid"]:  scores[c] = scores.get(c, 0) + wm
    for c in POOLS["fair"]: scores[c] = scores.get(c, 0) + wf
    for c in UNDERTONE_BONUS.get(undertone, []):
        scores[c] = scores.get(c, 0) + 0.20
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]

# ── UI ────────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <h1>✦ Skin Tone Analyzer</h1>
    <p>Upload a photo · Get your undertone & personalized color palette</p>
</div>
""", unsafe_allow_html=True)

try:
    sess = load_model()
except Exception:
    st.error("Failed to load model:")
    st.code(traceback.format_exc())
    st.stop()

uploaded = st.file_uploader("", type=["jpg","jpeg","png"], label_visibility="collapsed")

if uploaded:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
        tmp.write(uploaded.read())
        tmp_path = tmp.name

    try:
        with st.spinner("Analyzing your skin tone..."):
            tone, probs  = pred_tone(tmp_path, sess)
            tv           = tone_value(probs)
            lab, overlay = extract_lab_and_mask(tmp_path)
            undertone    = get_undertone(lab) if lab is not None else None
            ranked       = recommend(tv, undertone)

        col1, col2 = st.columns([1,1], gap="medium")
        with col1:
            st.markdown('<p class="section-title">Your Photo</p>', unsafe_allow_html=True)
            st.image(Image.open(tmp_path), use_container_width=True, caption="Original")
        with col2:
            st.markdown('<p class="section-title">Face Region Detected</p>', unsafe_allow_html=True)
            if overlay is not None:
                st.image(overlay, use_container_width=True, caption="Skin mask overlay")
            else:
                st.warning("No face detected.")

        st.markdown('<hr class="divider">', unsafe_allow_html=True)

        ut_class = undertone if undertone else "neutral"
        ut_emoji = {"warm":"🌞","cool":"❄️","neutral":"🍂"}.get(undertone,"")
        st.markdown(f"""
        <div class="card">
            <p class="section-title">Skin Tone Result</p>
            <span class="tone-badge">{tone.upper()}</span>
            <span class="undertone-badge {ut_class}">{ut_emoji} {undertone.capitalize() if undertone else "Unknown"} Undertone</span>
            <p style="font-size:0.8rem;color:#9a7a64;margin-top:0.5rem;">Tone value: {tv:.3f} &nbsp;|&nbsp; Scale: 0 (dark) → 1 (fair)</p>
            <hr class="divider">
            <p class="section-title" style="font-size:0.95rem;">Class Probabilities</p>
        """, unsafe_allow_html=True)

        for i, name in enumerate(cls_names):
            pct = float(probs[i]) * 100
            st.markdown(f"""
            <div class="prob-bar-wrap">
                <div class="prob-label"><span>{name.capitalize()}</span><span>{pct:.1f}%</span></div>
                <div class="prob-bar-bg"><div class="prob-bar-fill" style="width:{pct}%"></div></div>
            </div>""", unsafe_allow_html=True)

        st.markdown("</div>", unsafe_allow_html=True)
        st.markdown('<hr class="divider">', unsafe_allow_html=True)
        st.markdown('<p class="section-title">🎨 Recommended Dress Colors</p>', unsafe_allow_html=True)

        swatch_html = '<div class="card"><div class="swatch-row">'
        for color, score in ranked:
            hex_color = COLOR_MAP.get(color, "#cccccc")
            border    = "1px solid #ccc" if hex_color in ("#FFFFFF","#FFFDD0") else f"2px solid {hex_color}"
            swatch_html += f"""
            <div class="swatch">
                <div class="swatch-circle" style="background:{hex_color};border:{border};"></div>
                <span class="swatch-name">{color}</span>
                <span class="swatch-score">{score:.2f}</span>
            </div>"""
        swatch_html += "</div></div>"
        st.markdown(swatch_html, unsafe_allow_html=True)

    except Exception:
        st.error("Something went wrong:")
        st.code(traceback.format_exc())
    finally:
        os.unlink(tmp_path)

else:
    st.markdown("""
    <div class="card" style="text-align:center;padding:3rem 1rem;">
        <p style="font-size:2rem;margin-bottom:0.5rem;">📸</p>
        <p style="color:#7a5c44;font-size:1rem;">Upload a well-lit, front-facing photo to get started</p>
        <p style="color:#b09080;font-size:0.85rem;margin-top:0.4rem;">Works best with natural lighting · No sunglasses · Clear face visible</p>
    </div>
    """, unsafe_allow_html=True)