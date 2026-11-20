import json
import math
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

BASE = Path(__file__).parent

st.set_page_config(
    page_title="Phone addiction check",
    page_icon="📱",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ----------------------------------------------------------------------------
# Styling (works with light or dark theme; .streamlit/config.toml sets the look)
# ----------------------------------------------------------------------------
st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Manrope:wght@400;500;600;700;800&display=swap');

.stApp, .stApp p, .stApp label, .stApp h1, .stApp h2, .stApp h3, .stApp h4,
.stApp button, .stApp input, .stApp div[data-testid="stMarkdownContainer"] {
    font-family: 'Manrope', sans-serif;
}
.block-container { max-width: 1080px; padding-top: 2.5rem; padding-bottom: 5rem; }
[data-testid="stVerticalBlock"] { gap: 1.5rem; }
[data-testid="stSidebar"], [data-testid="collapsedControl"], header[data-testid="stHeader"] { display: none; }

/* ---------- hero ---------- */
.hero { display: flex; align-items: center; justify-content: space-between; gap: 2rem;
        padding: 2.6rem 3rem; border-radius: 24px; border: 1px solid rgba(128,128,128,0.22);
        background: linear-gradient(120deg, rgba(61,90,254,0.16) 0%, rgba(42,157,143,0.10) 55%, rgba(233,196,106,0.12) 100%); }
.hero-title { font-size: 3rem; font-weight: 800; letter-spacing: -0.03em; line-height: 1.05; margin: 0; }
.hero-sub { font-size: 1.12rem; line-height: 1.55; opacity: 0.75; margin: 1rem 0 0 0; max-width: 31rem; }
.hero-art { flex: 0 0 auto; color: inherit; }
@media (max-width: 760px) { .hero { padding: 1.6rem; } .hero-title { font-size: 2.1rem; } .hero-art { display: none; } }

/* ---------- section cards ---------- */
div[class*="st-key-sec_"] { background: rgba(255,255,255,0.6); border: 1px solid rgba(128,128,128,0.22);
                            border-radius: 20px; padding: 1.2rem 1.4rem 1.6rem 1.4rem; }
.sec-head { display: flex; align-items: center; gap: 0.9rem; margin-bottom: 0.4rem; }
.sec-icon { width: 46px; height: 46px; border-radius: 14px; display: flex; align-items: center; justify-content: center;
            font-size: 1.4rem; background: rgba(61,90,254,0.12); }
.sec-title { font-size: 1.25rem; font-weight: 700; margin: 0; line-height: 1.2; }
.sec-sub { font-size: 0.92rem; opacity: 0.62; margin: 0.1rem 0 0 0; }
.stApp label p { font-weight: 600; }

/* ---------- button ---------- */
.stApp button[kind="primaryFormSubmit"], .stApp button[kind="primary"] {
    height: 3.5rem; border-radius: 14px; font-size: 1.1rem; font-weight: 700; letter-spacing: 0.01em; }

/* ---------- result ---------- */
.result { display: grid; grid-template-columns: minmax(260px, 360px) 1fr; gap: 2.6rem; align-items: center;
          padding: 2.2rem 2.6rem; border-radius: 24px; border: 1px solid rgba(128,128,128,0.25);
          background: linear-gradient(120deg, rgba(61,90,254,0.10), rgba(128,128,128,0.04)); }
@media (max-width: 760px) { .result { grid-template-columns: 1fr; padding: 1.5rem; gap: 1.2rem; } }
.result-level { font-size: 2rem; font-weight: 800; letter-spacing: -0.02em; margin: 0; line-height: 1.1; }
.result-advice { font-size: 1.05rem; line-height: 1.55; opacity: 0.8; margin: 0.7rem 0 0 0; }
.tips { list-style: none; padding: 0; margin: 1.3rem 0 0 0; }
.tips li { padding: 0.6rem 0 0.6rem 1.7rem; position: relative; font-size: 0.98rem; line-height: 1.45;
           border-top: 1px solid rgba(128,128,128,0.2); }
.tips li::before { content: ""; position: absolute; left: 0; top: 1.05rem; width: 10px; height: 10px;
                   border-radius: 50%; background: var(--dot, #3d5afe); }
.note { font-size: 0.84rem; opacity: 0.55; margin: 1.1rem 0 0 0; }
</style>
""",
    unsafe_allow_html=True,
)


# ----------------------------------------------------------------------------
# Load model + schema (created by the training script)
# ----------------------------------------------------------------------------
@st.cache_data
def load_schema():
    with open(BASE / "models" / "feature_schema.json") as f:
        return json.load(f)


@st.cache_resource
def load_model():
    return joblib.load(BASE / "models" / "best_model.pkl")


try:
    schema = load_schema()
except FileNotFoundError:
    st.error("models/feature_schema.json not found. Run `dvc repro` first.")
    st.stop()

try:
    model = load_model()
except FileNotFoundError:
    st.error("models/best_model.pkl not found. Run `dvc repro` first.")
    st.stop()
except Exception as e:
    st.error(f"Could not load the model: {e}")
    st.stop()

# ----------------------------------------------------------------------------
# Fields: sections, icons and friendly labels
# ----------------------------------------------------------------------------
SECTIONS = [
    ("sec_about", "👤", "About you", "A few basic details.",
     ["Age", "Gender"]),
    ("sec_phone", "📱", "Phone habits", "How often and why you use your phone.",
     ["Daily_Usage_Hours", "Weekend_Usage_Hours", "Phone_Checks_Per_Day",
      "Apps_Used_Daily", "Screen_Time_Before_Bed", "Phone_Usage_Purpose"]),
    ("sec_time", "⏱️", "Screen time breakdown", "How your daily hours are split.",
     ["Time_on_Social_Media", "Time_on_Gaming", "Time_on_Education"]),
    ("sec_health", "🌙", "Health and wellbeing", "Sleep, mood and daily life.",
     ["Sleep_Hours", "Exercise_Hours", "Anxiety_Level", "Depression_Level",
      "Self_Esteem", "Intellectual_Performance", "Social_Interactions",
      "Family_Communication"]),
]

LABELS = {
    "Daily_Usage_Hours": "Phone use per day (hours)",
    "Weekend_Usage_Hours": "Phone use on weekends (hours)",
    "Phone_Checks_Per_Day": "Times you check your phone per day",
    "Apps_Used_Daily": "Apps used per day",
    "Screen_Time_Before_Bed": "Screen time before bed (hours)",
    "Phone_Usage_Purpose": "Main reason for phone use",
    "Time_on_Social_Media": "Time on social media (hours)",
    "Time_on_Gaming": "Time on gaming (hours)",
    "Time_on_Education": "Time on learning (hours)",
    "Sleep_Hours": "Sleep per night (hours)",
    "Exercise_Hours": "Exercise per day (hours)",
    "Intellectual_Performance": "Intellectual performance",
}


def label_for(name):
    return LABELS.get(name, name.replace("_", " "))


def render_field(name, container):
    """Draw one input widget and return its value."""
    with container:
        if name in schema["categorical"]:
            return st.selectbox(label_for(name), schema["categorical"][name], key=name)

        info = schema["numeric"][name]
        lo, hi, med = info["min"], info["max"], info["median"]
        if info["is_int"]:
            lo, hi, med = int(lo), int(hi), int(round(med))
            if lo == hi:
                return st.number_input(label_for(name), value=lo, key=name)
            return st.slider(label_for(name), lo, hi, med, step=1, key=name)
        lo, hi, med = round(float(lo), 1), round(float(hi), 1), round(float(med), 1)
        if lo >= hi:
            return st.number_input(label_for(name), value=float(lo), key=name)
        return st.slider(label_for(name), lo, hi, min(max(med, lo), hi), step=0.1,
                         format="%.1f", key=name)


def bordered(key):
    try:
        return st.container(border=True, key=key)
    except TypeError:                      # older Streamlit without container keys
        return st.container(border=True)


# ----------------------------------------------------------------------------
# Result helpers
# ----------------------------------------------------------------------------
LEVELS = [
    (0.25, "Low", "#2a9d8f", "Your habits look balanced. Keep it up.",
     ["Keep screens out of the bedroom at night.",
      "Stay active away from your phone.",
      "Check in on your habits every few months."]),
    (0.50, "Moderate", "#c9a227", "Some signs of heavy use. Late-night screen time and daily hours matter most.",
     ["Set a daily limit with your phone's screen-time tool.",
      "Stop using your phone 30 minutes before bed.",
      "Turn off notifications from apps you don't need."]),
    (0.75, "High", "#e08a3c", "Your habits point to heavy use. Less screen time before bed and fewer daily hours would help most.",
     ["Cut social media and gaming time first.",
      "Keep your phone out of reach while you sleep.",
      "Swap one daily scroll session for exercise or time with a friend."]),
    (1.01, "Very high", "#e76f51", "Your habits point to very heavy use. Small limits now can make a big difference.",
     ["Set firm daily limits and turn off non-essential notifications.",
      "Keep the phone out of the bedroom.",
      "If it affects your sleep, mood, school or work, talk to a doctor or counsellor."]),
]


def level_for(pct):
    for limit, name, color, advice, tips in LEVELS:
        if pct < limit:
            return name, color, advice, tips
    return LEVELS[-1][1:]


def gauge_svg(pred, hi, pct):
    cx, cy, r = 150, 150, 120
    a = math.pi * (1 - pct)
    px, py = cx + r * math.cos(a), cy - r * math.sin(a)
    arc = "M 30 150 A 120 120 0 0 1 270 150"
    return (
        '<svg viewBox="0 0 300 185" width="100%" role="img" aria-label="Addiction level gauge">'
        '<defs><linearGradient id="ramp" gradientUnits="userSpaceOnUse" x1="30" y1="0" x2="270" y2="0">'
        '<stop offset="0" stop-color="#2a9d8f"/><stop offset="0.4" stop-color="#e9c46a"/>'
        '<stop offset="0.7" stop-color="#f4a261"/><stop offset="1" stop-color="#e76f51"/>'
        '</linearGradient></defs>'
        f'<path d="{arc}" fill="none" stroke="url(#ramp)" stroke-width="22" stroke-linecap="round" opacity="0.22"/>'
        f'<path d="{arc}" pathLength="100" fill="none" stroke="url(#ramp)" stroke-width="22" '
        f'stroke-linecap="round" stroke-dasharray="{max(pct * 100, 0.5):.1f} 100"/>'
        f'<circle cx="{px:.1f}" cy="{py:.1f}" r="15" fill="#ffffff" stroke="#0f172a" stroke-width="5"/>'
        f'<text x="150" y="138" text-anchor="middle" font-size="58" font-weight="800" fill="currentColor">{pred:.1f}</text>'
        f'<text x="150" y="168" text-anchor="middle" font-size="15" fill="currentColor" opacity="0.6">out of {hi:g}</text>'
        '</svg>'
    )


PHONE_ART = (
    '<svg class="hero-art" width="132" height="214" viewBox="0 0 132 214" fill="none" role="img" aria-label="Phone">'
    '<rect x="4" y="4" width="124" height="206" rx="26" stroke="currentColor" stroke-width="5" opacity="0.85"/>'
    '<rect x="46" y="14" width="40" height="7" rx="3.5" fill="currentColor" opacity="0.35"/>'
    '<rect x="26" y="120" width="16" height="56" rx="6" fill="#2a9d8f"/>'
    '<rect x="50" y="92" width="16" height="84" rx="6" fill="#e9c46a"/>'
    '<rect x="74" y="64" width="16" height="112" rx="6" fill="#f4a261"/>'
    '<rect x="98" y="40" width="16" height="136" rx="6" fill="#e76f51"/>'
    '<rect x="42" y="190" width="48" height="5" rx="2.5" fill="currentColor" opacity="0.35"/>'
    '</svg>'
)

# ----------------------------------------------------------------------------
# Page
# ----------------------------------------------------------------------------
st.markdown(
    f"""
<div class="hero">
  <div>
    <p class="hero-title">Phone addiction check</p>
    <p class="hero-sub">Answer a few questions about your daily habits and get an estimate of your addiction level. It takes about a minute.</p>
  </div>
  {PHONE_ART}
</div>
""",
    unsafe_allow_html=True,
)

inputs = {}

with st.form("predict_form", border=False):
    all_fields = list(schema["numeric"].keys()) + list(schema["categorical"].keys())

    sections = [(k, i, t, s, [f for f in fields if f in all_fields]) for k, i, t, s, fields in SECTIONS]
    used = {f for *_, fields in sections for f in fields}
    leftover = [f for f in all_fields if f not in used]
    if leftover:
        sections.append(("sec_other", "➕", "Other details", "Everything else the model uses.", leftover))

    for key, icon, title, sub, fields in sections:
        if not fields:
            continue
        with bordered(key):
            st.markdown(
                f'<div class="sec-head"><div class="sec-icon">{icon}</div>'
                f'<div><p class="sec-title">{title}</p><p class="sec-sub">{sub}</p></div></div>',
                unsafe_allow_html=True,
            )
            cols = st.columns(3, gap="large")
            for i, name in enumerate(fields):
                inputs[name] = render_field(name, cols[i % 3])

    submitted = st.form_submit_button("Predict addiction level", type="primary",
                                      use_container_width=True)

if submitted:
    row = dict(schema["defaults"])      # columns the model ignores
    row.update(inputs)

    # engineered features (formulas come from params.yaml via the schema)
    eps = float(schema.get("epsilon", 1e-5))
    for new_col, (num, den) in schema.get("ratio_features", {}).items():
        row[new_col] = float(row[num]) / (float(row[den]) + eps)

    X_new = pd.DataFrame([row])[schema["all_columns"]]
    pred = float(model.predict(X_new)[0])
    lo, hi = schema["target_min"], schema["target_max"]
    pred = min(max(pred, lo), hi)
    pct = (pred - lo) / (hi - lo) if hi > lo else 0.0
    level, color, advice, tips = level_for(pct)

    tips_html = "".join(f"<li>{t}</li>" for t in tips)
    st.markdown(
        f"""
<div class="result" id="result" style="--dot:{color}">
  <div>{gauge_svg(pred, hi, pct)}</div>
  <div>
    <p class="result-level" style="color:{color}">{level} addiction level</p>
    <p class="result-advice">{advice}</p>
    <ul class="tips">{tips_html}</ul>
    <p class="note">This is a model estimate based on your answers, not a medical diagnosis.</p>
  </div>
</div>
""",
        unsafe_allow_html=True,
    )
    # scroll the result into view after it is drawn (optional; skipped if unavailable)
    try:
        import streamlit.components.v1 as components
        components.html(
            "<script>const el = window.parent.document.getElementById('result');"
            "if (el) el.scrollIntoView({behavior: 'smooth', block: 'start'});</script>",
            height=0,
        )
    except Exception:
        pass
