import streamlit as st
import joblib
import numpy as np
import plotly.graph_objects as go

st.set_page_config(
    page_title="Diabetes Risk Predictor",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Great+Vibes&family=Segoe+UI:wght@400;600;800&display=swap');

    .stApp {
        background: linear-gradient(135deg, #0f2027 0%, #203a43 50%, #2c5364 100%);
    }
    h1 {
        font-family: 'Segoe UI', sans-serif !important;
        background: linear-gradient(90deg, #00c9ff 0%, #92fe9d 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800 !important;
        font-size: 2.6rem !important;
    }
    h3 {
        color: #92fe9d !important;
        border-bottom: 1px solid rgba(146, 254, 157, 0.3);
        padding-bottom: 6px;
    }
    p, label, span, div {
        color: #e0f2f7 !important;
        font-family: 'Segoe UI', sans-serif;
    }
    .stSlider [role="slider"] {
        background-color: #00c9ff !important;
        box-shadow: 0 0 15px rgba(0, 201, 255, 0.8) !important;
        border: 2px solid #ffffff !important;
    }
    .stSlider [data-baseweb="slider"] > div > div > div > div {
        background: linear-gradient(90deg, #00c9ff, #92fe9d) !important;
    }
    [data-testid="stMetric"] {
        background: rgba(0, 201, 255, 0.08);
        border: 1px solid rgba(0, 201, 255, 0.3);
        border-radius: 14px;
        padding: 16px;
    }
    [data-testid="stMetricValue"] {
        color: #00c9ff !important;
        font-size: 2rem !important;
        font-weight: 800 !important;
    }
    [data-testid="stMetricLabel"] {
        color: #92fe9d !important;
        font-weight: 600 !important;
    }
    .stButton > button {
        background: linear-gradient(90deg, #00c9ff 0%, #92fe9d 100%);
        color: #0f2027 !important;
        border: none;
        border-radius: 12px;
        padding: 0.75em 2em;
        font-weight: 700;
        font-size: 1rem;
        transition: all 0.3s ease;
        width: 100%;
    }
    .stButton > button:hover {
        transform: translateY(-3px);
        box-shadow: 0 12px 25px rgba(0, 201, 255, 0.5);
    }
    div[data-testid="stAlert"] {
        border-radius: 12px !important;
        border-left: 5px solid !important;
    }
    .stSelectbox > div > div {
        background: rgba(0, 201, 255, 0.08) !important;
        border: 1px solid rgba(0, 201, 255, 0.3) !important;
        border-radius: 10px;
    }
    .signature {
        font-family: 'Great Vibes', cursive !important;
        font-size: 2rem !important;
        text-align: right;
        background: linear-gradient(90deg, #00c9ff 0%, #92fe9d 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-top: 20px;
    }
</style>
""", unsafe_allow_html=True)

reg_model = joblib.load("diabetes_reg_model.joblib")
clf_model = joblib.load("diabetes_clf_model.joblib")

REAL_STATS = {
    "age": (48.518, 13.109), "bmi": (26.376, 4.418),
    "blood_pressure": (94.647, 13.831),
    "total_cholesterol": (189.140, 34.809),
    "ldl": (115.445, 30.413), "hdl": (49.788, 12.934),
    "chol_hdl_ratio": (4.070, 1.291),
    "triglycerides": (4.477, 1.281),
    "blood_sugar": (91.445, 11.547),
}
SQRT_N = 21.024
SEX_SCALED = {"Female": -0.0446, "Male": 0.0507}

def to_model_scale(value, feature):
    mean, std = REAL_STATS[feature]
    return (value - mean) / (std * SQRT_N)

st.title("🩺 Diabetes Progression & Risk Predictor")
st.write("Enter the patient's measurements below. The model will estimate "
         "the diabetes progression score and classify the risk level.")
st.caption("⚠️ Educational demo — not a clinical tool.")
st.markdown("---")

col1, col2 = st.columns(2)
with col1:
    st.markdown("### 👤 Patient Info")
    age  = st.slider("Age (years)",               19, 80, 48, 1)
    sex  = st.selectbox("Sex",                    ["Female", "Male"])
    bmi  = st.slider("BMI (kg/m²)",               15.0, 50.0, 26.4, 0.1)
    bp   = st.slider("Blood Pressure (mm Hg)",    60, 130, 95, 1)
    chol = st.slider("Total Cholesterol (mg/dL)", 100, 320, 189, 1)

with col2:
    st.markdown("### 🧪 Lab Results")
    ldl   = st.slider("LDL Cholesterol (mg/dL)",   50, 220, 115, 1)
    hdl   = st.slider("HDL Cholesterol (mg/dL)",   20, 110, 50, 1)
    ratio = st.slider("Chol/HDL Ratio",            2.0, 8.0, 4.1, 0.1)
    trig  = st.slider("Triglycerides (log scale)", 3.0, 6.5, 4.5, 0.1)
    sugar = st.slider("Blood Sugar (mg/dL)",       60, 140, 91, 1)

st.markdown("---")

if st.button("🔮 Predict"):
    scaled = np.array([[
        to_model_scale(age, "age"), SEX_SCALED[sex],
        to_model_scale(bmi, "bmi"),
        to_model_scale(bp, "blood_pressure"),
        to_model_scale(chol, "total_cholesterol"),
        to_model_scale(ldl, "ldl"), to_model_scale(hdl, "hdl"),
        to_model_scale(ratio, "chol_hdl_ratio"),
        to_model_scale(trig, "triglycerides"),
        to_model_scale(sugar, "blood_sugar"),
    ]])

    score = reg_model.predict(scaled)[0]
    risk  = clf_model.predict(scaled)[0]
    proba = clf_model.predict_proba(scaled)[0]

    risk_labels = ["Low Risk", "Moderate Risk", "High Risk"]
    risk_colors = ["#92fe9d", "#ffd93d", "#ff6b6b"]
    risk_icons  = ["✅", "⚠️", "🚨"]

    current_label = risk_labels[risk]
    current_color = risk_colors[risk]
    current_conf  = proba[risk] * 100
    current_icon  = risk_icons[risk]

    st.markdown("### 📊 Results")

    # ---------- GAUGE (now with Diabetes Status labels) ----------
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=score,
        number={'font': {'color': '#00c9ff', 'size': 44}},
        title={'text': "Progression Score",
               'font': {'color': '#92fe9d', 'size': 20}},
        gauge={
            'axis': {'range': [0, 350], 'tickcolor': '#e0f2f7',
                     'tickfont': {'color': '#e0f2f7'}},
            'bar': {'color': current_color, 'thickness': 0.25},
            'bgcolor': 'rgba(0,0,0,0)',
            'borderwidth': 2, 'bordercolor': '#00c9ff',
            'steps': [
                {'range': [0, 115],   'color': 'rgba(146, 254, 157, 0.35)'},
                {'range': [115, 170], 'color': 'rgba(255, 217, 61, 0.35)'},
                {'range': [170, 350], 'color': 'rgba(255, 107, 107, 0.35)'},
            ],
            'threshold': {
                'line': {'color': "white", 'width': 3},
                'thickness': 0.8, 'value': score
            }
        }
    ))
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
        font={'color': '#e0f2f7'}, height=320,
        margin=dict(l=20, r=20, t=60, b=20)
    )
    st.plotly_chart(fig, use_container_width=True)

    # ---------- NEW LEGEND: Normal / Prediabetic / Diabetic ----------
    st.markdown(
        """
        <div style="display:flex; justify-content:space-around;
                    margin-top:-15px; margin-bottom:20px;
                    font-size:0.9rem; font-weight:600;">
            <span style="color:#92fe9d;">● Normal (0-115)</span>
            <span style="color:#ffd93d;">● Prediabetic (115-170)</span>
            <span style="color:#ff6b6b;">● Diabetic (170+)</span>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    # ---------- RISK CARD (unchanged) ----------
    st.markdown(
        f"""
        <div style="
            background: linear-gradient(135deg, {current_color}22, {current_color}11);
            border: 2px solid {current_color};
            border-radius: 20px; padding: 30px; text-align: center;
            margin: 20px 0; box-shadow: 0 10px 30px {current_color}33;
        ">
            <div style="font-size: 3rem; margin-bottom: 10px;">{current_icon}</div>
            <div style="font-size: 2rem; font-weight: 800; color: {current_color};
                        letter-spacing: 1px; margin-bottom: 8px;">
                {current_label.upper()}
            </div>
            <div style="font-size: 1.4rem; color: #e0f2f7; font-weight: 600;">
                Confidence: {current_conf:.1f}%
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown(f"#### Confidence — {current_label}")
    st.progress(float(proba[risk]), text=f"{current_conf:.1f}%")

st.markdown("---")
st.markdown('<div class="signature">Created by — Manan Varshney</div>',
            unsafe_allow_html=True)
st.caption("Model: Linear Regression + Random Forest · Dataset: sklearn load_diabetes()")
