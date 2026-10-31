"""
╔══════════════════════════════════════════════════════════╗
║          SpendScope — Customer Spending Predictor        ║
║          Premium Streamlit UI  |  ML-Powered             ║
╚══════════════════════════════════════════════════════════╝
"""

import pickle
import os
import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime

# ── Page Configuration ─────────────────────────────────────────────────────────
st.set_page_config(
    page_title="SpendScope — Customer Spending Predictor",
    page_icon="assets/logo.png",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Premium Custom CSS ─────────────────────────────────────────────────────────
st.markdown("""
<style>
/* ── Google Fonts ── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&family=JetBrains+Mono:wght@400;600&display=swap');

/* ── Global Reset ── */
*, *::before, *::after { box-sizing: border-box; }

html, body, [data-testid="stAppViewContainer"] {
    background: #070B14 !important;
    font-family: 'Inter', sans-serif;
    color: #E2E8F0;
}

/* ── Hide Streamlit Branding ── */
#MainMenu, footer, header { visibility: hidden; }
[data-testid="stDecoration"] { display: none; }

/* ── Main Content Area ── */
[data-testid="stAppViewContainer"] > .main {
    background: #070B14 !important;
}
[data-testid="block-container"] {
    padding: 1.5rem 2rem 4rem !important;
    max-width: 1400px;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0D1321 0%, #0F172A 100%) !important;
    border-right: 1px solid rgba(139, 92, 246, 0.2) !important;
    padding-top: 1rem;
}
[data-testid="stSidebar"] * { color: #CBD5E1 !important; }
[data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3 {
    color: #A78BFA !important;
}
[data-testid="stSidebarContent"] { padding: 1.5rem 1rem !important; }

/* ── Hero Banner ── */
.hero-banner {
    background: linear-gradient(135deg, #0F172A 0%, #1E1035 40%, #0C1A3A 100%);
    border: 1px solid rgba(139, 92, 246, 0.3);
    border-radius: 20px;
    padding: 2.5rem 3rem;
    margin-bottom: 2rem;
    position: relative;
    overflow: hidden;
}
.hero-banner::before {
    content: '';
    position: absolute;
    top: -50%;
    right: -10%;
    width: 500px;
    height: 500px;
    background: radial-gradient(circle, rgba(139,92,246,0.15) 0%, transparent 70%);
    pointer-events: none;
}
.hero-banner::after {
    content: '';
    position: absolute;
    bottom: -30%;
    left: 20%;
    width: 300px;
    height: 300px;
    background: radial-gradient(circle, rgba(6,182,212,0.1) 0%, transparent 70%);
    pointer-events: none;
}
.hero-title {
    font-size: 2.8rem;
    font-weight: 800;
    background: linear-gradient(135deg, #A78BFA 0%, #06B6D4 50%, #818CF8 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin: 0 0 0.5rem 0;
    line-height: 1.1;
}
.hero-subtitle {
    font-size: 1.05rem;
    color: #94A3B8;
    font-weight: 400;
    margin: 0;
    letter-spacing: 0.01em;
}
.hero-badge {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    background: rgba(139,92,246,0.15);
    border: 1px solid rgba(139,92,246,0.3);
    border-radius: 100px;
    padding: 4px 14px;
    font-size: 0.75rem;
    font-weight: 600;
    color: #A78BFA;
    letter-spacing: 0.05em;
    text-transform: uppercase;
    margin-bottom: 1rem;
}

/* ── Metric Cards ── */
.metric-card {
    background: linear-gradient(135deg, #0F172A 0%, #131C30 100%);
    border: 1px solid rgba(139,92,246,0.2);
    border-radius: 16px;
    padding: 1.4rem 1.5rem;
    transition: all 0.3s ease;
    position: relative;
    overflow: hidden;
    height: 100%;
}
.metric-card:hover {
    border-color: rgba(139,92,246,0.5);
    transform: translateY(-2px);
    box-shadow: 0 8px 30px rgba(139,92,246,0.15);
}
.metric-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 2px;
    background: linear-gradient(90deg, #8B5CF6, #06B6D4);
}
.metric-icon { font-size: 1.8rem; margin-bottom: 0.6rem; }
.metric-label {
    font-size: 0.75rem;
    font-weight: 600;
    color: #64748B;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-bottom: 0.3rem;
}
.metric-value {
    font-size: 1.7rem;
    font-weight: 800;
    color: #F1F5F9;
    line-height: 1;
    font-family: 'JetBrains Mono', monospace;
}
.metric-delta {
    font-size: 0.75rem;
    color: #10B981;
    margin-top: 0.3rem;
    font-weight: 500;
}

/* ── Section Headers ── */
.section-header {
    display: flex;
    align-items: center;
    gap: 10px;
    margin: 2.5rem 0 1.2rem 0;
    padding-bottom: 0.8rem;
    border-bottom: 1px solid rgba(139,92,246,0.15);
}
.section-title {
    font-size: 1.2rem;
    font-weight: 700;
    color: #E2E8F0;
    margin: 0;
}
.section-pill {
    background: rgba(139,92,246,0.15);
    border: 1px solid rgba(139,92,246,0.25);
    border-radius: 100px;
    padding: 2px 10px;
    font-size: 0.68rem;
    font-weight: 600;
    color: #A78BFA;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}

/* ── Input Card ── */
.input-card {
    background: linear-gradient(135deg, #0F172A 0%, #131C30 100%);
    border: 1px solid rgba(100,116,139,0.2);
    border-radius: 16px;
    padding: 1.8rem;
    margin-bottom: 1rem;
}
.input-card-title {
    font-size: 0.8rem;
    font-weight: 700;
    color: #64748B;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-bottom: 1rem;
    display: flex;
    align-items: center;
    gap: 6px;
}

/* ── Streamlit Widgets Override ── */
[data-testid="stNumberInput"] input,
[data-testid="stSelectbox"] select,
.stSelectbox > div > div {
    background: #1E293B !important;
    border: 1px solid rgba(100,116,139,0.3) !important;
    border-radius: 10px !important;
    color: #E2E8F0 !important;
    font-family: 'Inter', sans-serif !important;
}
[data-testid="stNumberInput"] input:focus,
.stSelectbox > div > div:focus {
    border-color: rgba(139,92,246,0.5) !important;
    box-shadow: 0 0 0 3px rgba(139,92,246,0.1) !important;
}

label {
    font-size: 0.82rem !important;
    font-weight: 600 !important;
    color: #94A3B8 !important;
    letter-spacing: 0.02em !important;
}

/* ── Predict Button ── */
[data-testid="stButton"] > button {
    width: 100%;
    background: linear-gradient(135deg, #7C3AED 0%, #2563EB 100%) !important;
    border: none !important;
    color: #fff !important;
    font-family: 'Inter', sans-serif !important;
    font-weight: 700 !important;
    font-size: 1rem !important;
    letter-spacing: 0.03em;
    padding: 0.85rem 2rem !important;
    border-radius: 12px !important;
    transition: all 0.3s ease !important;
    cursor: pointer;
    box-shadow: 0 4px 20px rgba(124,58,237,0.4) !important;
}
[data-testid="stButton"] > button:hover {
    background: linear-gradient(135deg, #6D28D9 0%, #1D4ED8 100%) !important;
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 30px rgba(124,58,237,0.5) !important;
}
[data-testid="stButton"] > button:active {
    transform: translateY(0) !important;
}

/* ── Prediction Result Card ── */
.result-card {
    background: linear-gradient(135deg, #1E1035 0%, #0C1A3A 100%);
    border: 1px solid rgba(139,92,246,0.4);
    border-radius: 20px;
    padding: 2rem 2.5rem;
    text-align: center;
    position: relative;
    overflow: hidden;
    animation: resultPop 0.5s cubic-bezier(0.175, 0.885, 0.32, 1.275);
}
@keyframes resultPop {
    from { opacity: 0; transform: scale(0.92) translateY(10px); }
    to   { opacity: 1; transform: scale(1) translateY(0); }
}
.result-card::before {
    content: '';
    position: absolute;
    top: 50%; left: 50%;
    transform: translate(-50%, -50%);
    width: 200%;
    height: 200%;
    background: radial-gradient(ellipse, rgba(139,92,246,0.08) 0%, transparent 60%);
    pointer-events: none;
}
.result-label {
    font-size: 0.8rem;
    font-weight: 700;
    color: #A78BFA;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    margin-bottom: 0.5rem;
}
.result-amount {
    font-size: 3rem;
    font-weight: 900;
    background: linear-gradient(135deg, #A78BFA 0%, #06B6D4 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    font-family: 'JetBrains Mono', monospace;
    line-height: 1.1;
}
.result-currency {
    font-size: 1.5rem;
    font-weight: 700;
    color: #A78BFA;
}
.result-desc {
    font-size: 0.85rem;
    color: #64748B;
    margin-top: 0.6rem;
}
.result-tier {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    padding: 6px 16px;
    border-radius: 100px;
    font-size: 0.8rem;
    font-weight: 700;
    margin-top: 1rem;
}

/* ── Info Box ── */
.info-box {
    background: rgba(6,182,212,0.08);
    border: 1px solid rgba(6,182,212,0.2);
    border-left: 3px solid #06B6D4;
    border-radius: 10px;
    padding: 0.9rem 1.2rem;
    font-size: 0.82rem;
    color: #94A3B8;
    line-height: 1.6;
}

/* ── Sidebar Stat Pill ── */
.sidebar-stat {
    background: rgba(139,92,246,0.1);
    border: 1px solid rgba(139,92,246,0.2);
    border-radius: 12px;
    padding: 0.8rem 1rem;
    margin: 0.5rem 0;
    font-size: 0.82rem;
}
.sidebar-stat-label { color: #64748B; font-size: 0.72rem; text-transform: uppercase; letter-spacing: 0.06em; }
.sidebar-stat-value { color: #A78BFA; font-weight: 700; font-size: 1rem; font-family: 'JetBrains Mono', monospace; }

/* ── Plotly charts background ── */
.js-plotly-plot .plotly .bg { fill: transparent !important; }

/* ── Divider ── */
.grad-divider {
    height: 1px;
    background: linear-gradient(90deg, transparent, rgba(139,92,246,0.4), transparent);
    margin: 2rem 0;
    border: none;
}

/* ── Tab customization ── */
[data-testid="stTabs"] [role="tablist"] {
    background: rgba(15,23,42,0.6) !important;
    border-radius: 12px !important;
    padding: 4px !important;
    border: 1px solid rgba(100,116,139,0.15) !important;
}
[data-testid="stTabs"] [role="tab"] {
    font-family: 'Inter', sans-serif !important;
    font-weight: 600 !important;
    font-size: 0.85rem !important;
    color: #64748B !important;
    border-radius: 8px !important;
    transition: all 0.2s !important;
}
[data-testid="stTabs"] [role="tab"][aria-selected="true"] {
    background: linear-gradient(135deg, #7C3AED, #2563EB) !important;
    color: #fff !important;
}

/* ── Progress bars ── */
[data-testid="stProgress"] > div > div {
    background: linear-gradient(90deg, #7C3AED, #06B6D4) !important;
    border-radius: 100px !important;
}
</style>
""", unsafe_allow_html=True)


# ── Load Model & Dataset ────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_model():
    model_path = os.path.join(os.path.dirname(__file__),
                              "best_model", "best_model_prediction.pkl")
    with open(model_path, "rb") as f:
        return pickle.load(f)

@st.cache_data(show_spinner=False)
def load_data():
    return pd.read_csv(os.path.join(os.path.dirname(__file__),
                                    "datasets", "data_cleaned.csv"))

with st.spinner("Loading SpendScope AI..."):
    best_pipeline = load_model()
    df = load_data()


# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.image("assets/logo.png", width=80)
    st.markdown("""
    <h1 style="font-size:1.4rem; font-weight:800; margin:0.5rem 0 0.2rem;
               background:linear-gradient(135deg,#A78BFA,#06B6D4);
               -webkit-background-clip:text; -webkit-text-fill-color:transparent;
               background-clip:text;">SpendScope</h1>
    <p style="font-size:0.75rem; color:#475569; margin:0 0 1.5rem 0; font-weight:500;">
        AI-Powered Spend Intelligence
    </p>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown('<p style="font-size:0.72rem; font-weight:700; color:#475569; text-transform:uppercase; letter-spacing:0.08em;">📊 Dataset Stats</p>', unsafe_allow_html=True)

    st.markdown(f"""
    <div class="sidebar-stat">
        <div class="sidebar-stat-label">Total Records</div>
        <div class="sidebar-stat-value">{len(df):,}</div>
    </div>
    <div class="sidebar-stat">
        <div class="sidebar-stat-label">Avg. Spend</div>
        <div class="sidebar-stat-value">₹{df['Amount_Spent'].mean():,.0f}</div>
    </div>
    <div class="sidebar-stat">
        <div class="sidebar-stat-label">Max Spend</div>
        <div class="sidebar-stat-value">₹{df['Amount_Spent'].max():,.0f}</div>
    </div>
    <div class="sidebar-stat">
        <div class="sidebar-stat-label">Features</div>
        <div class="sidebar-stat-value">8</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown('<p style="font-size:0.72rem; font-weight:700; color:#475569; text-transform:uppercase; letter-spacing:0.08em;">⚡ Model Info</p>', unsafe_allow_html=True)
    st.markdown("""
    <div class="sidebar-stat">
        <div class="sidebar-stat-label">Architecture</div>
        <div class="sidebar-stat-value" style="font-size:0.82rem;">Random Forest</div>
    </div>
    <div class="sidebar-stat">
        <div class="sidebar-stat-label">Pipeline</div>
        <div class="sidebar-stat-value" style="font-size:0.82rem;">sklearn</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown(f"""
    <div style="font-size:0.7rem; color:#334155; text-align:center; padding-top:0.5rem;">
        🔄 Last updated<br>
        <span style="color:#475569; font-weight:600;">{datetime.now().strftime('%d %b %Y')}</span>
    </div>
    """, unsafe_allow_html=True)


# ── Hero Banner ────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero-banner">
    <div class="hero-badge">🤖 ML-Powered Regression</div>
    <h1 class="hero-title">Customer Spend<br>Prediction Engine</h1>
    <p class="hero-subtitle">
        Enter customer profile details below and let our trained AI model estimate
        the expected spending amount with precision.
    </p>
</div>
""", unsafe_allow_html=True)


# ── Dataset-Derived Stats Row ──────────────────────────────────────────────────
avg_spend   = df['Amount_Spent'].mean()
med_spend   = df['Amount_Spent'].median()
total_cust  = len(df)
gold_pct    = round((df['Membership_Type'] == 'Gold').mean() * 100, 1)

st.markdown("""
<div class="section-header">
  <span style="font-size:1.2rem;">📈</span>
  <h2 class="section-title">Dataset Overview</h2>
  <span class="section-pill">Live Stats</span>
</div>
""", unsafe_allow_html=True)

mc1, mc2, mc3, mc4 = st.columns(4)
cards = [
    (mc1, "👥", "Total Customers", f"{total_cust:,}", "ShopSmart dataset"),
    (mc2, "💰", "Avg. Spend",      f"₹{avg_spend:,.0f}", "Across all customers"),
    (mc3, "📊", "Median Spend",    f"₹{med_spend:,.0f}", "Middle 50th percentile"),
    (mc4, "🥇", "Gold Members",    f"{gold_pct}%", "Premium tier"),
]
for col, icon, label, val, desc in cards:
    with col:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-icon">{icon}</div>
            <div class="metric-label">{label}</div>
            <div class="metric-value">{val}</div>
            <div class="metric-delta">{desc}</div>
        </div>
        """, unsafe_allow_html=True)


# ── Tabs: Predict | Insights ───────────────────────────────────────────────────
st.markdown("<br>", unsafe_allow_html=True)
tab1, tab2 = st.tabs(["🎯  Predict Spending", "📊  Data Insights"])


# ══════════════════════════════════════════════════════════════════
#  TAB 1  —  PREDICTION FORM
# ══════════════════════════════════════════════════════════════════
with tab1:
    st.markdown("""
    <div class="section-header">
      <span style="font-size:1.2rem;">📝</span>
      <h2 class="section-title">Customer Profile Input</h2>
      <span class="section-pill">8 Features</span>
    </div>
    """, unsafe_allow_html=True)

    form_col, result_col = st.columns([3, 2], gap="large")

    with form_col:
        # ── Numerical Inputs ──
        st.markdown('<div class="input-card"><div class="input-card-title">🔢 Numerical Features</div>', unsafe_allow_html=True)
        n1, n2 = st.columns(2)
        with n1:
            Age = st.number_input("Age", min_value=18, max_value=100, value=30,
                                  help="Customer age in years")
            Website_Visits = st.number_input("Monthly Website Visits", min_value=0,
                                             max_value=200, value=10,
                                             help="Number of times customer visited the website per month")
        with n2:
            Annual_Income = st.number_input("Annual Income (₹)", min_value=10_000,
                                            max_value=8_000_000, value=600_000, step=10_000,
                                            help="Customer's annual income in INR")
            Previous_Purchases = st.number_input("Previous Purchases", min_value=0,
                                                  max_value=500, value=5,
                                                  help="Total number of past purchases")
        st.markdown('</div>', unsafe_allow_html=True)

        # ── Categorical Inputs ──
        st.markdown('<div class="input-card"><div class="input-card-title">🏷️ Categorical Features</div>', unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        with c1:
            Membership_Type   = st.selectbox("Membership Type",
                                              options=sorted(df['Membership_Type'].unique()),
                                              index=0,
                                              help="Customer's loyalty tier: Bronze → Silver → Gold")
            City_Tier         = st.selectbox("City Tier",
                                              options=sorted(df['City_Tier'].unique()),
                                              index=0,
                                              help="Tier 1=Metro, Tier 2=Large city, Tier 3=Small city")
        with c2:
            Preferred_Category = st.selectbox("Preferred Category",
                                               options=sorted(df['Preferred_Category'].unique()),
                                               index=0,
                                               help="The product category this customer shops most from")
            Discount_Used      = st.selectbox("Discount Used",
                                               options=df['Discount_Used'].unique(),
                                               index=0,
                                               help="Whether the customer used a discount coupon")
        st.markdown('</div>', unsafe_allow_html=True)

        # ── Predict Button ──
        if st.button("⚡ Predict Spending Amount", use_container_width=True):
            st.session_state["predict_clicked"] = True
            new_data = pd.DataFrame({
                "Age":               [Age],
                "Annual_Income":     [Annual_Income],
                "Website_Visits":    [Website_Visits],
                "Previous_Purchases":[Previous_Purchases],
                "Membership_Type":   [Membership_Type],
                "Preferred_Category":[Preferred_Category],
                "City_Tier":         [City_Tier],
                "Discount_Used":     [Discount_Used],
            })
            with st.spinner("Running ML pipeline…"):
                prediction = float(best_pipeline.predict(new_data)[0])
            st.session_state["prediction"] = round(prediction, 2)
            st.session_state["input_snapshot"] = new_data.copy()

    # ── Result Column ──
    with result_col:
        st.markdown("""
        <div class="section-header">
          <span style="font-size:1.2rem;">🎯</span>
          <h2 class="section-title">Prediction Result</h2>
        </div>
        """, unsafe_allow_html=True)

        if "prediction" in st.session_state:
            pred = st.session_state["prediction"]
            # Tier label
            if pred < 5000:
                tier_color, tier_label, tier_icon = "#64748B", "Low Spender",    "🪙"
            elif pred < 15000:
                tier_color, tier_label, tier_icon = "#F59E0B", "Mid Spender",    "💵"
            elif pred < 30000:
                tier_color, tier_label, tier_icon = "#8B5CF6", "High Spender",   "💎"
            else:
                tier_color, tier_label, tier_icon = "#06B6D4", "Premium Spender","🏆"

            st.markdown(f"""
            <div class="result-card">
                <div class="result-label">Estimated Spending</div>
                <div class="result-amount">
                    <span class="result-currency">₹</span>{pred:,.2f}
                </div>
                <div class="result-desc">Based on the provided customer profile</div>
                <div class="result-tier" style="background:rgba({int(tier_color[1:3],16)},{int(tier_color[3:5],16)},{int(tier_color[5:],16)},0.15);
                     border:1px solid {tier_color}40; color:{tier_color}; margin: 1rem auto 0; display:inline-flex;">
                    {tier_icon} {tier_label}
                </div>
            </div>
            """, unsafe_allow_html=True)

            # ── Gauge chart ──
            st.markdown("<br>", unsafe_allow_html=True)
            gauge_max = float(df['Amount_Spent'].max())
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=pred,
                number={"prefix": "₹", "font": {"color": "#A78BFA", "size": 22, "family": "JetBrains Mono"}},
                gauge={
                    "axis": {"range": [0, gauge_max], "tickcolor": "#475569",
                             "tickfont": {"color": "#475569", "size": 9}},
                    "bar":  {"color": "#7C3AED"},
                    "bgcolor": "#1E293B",
                    "bordercolor": "rgba(139,92,246,0.2)",
                    "steps": [
                        {"range": [0,            gauge_max*0.25], "color": "#1E293B"},
                        {"range": [gauge_max*0.25, gauge_max*0.5],  "color": "#1E293B"},
                        {"range": [gauge_max*0.5,  gauge_max*0.75], "color": "#1E293B"},
                        {"range": [gauge_max*0.75, gauge_max],      "color": "#1E293B"},
                    ],
                    "threshold": {"line": {"color": "#06B6D4", "width": 3},
                                  "thickness": 0.75, "value": avg_spend},
                },
                title={"text": "Spend vs Dataset Max", "font": {"color": "#64748B", "size": 11}},
            ))
            fig_gauge.update_layout(
                height=220,
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font_color="#94A3B8",
                margin=dict(l=20, r=20, t=40, b=10),
            )
            st.plotly_chart(fig_gauge, use_container_width=True)

            st.markdown(f"""
            <div class="info-box">
                🔵 <b>Avg baseline</b>: ₹{avg_spend:,.0f} &nbsp;|&nbsp;
                Your prediction is <b style="color:#A78BFA;">
                {'above' if pred > avg_spend else 'below'} average
                </b> by ₹{abs(pred - avg_spend):,.0f}
            </div>
            """, unsafe_allow_html=True)

        else:
            st.markdown("""
            <div style="background:rgba(15,23,42,0.6); border:2px dashed rgba(100,116,139,0.2);
                        border-radius:20px; padding:3rem 2rem; text-align:center;">
                <div style="font-size:3rem; margin-bottom:1rem;">🎯</div>
                <p style="color:#475569; font-size:0.9rem; font-weight:500;">
                    Fill in the customer profile on the left<br>and click
                    <strong style="color:#A78BFA">Predict Spending Amount</strong>
                    to see the result.
                </p>
            </div>
            """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════
#  TAB 2  —  DATA INSIGHTS
# ══════════════════════════════════════════════════════════════════
with tab2:
    st.markdown("""
    <div class="section-header">
      <span style="font-size:1.2rem;">📊</span>
      <h2 class="section-title">Exploratory Data Insights</h2>
      <span class="section-pill">ShopSmart Dataset</span>
    </div>
    """, unsafe_allow_html=True)

    CHART_BG   = "rgba(0,0,0,0)"
    GRID_COLOR = "rgba(100,116,139,0.1)"
    FONT_COLOR = "#94A3B8"

    def style_fig(fig, height=340):
        fig.update_layout(
            height=height,
            paper_bgcolor=CHART_BG,
            plot_bgcolor=CHART_BG,
            font=dict(color=FONT_COLOR, family="Inter"),
            margin=dict(l=10, r=10, t=40, b=20),
            legend=dict(bgcolor="rgba(0,0,0,0)", font_color=FONT_COLOR),
        )
        fig.update_xaxes(gridcolor=GRID_COLOR, zeroline=False, color=FONT_COLOR)
        fig.update_yaxes(gridcolor=GRID_COLOR, zeroline=False, color=FONT_COLOR)
        return fig

    # ── Row 1: Spend Distribution + Membership Breakdown ──
    r1c1, r1c2 = st.columns(2, gap="medium")

    with r1c1:
        fig1 = px.histogram(
            df, x="Amount_Spent", nbins=40,
            title="Spend Distribution",
            color_discrete_sequence=["#7C3AED"],
        )
        fig1.update_traces(opacity=0.85)
        fig1.add_vline(x=avg_spend, line_dash="dash", line_color="#06B6D4",
                        annotation_text=f"Avg ₹{avg_spend:,.0f}",
                        annotation_font_color="#06B6D4")
        st.plotly_chart(style_fig(fig1), use_container_width=True)

    with r1c2:
        spend_by_mem = df.groupby("Membership_Type")["Amount_Spent"].mean().reset_index()
        fig2 = px.bar(
            spend_by_mem, x="Membership_Type", y="Amount_Spent",
            title="Avg Spend by Membership Tier",
            color="Amount_Spent",
            color_continuous_scale=["#1E1035", "#7C3AED", "#06B6D4"],
        )
        fig2.update_coloraxes(showscale=False)
        st.plotly_chart(style_fig(fig2), use_container_width=True)

    # ── Row 2: Category + City Tier ──
    r2c1, r2c2 = st.columns(2, gap="medium")

    with r2c1:
        cat_spend = df.groupby("Preferred_Category")["Amount_Spent"].mean().reset_index()
        fig3 = px.pie(
            cat_spend, values="Amount_Spent", names="Preferred_Category",
            title="Avg Spend by Category",
            color_discrete_sequence=["#7C3AED","#06B6D4","#8B5CF6","#F59E0B"],
            hole=0.45,
        )
        fig3.update_traces(textfont_color="#fff")
        st.plotly_chart(style_fig(fig3), use_container_width=True)

    with r2c2:
        city_spend = df.groupby("City_Tier")["Amount_Spent"].mean().reset_index()
        fig4 = px.bar(
            city_spend, x="City_Tier", y="Amount_Spent",
            title="Avg Spend by City Tier",
            color="Amount_Spent",
            color_continuous_scale=["#1E1035","#8B5CF6","#06B6D4"],
        )
        fig4.update_coloraxes(showscale=False)
        st.plotly_chart(style_fig(fig4), use_container_width=True)

    # ── Row 3: Income vs Spend scatter + Age distribution ──
    r3c1, r3c2 = st.columns(2, gap="medium")

    with r3c1:
        fig5 = px.scatter(
            df.sample(min(500, len(df)), random_state=42),
            x="Annual_Income", y="Amount_Spent",
            color="Membership_Type",
            title="Income vs Spend (sample 500)",
            color_discrete_sequence=["#7C3AED","#06B6D4","#F59E0B"],
            opacity=0.7,
        )
        st.plotly_chart(style_fig(fig5, height=360), use_container_width=True)

    with r3c2:
        fig6 = px.box(
            df, x="Membership_Type", y="Amount_Spent",
            color="Membership_Type",
            title="Spend Boxplot by Membership",
            color_discrete_sequence=["#7C3AED","#06B6D4","#F59E0B"],
        )
        fig6.update_traces(showlegend=False)
        st.plotly_chart(style_fig(fig6, height=360), use_container_width=True)

    # ── Correlation Table ──
    st.markdown("""
    <div class="section-header" style="margin-top:2.5rem;">
      <span style="font-size:1.2rem;">🔗</span>
      <h2 class="section-title">Feature Correlation with Amount Spent</h2>
    </div>
    """, unsafe_allow_html=True)

    num_cols = ["Age","Annual_Income","Website_Visits","Previous_Purchases","Amount_Spent"]
    corr_df  = df[num_cols].corr()[["Amount_Spent"]].drop("Amount_Spent").rename(columns={"Amount_Spent":"Correlation"})
    corr_df["Abs"] = corr_df["Correlation"].abs()
    corr_df = corr_df.sort_values("Abs", ascending=False).drop(columns=["Abs"])
    corr_df["Bar"] = corr_df["Correlation"]

    fig_corr = px.bar(
        corr_df.reset_index(), x="index", y="Correlation",
        title="Pearson Correlation with Amount Spent",
        color="Correlation",
        color_continuous_scale=["#06B6D4","#0F172A","#7C3AED"],
        color_continuous_midpoint=0,
    )
    fig_corr.add_hline(y=0, line_color="rgba(100,116,139,0.4)")
    st.plotly_chart(style_fig(fig_corr, height=300), use_container_width=True)


# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("<br>", unsafe_allow_html=True)
st.markdown('<hr class="grad-divider">', unsafe_allow_html=True)
st.markdown("""
<div style="text-align:center; padding:1rem 0;">
    <p style="color:#334155; font-size:0.78rem; margin:0;">
        🚀 <strong style="color:#475569;">SpendScope</strong> &nbsp;·&nbsp;
        Customer Spending Prediction &nbsp;·&nbsp;
        Built with ❤️ using <strong style="color:#7C3AED;">scikit-learn</strong>
        + <strong style="color:#06B6D4;">Streamlit</strong>
    </p>
</div>
""", unsafe_allow_html=True)
