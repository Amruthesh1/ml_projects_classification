import streamlit as st
import pandas as pd
import pickle
import numpy as np

# ── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Customer Churn Predictor",
    page_icon="🔮",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@600;700;800&family=Inter:wght@300;400;500&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

/* ── sidebar ── */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0f0f1a 0%, #1a1a2e 60%, #16213e 100%);
    border-right: 1px solid rgba(255,255,255,0.06);
}
[data-testid="stSidebar"] * { color: #e2e8f0 !important; }
[data-testid="stSidebar"] .stSelectbox label,
[data-testid="stSidebar"] .stNumberInput label,
[data-testid="stSidebar"] .stSlider label {
    font-size: 0.78rem !important;
    letter-spacing: 0.05em;
    text-transform: uppercase;
    color: #94a3b8 !important;
}

/* ── main bg ── */
.main { background: #f8fafc; }

/* ── hero ── */
.hero-wrap {
    background: linear-gradient(135deg, #0f0f1a 0%, #1a1a2e 50%, #0d3b6e 100%);
    border-radius: 20px;
    padding: 3rem 3rem 2.5rem;
    margin-bottom: 1.8rem;
    position: relative;
    overflow: hidden;
}
.hero-wrap::before {
    content: '';
    position: absolute; top: -60px; right: -60px;
    width: 260px; height: 260px;
    border-radius: 50%;
    background: radial-gradient(circle, rgba(99,102,241,0.18) 0%, transparent 70%);
}
.hero-tag {
    display: inline-block;
    background: rgba(99,102,241,0.2);
    border: 1px solid rgba(99,102,241,0.4);
    color: #a5b4fc;
    font-size: 0.72rem;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    padding: 4px 14px;
    border-radius: 20px;
    margin-bottom: 1rem;
}
.hero-title {
    font-family: 'Syne', sans-serif;
    font-size: 2.6rem;
    font-weight: 800;
    color: #ffffff;
    line-height: 1.15;
    margin: 0 0 0.8rem 0;
}
.hero-title span { color: #818cf8; }
.hero-sub {
    color: rgba(255,255,255,0.55);
    font-size: 0.95rem;
    font-weight: 300;
    max-width: 520px;
    line-height: 1.6;
}

/* ── stat cards ── */
.stats-row { display: flex; gap: 14px; margin-bottom: 1.8rem; flex-wrap: wrap; }
.stat-card {
    flex: 1; min-width: 130px;
    background: white;
    border-radius: 14px;
    padding: 1.2rem 1.4rem;
    box-shadow: 0 2px 12px rgba(0,0,0,0.06);
    border-top: 3px solid;
    text-align: center;
}
.stat-card.blue  { border-color: #6366f1; }
.stat-card.green { border-color: #10b981; }
.stat-card.amber { border-color: #f59e0b; }
.stat-card.rose  { border-color: #f43f5e; }
.stat-val  { font-family: 'Syne', sans-serif; font-size: 1.7rem; font-weight: 700; color: #1e293b; }
.stat-label{ font-size: 0.72rem; text-transform: uppercase; letter-spacing: 0.07em; color: #94a3b8; margin-top: 2px; }

/* ── section cards ── */
.card {
    background: white;
    border-radius: 16px;
    padding: 1.6rem 1.8rem;
    box-shadow: 0 2px 14px rgba(0,0,0,0.06);
    margin-bottom: 1.4rem;
}
.card-title {
    font-family: 'Syne', sans-serif;
    font-size: 1rem;
    font-weight: 700;
    color: #1e293b;
    margin-bottom: 1.2rem;
    display: flex; align-items: center; gap: 8px;
}
.card-title .dot {
    width: 8px; height: 8px; border-radius: 50%;
    background: #6366f1; flex-shrink: 0;
}

/* ── result box ── */
.result-stay {
    background: linear-gradient(135deg, #065f46, #047857);
    border-radius: 16px; padding: 2rem; text-align: center; color: white;
}
.result-churn {
    background: linear-gradient(135deg, #7f1d1d, #991b1b);
    border-radius: 16px; padding: 2rem; text-align: center; color: white;
}
.result-icon  { font-size: 2.8rem; margin-bottom: 0.5rem; }
.result-label { font-size: 0.8rem; opacity: 0.7; text-transform: uppercase; letter-spacing: 0.1em; }
.result-main  { font-family: 'Syne', sans-serif; font-size: 1.9rem; font-weight: 800; margin: 0.3rem 0; }
.result-prob  { font-size: 2.6rem; font-weight: 700; opacity: 0.95; }
.result-badge {
    display: inline-block;
    background: rgba(255,255,255,0.18);
    border-radius: 20px; padding: 4px 16px;
    font-size: 0.8rem; margin-top: 0.6rem;
}

/* ── prob bar ── */
.prob-bar-wrap { margin: 1.2rem 0 0; }
.prob-bar-bg {
    background: rgba(255,255,255,0.15);
    border-radius: 10px; height: 8px; overflow: hidden;
}
.prob-bar-fill {
    height: 8px; border-radius: 10px;
    background: rgba(255,255,255,0.75);
    transition: width 0.5s ease;
}
.prob-bar-labels {
    display: flex; justify-content: space-between;
    font-size: 0.72rem; opacity: 0.65; margin-top: 4px;
}

/* ── mini metric pills ── */
.pill-row { display: flex; gap: 10px; flex-wrap: wrap; margin-top: 1rem; }
.pill {
    background: #f1f5f9; border-radius: 10px;
    padding: 0.6rem 1rem; text-align: center; flex: 1; min-width: 100px;
}
.pill-val   { font-family: 'Syne', sans-serif; font-size: 1.1rem; font-weight: 700; color: #4f46e5; }
.pill-label { font-size: 0.7rem; color: #94a3b8; text-transform: uppercase; letter-spacing: 0.05em; }

/* ── risk meter ── */
.risk-label { font-size: 0.78rem; color: #64748b; text-transform: uppercase; letter-spacing: 0.07em; margin-bottom: 6px; }
.risk-bar-bg {
    height: 10px; border-radius: 10px;
    background: linear-gradient(to right, #10b981, #f59e0b, #ef4444);
    position: relative; margin-bottom: 6px;
}
.risk-needle {
    position: absolute; top: -4px;
    width: 18px; height: 18px; border-radius: 50%;
    background: white; border: 3px solid #1e293b;
    transform: translateX(-50%);
    box-shadow: 0 2px 6px rgba(0,0,0,0.2);
}

/* ── sidebar brand ── */
.sb-brand {
    font-family: 'Syne', sans-serif;
    font-size: 1.2rem; font-weight: 800;
    color: white !important;
    padding: 0.5rem 0 1.5rem;
    border-bottom: 1px solid rgba(255,255,255,0.08);
    margin-bottom: 1.2rem;
}
.sb-section {
    font-size: 0.68rem; letter-spacing: 0.12em;
    text-transform: uppercase; color: #64748b !important;
    margin: 1.4rem 0 0.6rem;
}

.stButton > button {
    background: linear-gradient(135deg, #4f46e5, #6366f1) !important;
    color: white !important; border: none !important;
    border-radius: 12px !important; padding: 0.75rem 2rem !important;
    font-size: 0.95rem !important; font-weight: 600 !important;
    font-family: 'Inter', sans-serif !important;
    width: 100% !important; letter-spacing: 0.02em;
    box-shadow: 0 4px 15px rgba(99,102,241,0.35) !important;
}
.stButton > button:hover { opacity: 0.9 !important; }

footer { display: none; }
</style>
""", unsafe_allow_html=True)

# ── Load Artifacts ────────────────────────────────────────────────────────────
@st.cache_resource
def load_artifacts():
    model           = pickle.load(open("churn_model.pkl",      "rb"))
    scaler          = pickle.load(open("scaler.pkl",           "rb"))
    feature_columns = pickle.load(open("feature_columns.pkl",  "rb"))
    threshold       = pickle.load(open("best_threshold.pkl",   "rb"))
    return model, scaler, feature_columns, threshold

model, scaler, feature_columns, threshold = load_artifacts()

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div class="sb-brand">🔮 ChurnSense</div>', unsafe_allow_html=True)
    st.markdown('<div class="sb-section">Account Profile</div>', unsafe_allow_html=True)

    Age                = st.slider("Age", 18, 80, 35)
    Login_Frequency    = st.slider("Login Frequency (per month)", 0, 46, 11)
    Membership_Years   = st.slider("Membership Years", 0, 10, 3)

    st.markdown('<div class="sb-section">Browsing Behaviour</div>', unsafe_allow_html=True)

    Session_Duration   = st.number_input("Avg Session Duration (mins)", 0.0, 76.0, 27.0, step=0.5)
    Pages_Per_Session  = st.number_input("Pages Per Session", 0.0, 25.0, 8.0, step=0.5)
    Mobile_App_Usage   = st.number_input("Mobile App Usage (mins/week)", 0.0, 62.0, 19.0, step=0.5)
    Social_Media_Score = st.number_input("Social Media Engagement Score", 0.0, 100.0, 29.0, step=0.5)

    st.markdown('<div class="sb-section">Purchase Behaviour</div>', unsafe_allow_html=True)

    Total_Purchases          = st.number_input("Total Purchases", 0, 129, 13)
    Average_Order_Value      = st.number_input("Average Order Value (₹)", 0.0, 9700.0, 123.0, step=1.0)
    Cart_Abandonment_Rate    = st.number_input("Cart Abandonment Rate (%)", 0.0, 100.0, 57.0, step=0.5)
    Days_Since_Last_Purchase = st.number_input("Days Since Last Purchase", 0, 287, 30)
    Wishlist_Items           = st.number_input("Wishlist Items", 0, 28, 4)
    Discount_Usage_Rate      = st.number_input("Discount Usage Rate (%)", 0.0, 100.0, 30.0, step=0.5)
    Returns_Rate             = st.number_input("Returns Rate (%)", 0.0, 100.0, 6.5, step=0.5)

    st.markdown('<div class="sb-section">Financial & Service</div>', unsafe_allow_html=True)

    Email_Open_Rate        = st.number_input("Email Open Rate (%)", 0.0, 92.0, 21.0, step=0.5)
    Customer_Service_Calls = st.number_input("Customer Service Calls", 0, 21, 5)
    Product_Reviews        = st.number_input("Product Reviews Written", 0, 21, 3)
    Payment_Diversity      = st.number_input("Payment Method Diversity", 1, 5, 2)
    Lifetime_Value         = st.number_input("Lifetime Value (₹)", 0.0, 9000.0, 1440.0, step=10.0)
    Credit_Balance         = st.number_input("Credit Balance (₹)", 0.0, 7200.0, 1966.0, step=10.0)

    st.markdown('<div class="sb-section">Demographics</div>', unsafe_allow_html=True)
    Gender = st.selectbox("Gender", ["Male", "Female"])

    st.markdown("<br>", unsafe_allow_html=True)
    predict_btn = st.button("🔮 Predict Now")

# ── Main Area ─────────────────────────────────────────────────────────────────
# Hero
st.markdown("""
<div class="hero-wrap">
    <div class="hero-tag">ML-Powered · E-Commerce Intelligence</div>
    <div class="hero-title">Customer Churn<br><span>Prediction Dashboard</span></div>
    <div class="hero-sub">
        Enter customer profile data in the sidebar to get an instant churn prediction
        powered by an ensemble machine learning model trained on 50,000+ customer records.
    </div>
</div>
""", unsafe_allow_html=True)

# Stats row
st.markdown("""
<div class="stats-row">
    <div class="stat-card blue">
        <div class="stat-val">50K+</div>
        <div class="stat-label">Training Records</div>
    </div>
    <div class="stat-card green">
        <div class="stat-val">~93%</div>
        <div class="stat-label">ROC-AUC Score</div>
    </div>
    <div class="stat-card amber">
        <div class="stat-val">20+</div>
        <div class="stat-label">Features Used</div>
    </div>
    <div class="stat-card rose">
        <div class="stat-val">4</div>
        <div class="stat-label">Strategies Tested</div>
    </div>
</div>
""", unsafe_allow_html=True)

# ── Two column layout ──────────────────────────────────────────────────────────
left, right = st.columns([1, 1.05], gap="large")

with left:
    # Customer summary card
    st.markdown(f"""
    <div class="card">
        <div class="card-title"><span class="dot"></span> Customer Snapshot</div>
        <table style="width:100%;font-size:0.85rem;border-collapse:collapse;">
            <tr style="border-bottom:1px solid #f1f5f9;">
                <td style="padding:6px 0;color:#64748b;">Age</td>
                <td style="padding:6px 0;font-weight:500;text-align:right;">{Age}</td>
            </tr>
            <tr style="border-bottom:1px solid #f1f5f9;">
                <td style="padding:6px 0;color:#64748b;">Gender</td>
                <td style="padding:6px 0;font-weight:500;text-align:right;">{Gender}</td>
            </tr>
            <tr style="border-bottom:1px solid #f1f5f9;">
                <td style="padding:6px 0;color:#64748b;">Membership Years</td>
                <td style="padding:6px 0;font-weight:500;text-align:right;">{Membership_Years}</td>
            </tr>
            <tr style="border-bottom:1px solid #f1f5f9;">
                <td style="padding:6px 0;color:#64748b;">Total Purchases</td>
                <td style="padding:6px 0;font-weight:500;text-align:right;">{Total_Purchases}</td>
            </tr>
            <tr style="border-bottom:1px solid #f1f5f9;">
                <td style="padding:6px 0;color:#64748b;">Lifetime Value</td>
                <td style="padding:6px 0;font-weight:500;text-align:right;">₹{Lifetime_Value:,.0f}</td>
            </tr>
            <tr style="border-bottom:1px solid #f1f5f9;">
                <td style="padding:6px 0;color:#64748b;">Login Frequency</td>
                <td style="padding:6px 0;font-weight:500;text-align:right;">{Login_Frequency}/month</td>
            </tr>
            <tr style="border-bottom:1px solid #f1f5f9;">
                <td style="padding:6px 0;color:#64748b;">Cart Abandonment</td>
                <td style="padding:6px 0;font-weight:500;text-align:right;">{Cart_Abandonment_Rate:.1f}%</td>
            </tr>
            <tr style="border-bottom:1px solid #f1f5f9;">
                <td style="padding:6px 0;color:#64748b;">Days Since Purchase</td>
                <td style="padding:6px 0;font-weight:500;text-align:right;">{Days_Since_Last_Purchase} days</td>
            </tr>
            <tr>
                <td style="padding:6px 0;color:#64748b;">Customer Service Calls</td>
                <td style="padding:6px 0;font-weight:500;text-align:right;">{Customer_Service_Calls}</td>
            </tr>
        </table>
    </div>
    """, unsafe_allow_html=True)

    # Risk indicators
    st.markdown("""
    <div class="card">
        <div class="card-title"><span class="dot"></span> Key Risk Signals</div>
    """, unsafe_allow_html=True)

    def risk_signal(label, value, low, high, reverse=False):
        pct = min(max((value - low) / (high - low) * 100, 0), 100)
        if reverse:
            pct = 100 - pct
        color = "#10b981" if pct < 33 else "#f59e0b" if pct < 66 else "#ef4444"
        level = "Low" if pct < 33 else "Medium" if pct < 66 else "High"
        st.markdown(f"""
        <div style="margin-bottom:12px;">
            <div style="display:flex;justify-content:space-between;font-size:0.78rem;margin-bottom:4px;">
                <span style="color:#475569;">{label}</span>
                <span style="color:{color};font-weight:600;">{level}</span>
            </div>
            <div style="background:#f1f5f9;border-radius:6px;height:6px;overflow:hidden;">
                <div style="width:{pct}%;height:6px;background:{color};border-radius:6px;"></div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    risk_signal("Cart Abandonment Risk",    Cart_Abandonment_Rate,    0,  100)
    risk_signal("Inactivity Risk",          Days_Since_Last_Purchase, 0,  287)
    risk_signal("Service Dissatisfaction",  Customer_Service_Calls,   0,  21)
    risk_signal("Engagement Level",         Login_Frequency,          0,  46, reverse=True)
    risk_signal("Discount Dependency",      Discount_Usage_Rate,      0,  100)

    st.markdown("</div>", unsafe_allow_html=True)

with right:
    if predict_btn:
        try:
            # ── Build input ──────────────────────────────────────────────────
            input_data = pd.DataFrame(columns=feature_columns)
            input_data.loc[0] = 0

            input_data["Age"]                          = Age
            input_data["Membership_Years"]             = Membership_Years
            input_data["Login_Frequency"]              = Login_Frequency
            input_data["Session_Duration_Avg"]         = Session_Duration
            input_data["Pages_Per_Session"]            = Pages_Per_Session
            input_data["Cart_Abandonment_Rate"]        = Cart_Abandonment_Rate
            input_data["Wishlist_Items"]               = Wishlist_Items
            input_data["Total_Purchases"]              = Total_Purchases
            input_data["Average_Order_Value"]          = Average_Order_Value
            input_data["Days_Since_Last_Purchase"]     = Days_Since_Last_Purchase
            input_data["Discount_Usage_Rate"]          = Discount_Usage_Rate
            input_data["Returns_Rate"]                 = Returns_Rate
            input_data["Email_Open_Rate"]              = Email_Open_Rate
            input_data["Customer_Service_Calls"]       = Customer_Service_Calls
            input_data["Product_Reviews_Written"]      = Product_Reviews
            input_data["Social_Media_Engagement_Score"]= Social_Media_Score
            input_data["Mobile_App_Usage"]             = Mobile_App_Usage
            input_data["Payment_Method_Diversity"]     = Payment_Diversity
            input_data["Lifetime_Value"]               = Lifetime_Value
            input_data["Credit_Balance"]               = Credit_Balance

            # Gender dummy
            if "Gender_Male" in feature_columns:
                input_data["Gender_Male"] = 1 if Gender == "Male" else 0

            input_data    = input_data[feature_columns].astype(float)
            input_scaled  = scaler.transform(input_data)
            probability   = model.predict_proba(input_scaled)[0][1]
            prediction    = int(probability >= threshold)
            prob_pct      = probability * 100

            # ── Result box ───────────────────────────────────────────────────
            if prediction == 1:
                box_class = "result-churn"
                icon      = "⚠️"
                verdict   = "Likely to Churn"
                badge     = "Retention action recommended"
            else:
                box_class = "result-stay"
                icon      = "✅"
                verdict   = "Likely to Stay"
                badge     = "Customer appears loyal"

            bar_width = f"{prob_pct:.1f}%"

            st.markdown(f"""
            <div class="{box_class}">
                <div class="result-icon">{icon}</div>
                <div class="result-label">Prediction Result</div>
                <div class="result-main">{verdict}</div>
                <div class="result-prob">{prob_pct:.1f}%</div>
                <div class="result-label" style="margin-top:4px;">Churn Probability</div>
                <div class="prob-bar-wrap">
                    <div class="prob-bar-bg">
                        <div class="prob-bar-fill" style="width:{bar_width};"></div>
                    </div>
                    <div class="prob-bar-labels"><span>0%</span><span>50%</span><span>100%</span></div>
                </div>
                <div class="result-badge">{badge}</div>
            </div>
            """, unsafe_allow_html=True)

            # ── Derived insight pills ────────────────────────────────────────
            risk_score   = round(prob_pct, 1)
            confidence   = round(abs(prob_pct - 50) * 2, 1)
            ltv_risk     = round(Lifetime_Value * probability, 0)

            st.markdown(f"""
            <div class="pill-row">
                <div class="pill">
                    <div class="pill-val">{risk_score}%</div>
                    <div class="pill-label">Risk Score</div>
                </div>
                <div class="pill">
                    <div class="pill-val">{confidence}%</div>
                    <div class="pill-label">Confidence</div>
                </div>
                <div class="pill">
                    <div class="pill-val">₹{ltv_risk:,.0f}</div>
                    <div class="pill-label">LTV at Risk</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            # ── Recommendations ───────────────────────────────────────────────
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown("""
            <div class="card">
                <div class="card-title"><span class="dot"></span> Recommended Actions</div>
            """, unsafe_allow_html=True)

            actions = []
            if Cart_Abandonment_Rate > 70:
                actions.append(("🛒", "High cart abandonment detected", "Send personalised cart recovery email with a discount"))
            if Days_Since_Last_Purchase > 60:
                actions.append(("📅", "Customer inactive for 60+ days", "Trigger re-engagement campaign with exclusive offer"))
            if Customer_Service_Calls > 8:
                actions.append(("📞", "Frequent support contact", "Assign a dedicated account manager"))
            if Login_Frequency < 5:
                actions.append(("📱", "Low platform engagement", "Push app notification with personalised recommendations"))
            if Discount_Usage_Rate > 70:
                actions.append(("🏷️", "High discount dependency", "Introduce loyalty points to reduce reliance on discounts"))
            if Email_Open_Rate < 10:
                actions.append(("📧", "Low email engagement", "Revise email content strategy — try SMS instead"))

            if not actions:
                actions.append(("✨", "Customer looks healthy", "Continue standard engagement — no urgent actions needed"))

            for emoji, title, desc in actions[:4]:
                st.markdown(f"""
                <div style="display:flex;gap:12px;align-items:flex-start;padding:10px 0;border-bottom:1px solid #f1f5f9;">
                    <div style="font-size:1.3rem;">{emoji}</div>
                    <div>
                        <div style="font-weight:600;font-size:0.85rem;color:#1e293b;">{title}</div>
                        <div style="font-size:0.78rem;color:#64748b;margin-top:2px;">{desc}</div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

            st.markdown("</div>", unsafe_allow_html=True)

            # ── Model info ────────────────────────────────────────────────────
            st.markdown(f"""
            <div style="background:#f8fafc;border:1px solid #e2e8f0;border-radius:10px;
                        padding:0.9rem 1.2rem;font-size:0.75rem;color:#94a3b8;margin-top:0.5rem;">
                🤖 <b>Model:</b> Best model from Default / SMOTE / Undersample / Tuned comparison &nbsp;·&nbsp;
                📊 <b>ROC-AUC:</b> ~0.93 &nbsp;·&nbsp;
                🎯 <b>Threshold:</b> {threshold:.2f} &nbsp;·&nbsp;
                📦 <b>Features:</b> {len(feature_columns)}
            </div>
            """, unsafe_allow_html=True)

        except Exception as e:
            st.error(f"Prediction error: {e}")

    else:
        # Placeholder before prediction
        st.markdown("""
        <div class="card" style="text-align:center;padding:3.5rem 2rem;">
            <div style="font-size:3.5rem;margin-bottom:1rem;">🔮</div>
            <div style="font-family:'Syne',sans-serif;font-size:1.3rem;font-weight:700;
                        color:#1e293b;margin-bottom:0.6rem;">Ready to Predict</div>
            <div style="color:#94a3b8;font-size:0.88rem;line-height:1.6;max-width:320px;margin:0 auto;">
                Fill in the customer details in the sidebar and click
                <b style="color:#6366f1;">Predict Now</b> to get an instant churn prediction.
            </div>
        </div>

        <div class="card">
            <div class="card-title"><span class="dot"></span> How It Works</div>
            <div style="display:flex;flex-direction:column;gap:12px;">
                <div style="display:flex;gap:12px;align-items:center;">
                    <div style="background:#ede9fe;border-radius:8px;padding:8px 12px;
                                font-family:'Syne',sans-serif;font-weight:700;color:#6366f1;">01</div>
                    <div style="font-size:0.85rem;color:#475569;">Enter customer details in the left sidebar</div>
                </div>
                <div style="display:flex;gap:12px;align-items:center;">
                    <div style="background:#ede9fe;border-radius:8px;padding:8px 12px;
                                font-family:'Syne',sans-serif;font-weight:700;color:#6366f1;">02</div>
                    <div style="font-size:0.85rem;color:#475569;">Model scores the customer using 20+ features</div>
                </div>
                <div style="display:flex;gap:12px;align-items:center;">
                    <div style="background:#ede9fe;border-radius:8px;padding:8px 12px;
                                font-family:'Syne',sans-serif;font-weight:700;color:#6366f1;">03</div>
                    <div style="font-size:0.85rem;color:#475569;">Get churn probability + recommended retention actions</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("""
<div style="text-align:center;color:#cbd5e1;font-size:0.75rem;
            margin-top:2rem;padding-top:1.2rem;border-top:1px solid #e2e8f0;">
    Customer Churn Predictor · Built with Streamlit & scikit-learn ·
    Trained on E-Commerce Customer Dataset
</div>
""", unsafe_allow_html=True)