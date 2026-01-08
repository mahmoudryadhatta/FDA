# =========================================================
# FUZZY DELPHI ANALYZER — PHASE 1 (PAY-PER-PROJECT)
# Author: Dr. Mahmood Riyadh
# Payments: PayPal REST API (automatic verification; no SDK)
# =========================================================

import os
import hashlib
from io import BytesIO
import base64

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st

# =========================================================
# CONFIG
# =========================================================
st.set_page_config(
    page_title="Fuzzy Delphi Analyzer",
    page_icon="Logo.png",   # ✅ use your logo as favicon
    layout="wide"
)

FREE_LIMIT = 2
PRICE_USD = 14.00  # ✅ fixed price (updated)
APP_BASE_URL = os.getenv("APP_BASE_URL", "http://localhost:8501").rstrip("/")

PAYPAL_CLIENT_ID = os.getenv("PAYPAL_CLIENT_ID", "")
PAYPAL_SECRET = os.getenv("PAYPAL_SECRET", "")
PAYPAL_MODE = os.getenv("PAYPAL_MODE", "sandbox").lower().strip()  # sandbox | live
PAYPAL_API_BASE = "https://api-m.sandbox.paypal.com" if PAYPAL_MODE != "live" else "https://api-m.paypal.com"

st.markdown(
    """
    <head>
        <meta name="google-site-verification" content="CALz7HPWEhHE3Hnirgb62FHq-QbGbfz7UH0HPu1qf3A" />
    </head>
    """,
    unsafe_allow_html=True
)


# =========================================================
# OWNER INFO
# =========================================================
OWNER = {
    "name": "Dr. Mahmood Riyadh",
    "email": "mahmoud.r.ata@gmail.com",
    "phone": "+60 1117716353",
    "whatsapp": "https://wa.me/601117716353",
}

# =========================================================
# THEME COLORS
# =========================================================
ACCENT = "#60a5fa"   # blue
GOOD = "#22c55e"     # green
BAD = "#ef4444"      # red
WARN = "#f59e0b"     # amber

# =========================================================
# BACKGROUND IMAGE + PROFESSIONAL THEME (SAFE + SCROLL FIXED)
# =========================================================
def set_bg_from_local_png(png_file: str):
    with open(png_file, "rb") as f:
        data = f.read()
    b64 = base64.b64encode(data).decode()

    st.markdown(
        f"""
        <style>
        /* ---------- App background (image + dark overlay) ---------- */
        [data-testid="stAppViewContainer"] {{
            background:
              linear-gradient(rgba(6, 10, 18, 0.72), rgba(6, 10, 18, 0.72)),
              url("data:image/png;base64,{b64}");
            background-size: cover;
            background-position: center;
            background-repeat: no-repeat;
            background-attachment: fixed;
        }}

        /* ---------- Layout spacing ---------- */
        .block-container {{
            padding-top: 1.2rem;
            padding-bottom: 2.5rem;
        }}

        /* ---------- Sidebar ---------- */
        section[data-testid="stSidebar"] {{
            background: rgba(5, 8, 14, 0.82);
            border-right: 1px solid rgba(255,255,255,0.10);
            backdrop-filter: blur(10px);
        }}

        /* ---------- Global text ---------- */
        html, body, [data-testid="stAppViewContainer"] {{
            color: #EAF0FF !important;
        }}

       /* =========================================================
   SECTION TITLE FRAMES (VISIBLE TITLES + LEFT PADDING)
   ========================================================= */

/* Target Streamlit markdown headings (most reliable selectors) */
[data-testid="stMarkdownContainer"] h1,
[data-testid="stMarkdownContainer"] h2,
[data-testid="stMarkdownContainer"] h3,
[data-testid="stMarkdownContainer"] h4 {{
    display: inline-block !important;
    padding: 8px 14px !important;
    padding-left: 28px !important;   /* ✅ adds "1–2 spaces" look */
    margin: 6px 0 10px 0 !important;
    border-radius: 12px !important;
    background: rgba(8, 12, 20, 0.78) !important;
    border: 1px solid rgba(255,255,255,0.18) !important;
    backdrop-filter: blur(8px) !important;
    box-shadow: 0 6px 20px rgba(0,0,0,0.35) !important;
    color: #F3F7FF !important;
}}

/* Slightly smaller for lower headings */
[data-testid="stMarkdownContainer"] h3,
[data-testid="stMarkdownContainer"] h4 {{
    padding: 6px 12px !important;
    padding-left: 24px !important;   /* ✅ same idea but balanced */
}}


        /* ---------- pill + small-muted ---------- */
        .pill {{
            display:inline-block;
            padding: 6px 10px;
            border-radius: 999px;
            background: rgba(15, 24, 40, 0.72);
            border: 1px solid rgba(255,255,255,0.14);
            font-size: 0.92rem;
        }}
        .small-muted {{
            color: rgba(234,240,255,0.78) !important;
            font-size: 0.95rem;
            line-height: 1.4;
        }}

        /* ---------- KPI cards ---------- */
        .kpi {{
            background: rgba(15, 24, 40, 0.72);
            border: 1px solid rgba(255,255,255,0.14);
            border-radius: 18px;
            padding: 14px 16px;
            backdrop-filter: blur(10px);
            box-shadow: 0 10px 30px rgba(0,0,0,0.28);
        }}
        .kpi .t {{ font-size: 0.9rem; opacity: 0.85; }}
        .kpi .v {{ font-size: 1.7rem; font-weight: 800; margin-top: 4px; }}
        .kpi .s {{ font-size: 0.85rem; opacity: 0.78; margin-top: 2px; }}

        .hr {{
            border-top: 1px solid rgba(255,255,255,0.14);
            margin: 14px 0;
        }}

        /* ---------- Plotly chart container ---------- */
        [data-testid="stPlotlyChart"] {{
            background: rgba(10, 16, 28, 0.90) !important;
            border: 1px solid rgba(255,255,255,0.14) !important;
            border-radius: 18px !important;
            padding: 14px !important;
            backdrop-filter: blur(10px);
            box-shadow: 0 10px 30px rgba(0,0,0,0.25);
            overflow: hidden !important;
        }}
        [data-testid="stPlotlyChart"] > div {{
            background: transparent !important;
            border: none !important;
            padding: 0 !important;
        }}

        /* ---------- Dataframe container ---------- */
        [data-testid="stDataFrame"] {{
            background: rgba(10, 16, 28, 0.82);
            border: 1px solid rgba(255,255,255,0.14);
            border-radius: 18px;
            padding: 8px;
            backdrop-filter: blur(10px);
        }}

        /* ---------- Alerts ---------- */
        div[data-testid="stAlert"] {{
            background: rgba(10, 16, 28, 0.82) !important;
            border: 1px solid rgba(255,255,255,0.14) !important;
            border-radius: 16px !important;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )

# MUST be your real filename in same folder as app.py
set_bg_from_local_png("Background.png")
st.markdown(
    """
    <style>
    /* Make ONLY the download button text black */
    div[data-testid="stDownloadButton"] button {
        color: #000000 !important;          /* black text */
        background-color: #ffffff !important;
        border: 1px solid rgba(0,0,0,0.15);
        font-weight: 600;
    }

    /* Optional: hover effect */
    div[data-testid="stDownloadButton"] button:hover {
        background-color: #f3f4f6 !important;
        color: #000000 !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# =========================================================
# PLOT THEME (High readability on image backgrounds)
# =========================================================
PLOT_BG  = "rgba(8, 12, 20, 0.94)"
PAPER_BG = "rgba(0, 0, 0, 0)"
GRID_CLR = "rgba(255,255,255,0.14)"
FONT_CLR = "#F3F7FF"

def style_fig(fig, height=None, extra_margin_right=35):
    fig.update_layout(
        plot_bgcolor=PLOT_BG,
        paper_bgcolor=PAPER_BG,
        font=dict(color=FONT_CLR, size=14),
        title_font=dict(size=18, color=FONT_CLR),
        xaxis=dict(showgrid=True, gridcolor=GRID_CLR, zeroline=False, tickfont=dict(size=13)),
        yaxis=dict(showgrid=True, gridcolor=GRID_CLR, zeroline=False, tickfont=dict(size=13)),
        legend=dict(bgcolor="rgba(0,0,0,0)", borderwidth=0, font=dict(size=13)),
        margin=dict(l=55, r=extra_margin_right, t=70, b=55),
    )
    if height is not None:
        fig.update_layout(height=height)
    return fig

px.defaults.template = "plotly_dark"

# =========================================================
# SAMPLE DATA TEMPLATE (download button)
# =========================================================
def build_sample_template(scale: int) -> bytes:
    """
    Creates a simple sample dataset for download.
    Uses values within the selected Likert scale.
    """
    sample = pd.DataFrame({
        "Item_ID": ["Item_1", "Item_2", "Item_3", "Item_4", "Item_5"],
        "Expert_1": [scale, scale-1, max(1, scale-2), scale, scale-1],
        "Expert_2": [scale-1, scale-1, max(1, scale-3), scale-1, scale-2],
        "Expert_3": [scale, scale, max(1, scale-2), scale-1, scale-1],
        "Expert_4": [scale-1, scale, max(1, scale-3), scale, scale-2],
    })
    buf = BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as w:
        sample.to_excel(w, index=False, sheet_name="Sample")
    return buf.getvalue()

# =========================================================
# TFN MAPS (ADOPTED)
# =========================================================
TFN_MAP = {
    5: {1:(0,0,0.25), 2:(0,0.25,0.5), 3:(0.25,0.5,0.75), 4:(0.5,0.75,1), 5:(0.75,1,1)},
    7: {1:(0,0,0.1), 2:(0,0.1,0.3), 3:(0.1,0.3,0.5), 4:(0.3,0.5,0.75), 5:(0.5,0.75,0.9), 6:(0.75,0.9,1), 7:(0.9,1,1)},
    9: {1:(0,0.1,0.2), 2:(0.1,0.2,0.3), 3:(0.2,0.3,0.4), 4:(0.3,0.4,0.5), 5:(0.4,0.5,0.6),
        6:(0.5,0.6,0.7), 7:(0.6,0.7,0.8), 8:(0.7,0.8,0.9), 9:(0.8,0.9,1)}
}

# =========================================================
# HELPERS — DATA + MATH
# =========================================================
def dataset_fingerprint(df: pd.DataFrame) -> str:
    return hashlib.sha256(df.to_csv(index=False).encode()).hexdigest()[:16]

def fuzzify(x, scale: int):
    if pd.isna(x):
        raise ValueError("Missing value (blank/NaN).")
    try:
        val = int(float(str(x).strip()))
    except Exception:
        raise ValueError(f"Invalid rating '{x}' (must be numeric).")
    if val < 1 or val > scale:
        raise ValueError(f"Rating {val} out of range for {scale}-point scale.")
    return TFN_MAP[scale][val]

def di(tfn, g):
    return float(np.mean(np.abs(np.array(tfn) - np.array(g))))

def defuzz(g):
    return float((g[0] + 2*g[1] + g[2]) / 4)

# =========================================================
# HELPERS — PAYPAL REST (OAuth2 + Orders + Capture)
# =========================================================
def paypal_get_access_token() -> str:
    if not PAYPAL_CLIENT_ID or not PAYPAL_SECRET:
        raise RuntimeError("Missing PAYPAL_CLIENT_ID / PAYPAL_SECRET environment variables.")
    url = f"{PAYPAL_API_BASE}/v1/oauth2/token"
    r = requests.post(
        url,
        headers={"Accept": "application/json", "Accept-Language": "en_US"},
        data={"grant_type": "client_credentials"},
        auth=(PAYPAL_CLIENT_ID, PAYPAL_SECRET),
        timeout=30,
    )
    if r.status_code != 200:
        raise RuntimeError(f"PayPal OAuth error {r.status_code}: {r.text}")
    return r.json()["access_token"]

def paypal_create_order(amount_usd: float, project_id: str, product_name: str):
    token = paypal_get_access_token()
    url = f"{PAYPAL_API_BASE}/v2/checkout/orders"
    payload = {
        "intent": "CAPTURE",
        "purchase_units": [{
            "reference_id": project_id,
            "description": product_name,
            "amount": {"currency_code": "USD", "value": f"{amount_usd:.2f}"}
        }],
        "application_context": {
            "return_url": f"{APP_BASE_URL}/?paypal_success=1&pid={project_id}",
            "cancel_url": f"{APP_BASE_URL}/?paypal_cancel=1&pid={project_id}",
            "shipping_preference": "NO_SHIPPING",
            "user_action": "PAY_NOW",
        }
    }
    r = requests.post(
        url,
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {token}"},
        json=payload,
        timeout=30,
    )
    if r.status_code not in (201, 200):
        raise RuntimeError(f"PayPal create order error {r.status_code}: {r.text}")
    data = r.json()
    order_id = data["id"]
    approve_url = next((lnk["href"] for lnk in data.get("links", []) if lnk.get("rel") == "approve"), None)
    if not approve_url:
        raise RuntimeError("PayPal approve URL not returned.")
    return order_id, approve_url

def paypal_get_order_status(order_id: str) -> str:
    token = paypal_get_access_token()
    url = f"{PAYPAL_API_BASE}/v2/checkout/orders/{order_id}"

    r = requests.get(
        url,
        headers={
            "Accept": "application/json",
            "Content-Type": "application/json",
            "Authorization": f"Bearer {token}",
        },
        timeout=30,
    )
    if r.status_code != 200:
        raise RuntimeError(f"PayPal get order error {r.status_code}: {r.text}")
    return r.json().get("status", "")


def paypal_capture_order(order_id: str) -> bool:
    # already processed in session
    if order_id in st.session_state.paid_orders:
        return True

    # if already completed at PayPal, accept it
    try:
        if paypal_get_order_status(order_id) == "COMPLETED":
            st.session_state.paid_orders.add(order_id)
            return True
    except Exception:
        pass

    token = paypal_get_access_token()
    url = f"{PAYPAL_API_BASE}/v2/checkout/orders/{order_id}/capture"

    headers = {
        "Accept": "application/json",
        "Content-Type": "application/json",
        "Authorization": f"Bearer {token}",
        "Prefer": "return=representation",
    }

    # IMPORTANT: send an explicit JSON body (some PayPal setups reject empty body)
    r = requests.post(url, headers=headers, data="{}", timeout=30)

    # Handle "already captured" safely
    if r.status_code == 422 and "ORDER_ALREADY_CAPTURED" in r.text:
        st.session_state.paid_orders.add(order_id)
        return True

    if r.status_code not in (200, 201):
        raise RuntimeError(f"PayPal capture error {r.status_code}: {r.text}")

    ok = (r.json().get("status") == "COMPLETED")
    if ok:
        st.session_state.paid_orders.add(order_id)
    return ok

# =========================================================
# SESSION
# =========================================================
if "unlocked" not in st.session_state:
    st.session_state.unlocked = False
if "unlocked_pid" not in st.session_state:
    st.session_state.unlocked_pid = None
if "paid_orders" not in st.session_state:
    st.session_state.paid_orders = set()


# =========================================================
# HEADER WITH LOGO
# =========================================================
col_logo, col_title = st.columns([1, 6])

with col_logo:
    st.image("Logo.png", width=90)

with col_title:
    st.markdown(
        """
        <div style="
            font-size: 2.2rem;
            font-weight: 800;
            margin-top: 10px;
            color: #F3F7FF;
            text-shadow: 0 1px 14px rgba(30,120,255,0.25);
        ">
            Fuzzy Delphi Analyzer
        </div>
        """,
        unsafe_allow_html=True,
    )

mode_badge = "🟡 Sandbox" if PAYPAL_MODE != "live" else "🟢 Live"
st.markdown(
    f'<span class="pill">{mode_badge} • Pay-per-project unlock: <b>USD {PRICE_USD:.0f}</b></span>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="small-muted">Step 1: Select Likert scale → Step 2: Download sample → Step 3: Upload dataset → Step 4: Unlock export</div>',
    unsafe_allow_html=True
)

st.markdown('<div class="hr"></div>', unsafe_allow_html=True)


# =========================================================
# SIDEBAR
# =========================================================
st.sidebar.markdown("### Step 1 — Likert scale")
scale = st.sidebar.radio("Likert scale", [5, 7, 9], horizontal=True, index=1)

st.sidebar.markdown("### Step 2 — Thresholds")
score_cut = st.sidebar.slider("Score cut-off  S ≥", 0.5, 1.0, 0.70, 0.01)
di_thr = st.sidebar.slider("Agreement threshold  dᵢ ≤", 0.05, 0.5, 0.20, 0.01)
gc_cut = st.sidebar.slider("Group consensus  GC% ≥", 50, 100, 70)

st.sidebar.markdown("---")
st.sidebar.markdown("### Payment")
st.sidebar.markdown(f"**Fixed price:** USD {PRICE_USD:.0f}")
if not PAYPAL_CLIENT_ID or not PAYPAL_SECRET:
    st.sidebar.warning("PayPal credentials missing.\nSet PAYPAL_CLIENT_ID and PAYPAL_SECRET.\nUnlock will not work until set.")

# =========================================================
# SAMPLE DOWNLOAD BUTTON
# =========================================================
st.markdown("### Sample dataset (optional)")
st.markdown('<div class="small-muted">Download a ready-to-use template with <b>Item_ID</b> and <b>Expert_*</b> columns.</div>', unsafe_allow_html=True)
sample_bytes = build_sample_template(scale)
st.download_button(
    "⬇️ Download Sample Excel",
    data=sample_bytes,
    file_name=f"fuzzy_delphi_sample_scale_{scale}.xlsx",
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
)

st.markdown('<div class="hr"></div>', unsafe_allow_html=True)

# =========================================================
# AUTO-VERIFY AFTER PAYPAL REDIRECT (BEFORE upload)
# =========================================================
q = st.query_params
if q.get("paypal_success") == "1" and "token" in q and "pid" in q:
    order_id = q["token"]
    pid_from_url = q["pid"]
    try:
        if paypal_capture_order(order_id):
            st.session_state.unlocked = True
            st.session_state.unlocked_pid = pid_from_url
            st.success("Payment verified ✅ Upload the SAME dataset used during payment to unlock full export.")
            st.query_params.clear()
    except Exception as e:
        st.error(f"PayPal verification error: {e}")

if q.get("paypal_cancel") == "1":
    st.info("Payment cancelled. You can try again anytime.")
    st.query_params.clear()

# =========================================================
# UPLOAD
# =========================================================
st.markdown("### Upload Dataset")
st.markdown('<div class="small-muted">Required columns: <b>Item_ID</b> + expert columns like <b>Expert_1, Expert_2, ...</b> (numeric ratings).</div>', unsafe_allow_html=True)

file = st.file_uploader("Upload CSV / Excel", ["csv", "xlsx"])
if not file:
    st.stop()

df = pd.read_csv(file) if file.name.endswith(".csv") else pd.read_excel(file)

experts = [c for c in df.columns if c.lower().startswith("expert")]

if "Item_ID" not in df.columns:
    st.error("Missing required column: Item_ID")
    st.stop()
if len(experts) == 0:
    st.error("No expert columns found. Expected columns like Expert_1, Expert_2, ...")
    st.stop()

pid = dataset_fingerprint(df)

# Payment is tied to dataset PID
if st.session_state.unlocked and st.session_state.unlocked_pid:
    if pid != st.session_state.unlocked_pid:
        st.session_state.unlocked = False
        st.warning("This payment is locked to a different dataset. Upload the paid dataset or unlock again.")

# =========================================================
# VALIDATE RATINGS (show exact bad cells)
# =========================================================
bad_cells = []
for idx, row in df.iterrows():
    item_id = row.get("Item_ID", f"Row{idx+1}")
    for e in experts:
        try:
            _ = fuzzify(row[e], scale)
        except Exception as ex:
            bad_cells.append((item_id, e, row[e], str(ex)))

if bad_cells:
    st.error("❌ Dataset contains invalid ratings. Fix these cells and re-upload:")
    st.dataframe(pd.DataFrame(bad_cells, columns=["Item_ID", "Column", "Value", "Error"]), use_container_width=True)
    st.stop()

# =========================================================
# ANALYSIS
# =========================================================
rows = []
for _, r in df.iterrows():
    tfns = [fuzzify(r[e], scale) for e in experts]
    g = tuple(np.mean(tfns, axis=0))
    di_vals = [di(t, g) for t in tfns]
    dbar = float(np.mean(di_vals))
    score = defuzz(g)
    agree = int(sum(dv <= di_thr for dv in di_vals))
    gc = float(100 * agree / len(di_vals))
    decision = "Retain" if (score >= score_cut and dbar <= di_thr and gc >= gc_cut) else "Drop"

    rows.append({
        "Item": r["Item_ID"],
        "m1": round(g[0], 3),
        "m2": round(g[1], 3),
        "m3": round(g[2], 3),
        "Score": round(score, 3),
        "d̄": round(dbar, 3),
        "GC %": round(gc, 1),
        "Agreed": f"{agree}/{len(di_vals)}",
        "Decision": decision,
    })

res = pd.DataFrame(rows)
display_res = res if st.session_state.unlocked else res.head(FREE_LIMIT)

# =========================================================
# DASHBOARD
# =========================================================
total_items = len(res)
retained = int((res["Decision"] == "Retain").sum())
dropped = total_items - retained
avg_score = float(res["Score"].mean()) if total_items else 0.0
avg_gc = float(res["GC %"].mean()) if total_items else 0.0
avg_dbar = float(res["d̄"].mean()) if total_items else 0.0

k1, k2, k3, k4, k5 = st.columns(5)
cards = [
    ("Total Items", f"{total_items}", "Items analyzed"),
    ("Retained", f"{retained}", f"Criteria: S≥{score_cut:.2f}, d̄≤{di_thr:.2f}, GC≥{gc_cut}%"),
    ("Dropped", f"{dropped}", "Did not meet thresholds"),
    ("Avg Score (S)", f"{avg_score:.3f}", "Across all items"),
    ("Avg Consensus", f"{avg_gc:.1f}%", "Across all items"),
]
for col, (t, v, s) in zip([k1, k2, k3, k4, k5], cards):
    col.markdown(
        f"""
        <div class="kpi">
          <div class="t">{t}</div>
          <div class="v">{v}</div>
          <div class="s">{s}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.markdown('<div class="hr"></div>', unsafe_allow_html=True)
st.subheader("Results Table")
st.dataframe(display_res, use_container_width=True)

# =========================================================
# VISUAL ANALYTICS (IMPROVED: Donut + Gauge + Scatter)
# =========================================================
st.markdown("### Visual Analytics")

c1, c2 = st.columns(2)

# ---------- 1) DONUT (NO OVERLAP: legend bottom + labels inside) ----------
with c1:
    st.markdown("**Screening Outcome Composition**")

    donut_counts = {"Retain": retained, "Drop": dropped}

    fig_donut = go.Figure(
        data=[
            go.Pie(
                labels=list(donut_counts.keys()),
                values=list(donut_counts.values()),
                hole=0.65,
                sort=False,
                marker=dict(
                    colors=[GOOD, BAD],
                    line=dict(color="rgba(255,255,255,0.18)", width=1),
                ),

                # ✅ Put slice text INSIDE to avoid collisions
                textinfo="label+percent",
                textposition="inside",
                insidetextorientation="radial",
                textfont=dict(size=14, color=FONT_CLR),

                hovertemplate="<b>%{label}</b><br>Count: %{value}<br>Share: %{percent}<extra></extra>",
            )
        ]
    )

    # Center annotation (Total)
    fig_donut.update_layout(
        height=430,
        margin=dict(l=35, r=35, t=25, b=55),
        paper_bgcolor="rgba(0,0,0,0)",

        # ✅ Legend moved to bottom (no overlap with chart labels)
        showlegend=True,
        legend=dict(
            title="Decision",
            orientation="h",
            yanchor="top",
            y=-0.05,
            xanchor="center",
            x=0.5,
            font=dict(size=13, color=FONT_CLR),
        ),

        annotations=[
            dict(
                text=f"<b>Total</b><br>{retained + dropped}",
                x=0.5,
                y=0.5,
                font=dict(size=18, color=FONT_CLR),
                showarrow=False,
            )
        ],
    )

    c1.plotly_chart(fig_donut, use_container_width=True, config={"displayModeBar": False})

# ---------- 2) GAUGE (more detail + balanced) ----------
with c2:
    st.markdown("**Average Distance (d̄) Quality Gauge**")

    # Range: always show up to at least 0.5, and give headroom
    max_range = max(0.5, float(di_thr) * 2.5)

    # Define zones
    good_end = float(di_thr)
    caution_end = min(max_range, float(di_thr) * 1.5)

    fig_gauge = go.Figure(
        go.Indicator(
            mode="gauge+number+delta",
            value=avg_dbar,
            number={"font": {"size": 54, "color": FONT_CLR}},
            delta={
                "reference": float(di_thr),
                "valueformat": ".3f",
                "increasing": {"color": BAD},
                "decreasing": {"color": GOOD},
                "font": {"size": 16, "color": FONT_CLR},
            },
            gauge={
                "shape": "angular",
                "axis": {
                    "range": [0, max_range],
                    "tickwidth": 1,
                    "tickcolor": "rgba(255,255,255,0.35)",
                    "tickfont": {"size": 13, "color": FONT_CLR},
                    "tickmode": "auto",
                    "nticks": 6,
                },
                "bar": {"color": ACCENT, "thickness": 0.28},
                "bgcolor": "rgba(0,0,0,0)",
                "borderwidth": 0,
                "steps": [
                    {"range": [0, good_end], "color": "rgba(34,197,94,0.25)"},         # Good
                    {"range": [good_end, caution_end], "color": "rgba(245,158,11,0.20)"}, # Caution
                    {"range": [caution_end, max_range], "color": "rgba(239,68,68,0.18)"}, # Poor
                ],
                "threshold": {
                    "line": {"color": WARN, "width": 5},
                    "thickness": 0.82,
                    "value": float(di_thr),
                },
            },
        )
    )

    fig_gauge.update_layout(
        height=430,
        margin=dict(l=80, r=120, t=35, b=35),  # keeps ticks inside + balanced
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        annotations=[
            dict(
                x=0.5, y=0.02, xref="paper", yref="paper",
                text=f"Threshold: dᵢ ≤ {di_thr:.3f}  •  Good / Caution / Poor zones",
                showarrow=False,
                font=dict(size=13, color="rgba(234,240,255,0.85)"),
            )
        ],
    )

    c2.plotly_chart(fig_gauge, use_container_width=True, config={"displayModeBar": False})


# ---------- 3) SCATTER (clean labels + modebar back + hide items) ----------
fig_scatter = px.scatter(
    res,
    x="Score",
    y="d̄",
    color="Decision",
    custom_data=["GC %", "Agreed"],  # ✅ no Item in hover data
    title=None,
    color_discrete_map={"Retain": GOOD, "Drop": BAD},
)

# ✅ Hover: no item shown
fig_scatter.update_traces(
    marker=dict(size=11, opacity=0.95),
    hovertemplate=(
        "Score: %{x:.3f}<br>"
        "Average distance (d̄): %{y:.3f}<br>"
        "Consensus (GC%): %{customdata[0]:.1f}%<br>"
        "Agreement: %{customdata[1]}<br>"
        "<extra></extra>"
    ),
)

# Lines
fig_scatter.add_vline(x=score_cut, line_width=2, line_dash="dash", line_color="#a78bfa")
fig_scatter.add_hline(y=di_thr,   line_width=2, line_dash="dash", line_color=ACCENT)

# ✅ Axes ranges (needed to position annotations safely)
x_min = float(res["Score"].min()) if len(res) else 0.5
x_max = float(res["Score"].max()) if len(res) else 1.0
x_pad = max(0.02, (x_max - x_min) * 0.08)

y_min = float(res["d̄"].min()) if len(res) else 0.0
y_max = float(res["d̄"].max()) if len(res) else 0.5
y_pad = max(0.02, (y_max - y_min) * 0.10)

fig_scatter.update_xaxes(range=[x_min - x_pad, x_max + x_pad])
fig_scatter.update_yaxes(range=[max(0, y_min - y_pad), y_max + y_pad])

# ✅ Annotation positions:
# - Score label near top of the vertical line
# - Distance label at RIGHT end of the horizontal line
x_right = (x_max + x_pad)  # far right edge
y_top   = (y_max + y_pad)  # top edge

fig_scatter.add_annotation(
    x=score_cut,
    y=y_top,
    text="Score threshold",
    showarrow=False,
    yanchor="bottom",
    xanchor="center",
    font=dict(size=12, color=FONT_CLR),
    bgcolor="rgba(8,12,20,0.60)",
    bordercolor="rgba(255,255,255,0.18)",
    borderwidth=1,
    borderpad=4,
)

fig_scatter.add_annotation(
    x=x_right,
    y=di_thr,
    text="Distance threshold",
    showarrow=False,
    xanchor="right",     # ✅ pinned to the right of the plot
    yanchor="middle",
    font=dict(size=12, color=FONT_CLR),
    bgcolor="rgba(8,12,20,0.60)",
    bordercolor="rgba(255,255,255,0.18)",
    borderwidth=1,
    borderpad=4,
)
fig_scatter.update_layout(title_text="", title=None)
# Layout polish + extra right margin so legend text never clips
fig_scatter.update_layout(
    height=520,
    margin=dict(l=60, r=120, t=20, b=55),  # ✅ more right margin for legend
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor=PLOT_BG,
    font=dict(color=FONT_CLR, size=14),
    xaxis=dict(title="Score", gridcolor="rgba(255,255,255,0.12)", zeroline=False),
    yaxis=dict(title="Average distance (d̄)", gridcolor="rgba(255,255,255,0.12)", zeroline=False),
    legend=dict(
        title="Decision",
        bgcolor="rgba(8,12,20,0.55)",
        bordercolor="rgba(255,255,255,0.18)",
        borderwidth=1,
        font=dict(size=13, color=FONT_CLR),
    ),
)

st.markdown("**Score vs Distance (Item Quality Map)**")

# ✅ Bring back Plotly controls (modebar)
st.plotly_chart(
    style_fig(fig_scatter, height=520, extra_margin_right=120),
    use_container_width=True,
    config={
        "displayModeBar": True,          # ✅ show controls
        "displaylogo": False,
        "scrollZoom": True              # ✅ nice UX
    },
)
# =========================================================
# WHAT CLIENT GETS AFTER UNLOCK (Readable Glass Card)
# =========================================================
st.markdown(
    """
<div style="
  background: rgba(8, 12, 20, 0.72);
  border: 1px solid rgba(255,255,255,0.16);
  border-radius: 18px;
  padding: 18px;
  backdrop-filter: blur(10px);
  margin-top: 18px;
">

  <div style="
    font-size: 1.25rem;
    font-weight: 800;
    color: #F3F7FF;
    margin-bottom: 10px;
  ">
    📦 What you will receive after unlocking
  </div>

  <div style="color: #EAF0FF; font-size: 1.02rem; line-height: 1.55;">
    After unlocking, you can download a <b>publication-ready Excel file</b> that includes:
    <ul style="margin-top: 10px;">
      <li><b>Aggregated fuzzy numbers:</b> m̄1, m̄2, m̄3</li>
      <li><b>Defuzzified score:</b> S = (m1 + 2·m2 + m3) / 4</li>
      <li><b>Average distance:</b> d̄</li>
      <li><b>Group consensus:</b> GC% = 100 × (n_agreed / n_experts)</li>
      <li><b>Agreement count:</b> n_agreed / n_experts</li>
      <li><b>Final decision:</b> Retain / Drop</li>
    </ul>
  </div>

  <div style="
    margin-top: 12px;
    background: rgba(255,255,255,0.92);
    border-radius: 12px;
    padding: 12px 14px;
    color: #111827;
    font-weight: 600;
  ">
    ✅ The Excel output is ready for theses, journals, and Delphi methodology reporting.
  </div>

</div>
""",
    unsafe_allow_html=True,
)

# =========================================================
# PAYMENT
# =========================================================
if not st.session_state.unlocked:
    st.warning(f"Free preview is limited to {FREE_LIMIT} item(s). Full export requires a one-time unlock (USD {PRICE_USD:.0f}).")

    colp1, colp2 = st.columns([1, 2])
    with colp1:
        if st.button(f"🔓 Unlock full export (USD {PRICE_USD:.0f})", type="primary"):
            try:
                order_id, approve_url = paypal_create_order(
                    amount_usd=PRICE_USD,
                    project_id=pid,
                    product_name="Fuzzy Delphi Full Analysis",
                )
                st.markdown(f"[✅ Proceed to PayPal Checkout]({approve_url})")
            except Exception as e:
                st.error(f"PayPal error: {e}")

    with colp2:
        st.info("After payment, PayPal redirects back automatically and the export unlocks instantly for the paid dataset.")

# =========================================================
# EXPORT
# =========================================================
if st.session_state.unlocked:
    st.success("Export is unlocked ✅")
    buf = BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as w:
        res.to_excel(w, index=False, sheet_name="Results")
    st.download_button("⬇️ Download Excel (Full Results)", buf.getvalue(), "fuzzy_delphi_results.xlsx")

# =========================================================
# FOOTER
# =========================================================
st.markdown('<div class="hr"></div>', unsafe_allow_html=True)
st.markdown(
    f"""
<div class="small-muted">
<b>If you require any assistance, consultation, or information please contact:</b><br>
<b>{OWNER['name']}</b><br>
📧 {OWNER['email']} • 📞 {OWNER['phone']} • <a href="{OWNER['whatsapp']}" target="_blank">💬 WhatsApp</a>
</div>
""",
    unsafe_allow_html=True,
)


