"""FinSight — Financial Offer Experimentation & Risk-Aware Customer Intelligence.

Streamlit entry point. Run with:
    streamlit run app/streamlit_app.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

st.set_page_config(
    page_title="FinSight",
    page_icon="\U0001F4CA",
    layout="wide",
    initial_sidebar_state="expanded",
)

CUSTOM_CSS = """
<style>
    :root {
        --fs-navy: #0F172A;
        --fs-slate: #475569;
        --fs-accent: #2563EB;
        --fs-accent-light: #DBEAFE;
        --fs-bg: #F8FAFC;
    }
    .stApp { background-color: var(--fs-bg); }
    h1, h2, h3 { color: var(--fs-navy); font-weight: 700; }
    [data-testid="stMetricValue"] { color: var(--fs-navy); font-weight: 700; }
    [data-testid="stMetricLabel"] { color: var(--fs-slate); }
    .fs-tagline { color: var(--fs-slate); font-size: 1.05rem; margin-top: -0.6rem; }
    .fs-disclaimer {
        background-color: #FEF3C7; border-left: 4px solid #D97706;
        padding: 0.75rem 1rem; border-radius: 4px; font-size: 0.88rem; color: #78350F;
    }
    .fs-section-note {
        background-color: var(--fs-accent-light); border-left: 4px solid var(--fs-accent);
        padding: 0.75rem 1rem; border-radius: 4px; font-size: 0.92rem; color: var(--fs-navy);
    }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


def main() -> None:
    st.sidebar.title("\U0001F4CA FinSight")
    st.sidebar.caption("Financial Offer Experimentation & Risk-Aware Customer Intelligence")

    page = st.sidebar.radio(
        "Navigate",
        [
            "1. Executive Overview",
            "2. Experiment & Statistics",
            "3. Customer & ML Insights",
            "4. Targeting & Business Simulator",
        ],
        label_visibility="collapsed",
    )

    st.sidebar.markdown("---")
    st.sidebar.markdown(
        "<div class='fs-disclaimer'>This project uses public/simulated data for "
        "educational and portfolio purposes and is not intended for real-world "
        "credit or financial decision-making.</div>",
        unsafe_allow_html=True,
    )
    st.sidebar.markdown("---")
    st.sidebar.markdown(
        "[📂 View source on GitHub](https://github.com/mansik-11/finSight)  \n"
        "[👤 LinkedIn](https://linkedin.com/in/mansi-kaushik)"
    )
    st.sidebar.caption("Built with Python, DuckDB, scikit-learn, XGBoost, SHAP, MLflow & Streamlit.")

    try:
        if page.startswith("1"):
            from app.views import page_1_executive_overview as p
        elif page.startswith("2"):
            from app.views import page_2_experiment_statistics as p
        elif page.startswith("3"):
            from app.views import page_3_customer_ml_insights as p
        else:
            from app.views import page_4_targeting_simulator as p
        p.render()
    except FileNotFoundError as e:
        st.error(
            f"**Required data or model artifacts are missing.**\n\n{e}\n\n"
            "Run the pipeline first:\n\n"
            "```bash\nmake data\nmake train\nmake evaluate\n```"
        )
    except Exception as e:  # pragma: no cover - defensive UI guard
        st.error(f"Something went wrong rendering this page: {e}")
        with st.expander("Technical details"):
            st.exception(e)


if __name__ == "__main__":
    main()
