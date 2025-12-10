from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

# -------------------------------------------------------------
# Page config
# -------------------------------------------------------------
st.set_page_config(
    page_title="Singapore HDB Resale Price – Interactive Model",
    page_icon="🏠",
    layout="centered",
)

st.title("🏠 Singapore HDB Resale Price — Interactive Model")
st.caption(
    "Estimated resale price based on a trained XGBoost model. "
    "Actual market prices can differ due to flat condition and other factors."
)

# -------------------------------------------------------------
# Paths / model loading
# -------------------------------------------------------------
PIPELINE_PATH = Path(__file__).resolve().parent / "pipeline.pkl"


@st.cache_resource(show_spinner=True)
def load_model():
    if not PIPELINE_PATH.exists():
        st.error(f"Model file not found at: {PIPELINE_PATH}")
        return None

    try:
        model = joblib.load(PIPELINE_PATH)
        return model
    except Exception as e:
        st.error(f"Failed to load model: {e}")
        return None


model = load_model()

if model is not None:
    st.success("Trained model loaded successfully.")
else:
    st.stop()

# -------------------------------------------------------------
# Helper data & functions
# -------------------------------------------------------------

# We keep the same 10 features that were used in train_model.py
FEATURE_COLS = [
    "floor_area_sqft",
    "years_of_lease_left",
    "mrt_nearest_distance",
    "Mall_Within_500m",
    "Hawker_Within_2km",
    "commercial",
    "mrt_interchange",
    "flat_type",
    "floor_level_range",
    "mrt_region",
]

# MRT "station" options – grouped by region, but still showing example station names
MRT_STATIONS = {
    "Central (e.g. Bishan, Toa Payoh, City Hall, Orchard, Dhoby Ghaut)": {"region": "Central"},
    "North (e.g. Yishun, Woodlands, Sembawang)": {"region": "North"},
    "East (e.g. Bedok, Tampines, Pasir Ris, Paya Lebar, Geylang)": {"region": "East"},
    "North-East (e.g. Hougang, Punggol, Sengkang, Serangoon, Kovan)": {"region": "North-East"},
    "West (e.g. Jurong East, Jurong West, Clementi, Boon Lay, Bukit Batok, Bukit Panjang)": {"region": "West"},
}


def compute_floor_level_range(current_floor: int, max_floor_lvl: int) -> str:
    """
    Recreate the same logic you used in the notebook:

    if mid_storey > (max_floor_lvl * 2/3):         'Upper Level'
    elif (mid_storey > max_floor_lvl/3) and
         (mid_storey <= max_floor_lvl * 2/3):      'Mid Upper Level'
    else:                                          'Mid Lower Level'
    """
    if max_floor_lvl <= 0:
        return "Mid Lower Level"

    ratio = current_floor / max_floor_lvl

    if ratio > 2 / 3:
        return "Upper Level"
    elif (ratio > 1 / 3) and (ratio <= 2 / 3):
        return "Mid Upper Level"
    else:
        return "Mid Lower Level"


def build_model_input(
    *,
    floor_area_sqft: float,
    years_of_lease_left: float,
    mrt_nearest_distance: float,
    mall_within_500m: int,
    hawker_within_2km: int,
    commercial: bool,
    mrt_interchange: bool,
    flat_type: str,
    floor_level_range: str,
    mrt_region: str,
) -> pd.DataFrame:
    """Create a single-row DataFrame with exactly the columns the pipeline expects."""

    row = {
        "floor_area_sqft": float(floor_area_sqft),
        "years_of_lease_left": float(years_of_lease_left),
        "mrt_nearest_distance": float(mrt_nearest_distance),
        "Mall_Within_500m": int(mall_within_500m),
        "Hawker_Within_2km": int(hawker_within_2km),
        "commercial": float(1 if commercial else 0),
        "mrt_interchange": float(1 if mrt_interchange else 0),
        "flat_type": flat_type,
        "floor_level_range": floor_level_range,
        "mrt_region": mrt_region,
    }

    df = pd.DataFrame([row], columns=FEATURE_COLS)
    return df


# -------------------------------------------------------------
# UI – form layout
# -------------------------------------------------------------
st.markdown("### Enter property details")

with st.form("price_form"):
    col_left, col_right = st.columns(2)

    with col_left:
        floor_area_sqft = st.number_input(
            "Floor area (sqft)",
            min_value=300.0,
            max_value=2500.0,   # was 2000.0 but some units are larger hence changed to 2500
            value=1000.0,
            step=10.0,
        )

        years_of_lease_left = st.number_input(
            "Years of lease left",
            min_value=1.0,
            max_value=99.0,
            value=70.0,
            step=1.0,
        )

        mrt_nearest_distance = st.number_input(
            "Nearest MRT distance (m)",
            min_value=50.0,
            max_value=3000.0,
            value=600.0,
            step=50.0,
            help="Approximate walking distance to the closest MRT station.",
        )

        mall_within_500m = st.number_input(
            "Shopping malls within 2km (count)",
            min_value=0,
            max_value=5,
            value=1,
            step=1,
        )

        hawker_within_2km = st.number_input(
            "Hawker centres within 2km (count)",
            min_value=0,
            max_value=5,
            value=2,
            step=1,
        )

    with col_right:
        flat_type = st.selectbox(
            "Flat type",
            ["2 ROOM", "3 ROOM", "4 ROOM", "5 ROOM", "EXECUTIVE"],
            index=2,
        )

        max_floor_lvl = st.number_input(
            "Max floor level of block",
            min_value=1,
            max_value=60,
            value=15,
            step=1,
            help="Highest floor of the block (from HDB info).",
        )

        current_floor = st.number_input(
            "Current floor level",
            min_value=1,
            max_value=60,
            value=5,
            step=1,
            help="The floor level of the specific unit.",
        )

        mrt_station_label = st.selectbox(
            "Nearest MRT station (by region with examples)",
            list(MRT_STATIONS.keys()),
            index=0,
            help="Pick the region that best matches your nearest station.",
        )
        mrt_region = MRT_STATIONS[mrt_station_label]["region"]

        commercial = st.checkbox(
            "Commercial-use flat (e.g. above shops/offices)?",
            value=False,
        )

        mrt_interchange = st.checkbox(
            "Nearest station is an interchange?",
            value=False,
        )

    submitted = st.form_submit_button("Predict Resale Price")

# -------------------------------------------------------------
# Prediction
# -------------------------------------------------------------
if submitted:
    if current_floor > max_floor_lvl:
        st.error("Current floor level cannot be higher than the block's max floor level.")
    else:
        floor_level_range = compute_floor_level_range(
            current_floor=int(current_floor),
            max_floor_lvl=int(max_floor_lvl),
        )

        ui_df = build_model_input(
            floor_area_sqft=floor_area_sqft,
            years_of_lease_left=years_of_lease_left,
            mrt_nearest_distance=mrt_nearest_distance,
            mall_within_500m=mall_within_500m,
            hawker_within_2km=hawker_within_2km,
            commercial=commercial,
            mrt_interchange=mrt_interchange,
            flat_type=flat_type,
            floor_level_range=floor_level_range,
            mrt_region=mrt_region,
        )

        with st.expander("🔍 Debug: model input row"):
            st.write(ui_df)

        try:
            pred_price = float(model.predict(ui_df)[0])

            # Big, coloured price card
            st.markdown(
                f"""
                <div style="
                    margin-top: 1rem;
                    padding: 1.2rem;
                    border-radius: 0.75rem;
                    background: linear-gradient(90deg, #065f46, #16a34a);
                    text-align: center;
                    border: 1px solid #bbf7d0;
                ">
                    <div style="font-size: 1rem; color: #dcfce7; letter-spacing: 0.03em;">
                        Estimated Resale Price
                    </div>
                    <div style="font-size: 2.1rem; font-weight: 700; color: #f0fdf4; margin-top: 0.25rem;">
                        S${pred_price:,.0f}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            st.caption("Prediction generated by the trained `pipeline.pkl` model.")
            st.caption(
                "Note: price is an estimate based on historical data – "
                "actual resale price depends on negotiation and flat condition."
            )

        except Exception as e:
            st.error(f"Prediction failed: {e}")

