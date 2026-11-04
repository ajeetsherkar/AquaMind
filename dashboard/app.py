
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st
import pandas as pd

from core.forecast import ForecastConfig
from core.pipeline import run_pipeline
from core.simulator import VALID_SCENARIOS


st.set_page_config(
    page_title="AquaMind | RO Monitoring",
    page_icon="💧",
    layout="wide",
)

# Permanent simulation disclaimer
st.error(
    "⚠️ SIMULATED DATA — This dashboard uses synthetic data. "
    "It is not connected to a physical water purifier."
)

st.title("💧 AquaMind")
st.caption("Water Purifier Prognostics and Health Monitoring")

SCENARIO_LABELS = {
    "healthy": "Healthy Operation",
    "prefilter_restriction": "Prefilter Restriction",
    "membrane_performance_loss": "Membrane Performance Loss",
    "feed_tds_spike": "Feed-TDS Spike",
    "stuck_sensor": "Stuck Sensor",
}

with st.sidebar:
    st.header("Simulation Controls")

    scenario = st.selectbox(
        "Select scenario",
        options=list(VALID_SCENARIOS),
        format_func=lambda value: SCENARIO_LABELS[value],
    )

    duration_seconds = st.slider(
        "Simulation duration (seconds)",
        min_value=60,
        max_value=600,
        value=300,
        step=60,
    )

    speed = st.select_slider(
        "Simulation speed / sampling resolution",
        options=["Slow", "Normal", "Fast"],
        value="Normal",
    )

    sampling_interval = {
        "Slow": 2.0,
        "Normal": 1.0,
        "Fast": 0.5,
    }[speed]

    seed = st.number_input(
        "Random seed",
        min_value=0,
        max_value=100000,
        value=42,
        step=1,
    )

    run_button = st.button(
        "▶ Run Simulation",
        type="primary",
        use_container_width=True,
    )

    st.caption(
        "Higher speed creates more samples per simulated second. "
        "This is a data-generation control, not real-time hardware playback."
    )

if run_button:
    with st.spinner("Running AquaMind pipeline..."):
        result = run_pipeline(
            scenario=scenario,
            duration_seconds=duration_seconds,
            sampling_interval=sampling_interval,
            seed=int(seed),
            forecast_config=ForecastConfig(),
        )

    st.session_state["aqu amind_result".replace(" ", "")] = result

result = st.session_state.get("aqu amind_result".replace(" ", ""))

if result is None:
    st.info(
        "Choose a scenario and click **Run Simulation** to view the "
        "monitoring results."
    )
    st.stop()

data = result["data"].copy()
forecast = result["forecast"]
validation = result["validation"]

# Prepare timestamps and display values
data["timestamp"] = pd.to_datetime(
    data["timestamp"],
    errors="coerce",
    utc=True,
)

st.subheader(f"Scenario: {SCENARIO_LABELS[result['scenario']]}")

status = result["status"]
if status == "Healthy":
    st.success(f"Sensor data status: {status}")
elif status == "Degraded":
    st.warning(f"Sensor data status: {status}")
else:
    st.error(f"Sensor data status: {status}")

# Key metrics
latest = data.iloc[-1]

metric_cols = st.columns(4)

with metric_cols[0]:
    health = latest.get("health_index")
    st.metric(
        "Health Index",
        "N/A" if pd.isna(health) else f"{health:.1f} / 100",
    )

with metric_cols[1]:
    st.metric(
        "Feed TDS",
        "N/A" if pd.isna(latest.get("feed_tds")) else
        f"{latest.get('feed_tds'):.1f} ppm",
    )

with metric_cols[2]:
    st.metric(
        "Permeate TDS",
        "N/A" if pd.isna(latest.get("permeate_tds")) else
        f"{latest.get('permeate_tds'):.1f} ppm",
    )

with metric_cols[3]:
    st.metric(
        "Permeate Flow",
        "N/A" if pd.isna(latest.get("permeate_flow")) else
        f"{latest.get('permeate_flow'):.3f}",
    )

# Health and forecast
left, right = st.columns(2)

with left:
    st.subheader("Health Index")
    st.line_chart(
        data.set_index("timestamp")[["health_index"]],
        height=280,
    )

with right:
    st.subheader("Service Forecast")

    if not forecast["forecast_valid"]:
        st.warning(forecast["reason"])
    elif forecast["time_to_threshold_hours"] is None:
        st.info(forecast["reason"])
    else:
        hours = forecast["time_to_threshold_hours"]
        if hours == 0:
            st.warning("Health is already at or below the forecast threshold.")
        else:
            st.metric(
                "Estimated time to health threshold",
                f"{hours:.2f} hours",
            )
        st.caption(forecast["reason"])

    st.caption(
        f"Forecast threshold: {forecast['threshold']} health-index points"
    )

# Indicator plots
st.subheader("Sensor and Physics Indicators")

available_columns = [
    column
    for column in [
        "feed_tds",
        "permeate_tds",
        "temperature",
        "p1",
        "p2",
        "p3",
        "delta_p",
        "system_pressure",
        "salt_rejection",
        "salt_passage",
        "normalized_permeate_flow",
        "recovery",
    ]
    if column in data.columns
]

selected_indicators = st.multiselect(
    "Choose indicators to plot",
    options=available_columns,
    default=[
        column
        for column in [
            "feed_tds",
            "permeate_tds",
            "p1",
            "p2",
            "p3",
            "normalized_permeate_flow",
            "recovery",
        ]
        if column in available_columns
    ],
)

if selected_indicators:
    st.line_chart(
        data.set_index("timestamp")[selected_indicators],
        height=350,
    )
else:
    st.info("Select one or more indicators to display.")

# Water quality and sensor-health status
st.subheader("Water Quality and Sensor Health")

quality_cols = st.columns(2)

with quality_cols[0]:
    st.markdown("**Water-quality assessment**")
    quality_columns = [
        column
        for column in data.columns
        if "water_quality" in column.lower()
        or "quality_status" in column.lower()
        or "quality_warning" in column.lower()
    ]

    if quality_columns:
        st.dataframe(
            data[quality_columns].tail(10),
            use_container_width=True,
        )
    else:
        st.info("No dedicated water-quality status column was returned.")

with quality_cols[1]:
    st.markdown("**Sensor validation details**")
    st.write(f"Overall status: **{validation.get('status', status)}**")
    st.write(f"Validation issues: {validation.get('issues', [])}")

    sensor_columns = [
        column
        for column in [
            "feed_tds",
            "permeate_tds",
            "temperature",
            "p1",
            "p2",
            "p3",
            "permeate_flow",
            "concentrate_flow",
        ]
        if column in data.columns
    ]

    sensor_summary = pd.DataFrame(
        {
            "Sensor": sensor_columns,
            "Missing readings": [
                int(data[column].isna().sum())
                for column in sensor_columns
            ],
            "Unique observed values": [
                int(data[column].nunique(dropna=True))
                for column in sensor_columns
            ],
        }
    )

    st.dataframe(sensor_summary, use_container_width=True)

# Raw processed data and CSV export
st.subheader("Processed Data")

st.dataframe(
    data.tail(100),
    use_container_width=True,
)

csv_data = data.to_csv(index=False).encode("utf-8")

st.download_button(
    label="⬇ Download Processed Data CSV",
    data=csv_data,
    file_name=f"aqu amind_{result['scenario']}.csv".replace(" ", ""),
    mime="text/csv",
    use_container_width=True,
)

st.caption(
    "AquaMind S3 dashboard — synthetic demonstration output. "
    "Thresholds and forecasts are provisional and require validation "
    "against the purifier SDD and real hardware data."
)
