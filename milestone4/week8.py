import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import glob
import os

st.set_page_config(page_title="Traffic Violation Dashboard", layout="wide")
st.title("🚦 Traffic Violation Analytics Dashboard")

# ------------------ DATA LOADING ------------------
@st.cache_data
def load_data():
    try:
        # --- Milestone 1 ---
        df = pd.read_csv(
            r"milestone1/Data/cleaned_violations/csv_output/part-00000-50ebba7e-0b3c-4a69-87ff-d2b08ebe04fa-c000.csv"
        )

        # --- Milestone 2 ---
        week3_time = pd.read_csv(
            r"milestone2/Output/week3_time_analysis.csv/part-00000-c0c27571-a962-4d9d-966b-d8813c1e6a60-c000.csv"
        )
        week4_total = pd.read_csv(
            r"milestone2/Output/week4_total_by_location.csv/part-00000-4b5ed24f-416b-4c7f-bf7c-60d45245ae35-c000.csv"
        )
        week4_top = pd.read_csv(
            r"milestone2/Output/week4_top_locations.csv/part-00000-1b24aaac-bb9f-42c6-851f-2782e5da2c6c-c000.csv"
        )

        # --- Milestone 3, Week 5 ---
        severity_df = pd.read_csv(r"milestone3/week5_output/csv_output/severity/part-00000-fdea79ed-0c6a-4560-b8eb-992b450ffe88-c000.csv")
        vehicle_df = pd.read_csv(r"milestone3/week5_output/csv_output/vehicle/part-00000-b93cc586-eca2-4c84-a201-dda6201d3a34-c000.csv")
        violation_type_df = pd.read_csv(r"milestone3/week5_output/csv_output/violation_type/part-00000-835afdf3-8642-4272-9deb-5e80c5b2e6fd-c000.csv")

        # Handle vehicle trend
        vehicle_df = vehicle_df.melt(var_name="hour", value_name="count")
        vehicle_df = vehicle_df[vehicle_df["hour"].str.isnumeric()]
        vehicle_df["hour"] = vehicle_df["hour"].astype(int)

        # --- Helper for Week 6 CSVs (no header) ---
        def load_spark_csv(folder_path, columns):
            files = glob.glob(os.path.join(folder_path, "*.csv"))
            if not files:
                return pd.DataFrame(columns=columns)
            return pd.read_csv(files[0], header=None, names=columns)

        # --- Week 6 Datasets ---
        hotspots_df = load_spark_csv(
            "milestone3/week6_output/csv/hotspots",
            ["lat_cell", "lon_cell", "violation_count", "z_score"]
        )

        top_violation_df = load_spark_csv(
            "milestone3/week6_output/csv/top_violation",
            ["lat_cell", "lon_cell", "Violation_Type", "violation_type_count"]
        )

        peak_hour_df = load_spark_csv(
            "milestone3/week6_output/csv/peak_hour",
            ["lat_cell", "lon_cell", "hour", "hour_count"]
        )

        cluster_df = load_spark_csv(
            "milestone3/week6_output/csv/clusters",
            ["cluster", "violation_count"]
        )

        combined_hotspots_df = load_spark_csv(
            "milestone3/week6_output/csv/combined_hotspots",
            ["lat_cell", "lon_cell", "violation_count", "z_score", "Violation_Type", "hour"]
        )

        return df, week3_time, week4_total, week4_top, severity_df, vehicle_df, violation_type_df, cluster_df, combined_hotspots_df, hotspots_df, peak_hour_df, top_violation_df

    except Exception as e:
        st.error(f"Error loading data: {e}")
        st.stop()

# Load all datasets
(df, week3_time, week4_total, week4_top,
 severity_df, vehicle_df, violation_type_df,
 cluster_df, combined_hotspots_df, hotspots_df,
 peak_hour_df, top_violation_df) = load_data()

st.success("✅ All data loaded successfully!")

# ------------------ FILTERS ------------------
st.sidebar.header("🔍 Filters")
violation_types = sorted(df["Violation_Type"].dropna().unique())
selected_type = st.sidebar.selectbox("Violation Type", ["All"] + violation_types)

severity_levels = sorted(df["Severity"].dropna().unique())
selected_severity = st.sidebar.selectbox("Severity", ["All"] + severity_levels)

df["Timestamp"] = pd.to_datetime(df["Timestamp"])
min_date, max_date = df["Timestamp"].min(), df["Timestamp"].max()
selected_date = st.sidebar.date_input("Date Range", [min_date, max_date])

filtered_df = df.copy()
if selected_type != "All":
    filtered_df = filtered_df[filtered_df["Violation_Type"] == selected_type]
if selected_severity != "All":
    filtered_df = filtered_df[filtered_df["Severity"] == selected_severity]
filtered_df = filtered_df[
    (filtered_df["Timestamp"] >= pd.to_datetime(selected_date[0])) &
    (filtered_df["Timestamp"] <= pd.to_datetime(selected_date[1]))
]

# ------------------ KEY INSIGHTS ------------------
st.subheader("📌 Key Insights (Filtered Data")
total_records = len(filtered_df)
most_common_type = filtered_df['Violation_Type'].mode()[0] if total_records > 0 else "N/A"
peak_hour = filtered_df['Timestamp'].dt.hour.value_counts().idxmax() if total_records > 0 else "N/A"
top_location = filtered_df['Location'].mode()[0] if 'Location' in filtered_df.columns and total_records > 0 else "N/A"
severe_count = filtered_df[filtered_df['Severity'].isin(['High','Severe','Critical'])].shape[0] if 'Severity' in filtered_df.columns and total_records > 0 else "N/A"

col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("Total Records", total_records)
col2.metric("Most Common Violation", most_common_type)
col3.metric("Peak Hour", peak_hour)
col4.metric("Top Location", top_location)
col5.metric("Severe Violations", severe_count)

# ------------------ VIOLATION TYPE DISTRIBUTION ------------------
st.subheader("📊 Violation Type Distribution")
type_counts = filtered_df["Violation_Type"].value_counts()
fig1, ax1 = plt.subplots()
ax1.bar(type_counts.index, type_counts.values)
plt.xticks(rotation=45)
st.pyplot(fig1)

# ------------------ HOURLY TREND (Week5 Vehicle) ------------------
st.subheader("⏱ Hourly Violation Trend")
if not vehicle_df.empty:
    hourly_counts = vehicle_df.groupby("hour")["count"].sum()
    fig2, ax2 = plt.subplots()
    ax2.plot(hourly_counts.index, hourly_counts.values, marker='o')
    plt.xlabel("Hour (0–23)")
    plt.ylabel("Count")
    plt.title("Hourly Trend (Vehicle Data)")
    st.pyplot(fig2)
else:
    st.warning("No vehicle trend data available.")

# ------------------ TOP VIOLATIONS (Week6) ------------------
st.subheader("📊 Top Violations")
if not top_violation_df.empty:
    fig3, ax3 = plt.subplots()
    ax3.bar(top_violation_df['Violation_Type'], top_violation_df['violation_type_count'])
    plt.xticks(rotation=45)
    plt.title("Top Violations by Type")
    st.pyplot(fig3)
else:
    st.warning("No top violation data available.")

# ------------------ HOTSPOT HEATMAP (Week6) ------------------
st.subheader("📍 Hotspot Heatmap")
if not combined_hotspots_df.empty:
    # Derive approximate lat/lon center from cell coordinates
    combined_hotspots_df["Latitude"] = combined_hotspots_df["lat_cell"].astype(float)
    combined_hotspots_df["Longitude"] = combined_hotspots_df["lon_cell"].astype(float)

    fig5, ax5 = plt.subplots(figsize=(10,6))
    sns.kdeplot(
        x=combined_hotspots_df["Longitude"],
        y=combined_hotspots_df["Latitude"],
        cmap="Reds",
        fill=True,
        bw_adjust=0.5,
        ax=ax5
    )
    ax5.set_xlabel("Longitude")
    ax5.set_ylabel("Latitude")
    ax5.set_title("Violation Density Heatmap")
    st.pyplot(fig5)
else:
    st.warning("No hotspot data available to plot.")

# ------------------ EXPORT FILTERED DATA ------------------
st.subheader("⬇ Export Filtered Results")
csv_export = filtered_df.to_csv(index=False)
st.download_button("Download Filtered CSV", csv_export, file_name="filtered_results.csv", mime="text/csv")

if not filtered_df.empty:
    json_export = filtered_df.to_json(orient="records")
    st.download_button("Download Filtered JSON", json_export, file_name="filtered_results.json", mime="application/json")

st.info("✅ Dashboard complete - Week 8 integrates insights, hotspots, and export options.")