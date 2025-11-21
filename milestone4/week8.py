import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import os
import glob

st.set_page_config(page_title="Traffic Violation Dashboard", layout="wide")
st.title("🚦 Traffic Violation Analytics Dashboard")

# ------------------ DATA LOADING ------------------
@st.cache_data
def load_csv_safe(path, columns=None):
    """Load CSV safely from file or folder"""
    if os.path.isfile(path):
        df = pd.read_csv(path, header=None if columns else 0)
        if columns:
            df.columns = columns
        return df
    elif os.path.isdir(path):
        files = glob.glob(os.path.join(path, "*.csv"))
        if not files:
            return pd.DataFrame(columns=columns if columns else [])
        df_list = [pd.read_csv(f, header=None if columns else 0) for f in files]
        df = pd.concat(df_list, ignore_index=True)
        if columns:
            df.columns = columns
        return df
    else:
        return pd.DataFrame(columns=columns if columns else [])

def load_all_data():
    try:
        # Milestone 1
        df = load_csv_safe(r"milestone1/Data/cleaned_violations/csv_output/")

        # Milestone 2
        week3_time = load_csv_safe(r"milestone2/Output/week3_time_analysis.csv")
        week4_total = load_csv_safe(r"milestone2/Output/week4_total_by_location.csv")
        week4_top = load_csv_safe(r"milestone2/Output/week4_top_locations.csv")

        # Milestone 3, Week 5
        severity_df = load_csv_safe(r"milestone3/week5_output/csv_output/severity/")
        vehicle_df = load_csv_safe(r"milestone3/week5_output/csv_output/vehicle/")
        violation_type_df = load_csv_safe(r"milestone3/week5_output/csv_output/violation_type/")

        if not vehicle_df.empty:
            vehicle_df = vehicle_df.melt(var_name="hour", value_name="count")
            vehicle_df = vehicle_df[vehicle_df["hour"].astype(str).str.isnumeric()]
            vehicle_df["hour"] = vehicle_df["hour"].astype(int)

        # Milestone 3, Week 6
        hotspots_columns = ["lat_cell","lon_cell","violation_count","z_score","Violation_Type","hour","extra1","extra2"]
        combined_hotspots_df = load_csv_safe(
            "milestone3/week6_output/csv/combined_hotspots", columns=hotspots_columns
        )

        top_violation_df = load_csv_safe(
            "milestone3/week6_output/csv/top_violation",
            columns=["lat_cell","lon_cell","Violation_Type","violation_type_count"]
        )

        # Clean Timestamp
        if not df.empty and "Timestamp" in df.columns:
            df["Timestamp"] = pd.to_datetime(df["Timestamp"], errors="coerce", dayfirst=True)
            df = df.dropna(subset=["Timestamp"])

        if not week3_time.empty and "Timestamp" in week3_time.columns:
            week3_time["Timestamp"] = pd.to_datetime(week3_time["Timestamp"], errors="coerce", dayfirst=True)
            week3_time = week3_time.dropna(subset=["Timestamp"])

        return df, week3_time, week4_total, week4_top, severity_df, vehicle_df, violation_type_df, combined_hotspots_df, top_violation_df

    except Exception as e:
        st.error(f"Error loading data: {e}")
        st.stop()

(df, week3_time, week4_total, week4_top,
 severity_df, vehicle_df, violation_type_df,
 combined_hotspots_df, top_violation_df) = load_all_data()
st.success("✅ All data loaded successfully!")

# ------------------ FILTERS ------------------
st.sidebar.header("🔍 Filters")
filtered_df = pd.DataFrame()
if not df.empty:
    # Clean Violation_Type
    df["Violation_Type"] = df["Violation_Type"].astype(str).str.strip().str.title()

    min_date, max_date = df["Timestamp"].min(), df["Timestamp"].max()
    selected_date = st.sidebar.date_input("Date Range", [min_date, max_date], key="date_input_week8")

    violation_types = sorted(df["Violation_Type"].dropna().unique())
    selected_type = st.sidebar.selectbox("Violation Type", ["All"] + violation_types, key="violation_type_week8")

    severity_levels = sorted(df["Severity"].dropna().unique())
    selected_severity = st.sidebar.selectbox("Severity", ["All"] + severity_levels, key="severity_week8")

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
st.subheader("📌 Key Insights (Filtered Data)")
if not filtered_df.empty:
    total_records = len(filtered_df)
    most_common_type = filtered_df['Violation_Type'].mode()[0]
    peak_hour = filtered_df['Timestamp'].dt.hour.value_counts().idxmax()
    top_location = filtered_df['Location'].mode()[0] if 'Location' in filtered_df.columns else "N/A"
    severe_count = filtered_df[filtered_df['Severity'].isin(['High','Severe','Critical'])].shape[0]
else:
    total_records = 0
    most_common_type = peak_hour = top_location = severe_count = "N/A"

col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("Total Records", total_records)
col2.metric("Most Common Violation", most_common_type)
col3.metric("Peak Hour", peak_hour)
col4.metric("Top Location", top_location)
col5.metric("Severe Violations", severe_count)

# ------------------ VIOLATION TYPE DISTRIBUTION ------------------
st.subheader("📊 Violation Type Distribution")
if not filtered_df.empty:
    type_counts = filtered_df["Violation_Type"].value_counts()
    fig1, ax1 = plt.subplots()
    ax1.bar(type_counts.index, type_counts.values, color='skyblue')
    plt.xticks(rotation=45)
    st.pyplot(fig1)
else:
    st.warning("No filtered data available.")

# ------------------ HOURLY TREND ------------------
st.subheader("⏱ Hourly Violation Trend")
if not filtered_df.empty:
    hourly_counts = filtered_df['Timestamp'].dt.hour.value_counts().reindex(range(24), fill_value=0)
    peak_hr = hourly_counts.idxmax()
    fig2, ax2 = plt.subplots(figsize=(10,5))
    ax2.plot(hourly_counts.index, hourly_counts.values, marker='o', color='blue', linewidth=2)
    ax2.scatter(peak_hr, hourly_counts[peak_hr], color='red', s=100, label=f"Peak Hour: {peak_hr}")
    ax2.set_xticks(range(24))
    ax2.set_xlabel("Hour of Day")
    ax2.set_ylabel("Number of Violations")
    ax2.set_title("Hourly Violation Trend")
    ax2.grid(True, linestyle='--', alpha=0.5)
    ax2.legend()
    st.pyplot(fig2)
else:
    st.warning("No data to plot hourly trend.")

# ------------------ TOP VIOLATIONS ------------------
st.subheader("📊 Top Violations")
if not filtered_df.empty:
    top_counts = filtered_df["Violation_Type"].value_counts()
    fig3, ax3 = plt.subplots()
    ax3.bar(top_counts.index, top_counts.values, color='seagreen')
    plt.xticks(rotation=45)
    ax3.set_title("Top Violations by Type")
    st.pyplot(fig3)
else:
    st.warning("No top violation data available.")

# ------------------ HOTSPOT HEATMAP ------------------
st.subheader("📍 Hotspot Heatmap")
if not combined_hotspots_df.empty:
    combined_hotspots_df["Latitude"] = combined_hotspots_df["lat_cell"].astype(float)
    combined_hotspots_df["Longitude"] = combined_hotspots_df["lon_cell"].astype(float)
    fig5, ax5 = plt.subplots(figsize=(10,6))
    ax5.scatter(
        combined_hotspots_df["Longitude"],
        combined_hotspots_df["Latitude"],
        s=50,
        c='red',
        alpha=0.6
    )
    ax5.set_xlabel("Longitude")
    ax5.set_ylabel("Latitude")
    ax5.set_title("Violation Hotspots")
    st.pyplot(fig5)
else:
    st.warning("No hotspot data available to plot.")

# ------------------ EXPORT FILTERED DATA ------------------
st.subheader("⬇ Export Filtered Results")
if not filtered_df.empty:
    csv_export = filtered_df.to_csv(index=False)
    st.download_button("Download Filtered CSV", csv_export, file_name="filtered_results.csv", mime="text/csv", key="download_csv_week8")
    json_export = filtered_df.to_json(orient="records")
    st.download_button("Download Filtered JSON", json_export, file_name="filtered_results.json", mime="application/json", key="download_json_week8")
else:
    st.info("No filtered data to export.")

st.info("✅ Dashboard complete.")

