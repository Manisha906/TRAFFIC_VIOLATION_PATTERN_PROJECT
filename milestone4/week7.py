import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import glob

st.set_page_config(page_title="Traffic Violation Dashboard", layout="wide")
st.title("🚦 Traffic Violation Analytics Dashboard")

# ------------------ LOAD DATA ------------------
@st.cache_data
def load_data():
    # --- WEEK 1/2 cleaned violations ---
    df_file = glob.glob(r"milestone1/Data/cleaned_violations/csv_output/*.csv")[0]
    df = pd.read_csv(df_file)

    # --- WEEK 5 (Pivot format) ---
    time_file = glob.glob(r"milestone3/week5_output/csv_output/vehicle/*.csv")[0]
    time_df = pd.read_csv(time_file)

    # Convert pivot 
    time_df = time_df.melt(var_name="hour", value_name="count")
    time_df = time_df[time_df["hour"].str.isnumeric()]  
    time_df["hour"] = time_df["hour"].astype(int)

    # --- WEEK 6 ---
    week6_file = glob.glob(r"milestone3/week6_output/csv/top_violation/*.csv")[0]
    week6 = pd.read_csv(week6_file)

    # --- WEEK 4 ---
    top_loc_file = glob.glob(r"milestone2/Output/week4_top_locations.csv/*.csv")[0]
    top_loc = pd.read_csv(top_loc_file)

    total_loc_file = glob.glob(r"milestone2/Output/week4_total_by_location.csv/*.csv")[0]
    total_loc = pd.read_csv(total_loc_file)

    # --- WEEK 3 ---
    time3_file = glob.glob(r"milestone2/Output/week3_time_analysis.csv/*.csv")[0]
    time3 = pd.read_csv(time3_file)

    return df, time_df, top_loc, total_loc, time3, week6

df, time_df, top_loc, total_loc, time3, week6 = load_data()
st.success("✅ Data loaded successfully!")

# ------------------ FILTERS ------------------
st.sidebar.header("🔍 Filters")

violation_types = sorted(df["Violation_Type"].dropna().unique())
selected_type = st.sidebar.selectbox("Violation Type", ["All"] + violation_types)

severity_levels = sorted(df["Severity"].dropna().unique())
selected_severity = st.sidebar.selectbox("Severity", ["All"] + severity_levels)
df["Timestamp"] = pd.to_datetime(df["Timestamp"], errors="coerce", dayfirst=True)
df = df.dropna(subset=["Timestamp"])
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

# ------------------ PREVIEW ------------------
st.subheader("📄 Filtered Violations Data Preview")
st.dataframe(filtered_df.head(20), use_container_width=True)
st.write(f"🔹 Total Records After Filter: {len(filtered_df)}")

# ------------------ VIOLATION TYPE FREQUENCY ------------------
st.subheader("📊 Violation Type Distribution")
type_counts = filtered_df["Violation_Type"].value_counts()
fig1, ax1 = plt.subplots()
ax1.bar(type_counts.index, type_counts.values)
plt.xticks(rotation=45)
st.pyplot(fig1)

# ------------------ WEEK 5: HOURLY TREND ------------------
st.subheader("⏱ Hourly Violation Trend")
temp = filtered_df.copy()
temp["hour"] = temp["Timestamp"].dt.hour
hourly_counts = temp["hour"].value_counts().sort_index()
fig2, ax2 = plt.subplots()
ax2.plot(hourly_counts.index, hourly_counts.values)
plt.xlabel("Hour (0-23)")
plt.ylabel("Count")
plt.title("Hourly Trend")
st.pyplot(fig2)

# ------------------ WEEK 4 TOP LOC ------------------
st.subheader("📍 Top Violation Locations")
top_locations = (
    filtered_df.groupby("Location").size().reset_index(name="Count")
              .sort_values("Count", ascending=False)
)
st.dataframe(top_locations, use_container_width=True)

# ------------------ WEEK 4 TOTAL LOC ------------------
st.subheader("📌 Total Violations by Location")
fig3, ax3 = plt.subplots(figsize=(10, 4))
ax3.bar(top_locations["Location"], top_locations["Count"])
plt.xticks(rotation=90)
plt.title("Total Violations by Location")
st.pyplot(fig3)

# ------------------ EXPORT ------------------
st.subheader("⬇ Export Filtered Results")
csv_export = filtered_df.to_csv(index=False)
st.download_button(
    "Download Filtered CSV",
    csv_export,
    file_name="filtered_results.csv",
    mime="text/csv"
)

st.info("Dashboard complete. Heatmaps & hotspot analysis will be added in Week 8.")