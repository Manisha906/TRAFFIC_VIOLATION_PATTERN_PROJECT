import os
import pandas as pd
import random
from faker import Faker

# -------------------------------
# CONFIGURATION
# -------------------------------
USE_REAL_DATA = False  # Set True to load your real CSV instead of generating fake data
REAL_DATA_PATH = "C:/TRAFFIC_VIOLATION_PROJECT/real_violations.csv"  # Update path

OUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')
os.makedirs(OUT_DIR, exist_ok=True)
print(f"📂 Data folder ready at: {OUT_DIR}")

# -------------------------------
# LOAD REAL DATA
# -------------------------------
if USE_REAL_DATA:
    if not os.path.exists(REAL_DATA_PATH):
        raise FileNotFoundError(f"Real data file not found at {REAL_DATA_PATH}")
    
    df = pd.read_csv(REAL_DATA_PATH)

    # Ensure required columns exist
    required_cols = ["Violation_ID", "Timestamp", "Location", "Violation_Type",
                     "Vehicle_Type", "Severity", "Fine_Amount"]
    missing_cols = [c for c in required_cols if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing columns in real data: {missing_cols}")

    # Optional: clean timestamps and fines
    df['Timestamp'] = pd.to_datetime(df['Timestamp'], errors='coerce')
    df['Fine_Amount'] = pd.to_numeric(df['Fine_Amount'], errors='coerce')

    print("✅ Real data loaded successfully")

# -------------------------------
# GENERATE FAKE DATA
# -------------------------------
else:
    fake = Faker()
    n_records = 100
    violation_types = ["Speeding", "Red Light", "No Helmet", "Parking", "Seat Belt"]
    vehicle_types = ["Car", "Bike", "Truck", "Bus", "Auto"]
    severity_levels = ["Low", "Medium", "High"]

    data = []
    for _ in range(n_records):
        violation_type = random.choice(violation_types)
        if random.random() < 0.05:
            violation_type = None  # missing value

        timestamp = fake.date_time_this_year()
        if random.random() < 0.05:
            timestamp = "32/13/2025"  # malformed timestamp

        data.append({
            "Violation_ID": fake.uuid4(),
            "Timestamp": timestamp,
            "Location": f"{fake.latitude()},{fake.longitude()}",
            "Violation_Type": violation_type,
            "Vehicle_Type": random.choice(vehicle_types),
            "Severity": random.choice(severity_levels),
            "Fine_Amount": random.choice([random.randint(100, 2000), None, "abc"]),
        })

    df = pd.DataFrame(data)
    print("✅ Fake data generated successfully")

# -------------------------------
# SAVE DATA
# -------------------------------
csv_path = os.path.join(OUT_DIR, "violations.csv")
json_path = os.path.join(OUT_DIR, "violations.json")

df.to_csv(csv_path, index=False)
df.to_json(json_path, orient="records", lines=True)

print("💾 CSV file saved at:", csv_path)
print("💾 JSON file saved at:", json_path)