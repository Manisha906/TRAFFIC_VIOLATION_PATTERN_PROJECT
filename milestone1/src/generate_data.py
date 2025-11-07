import os
import pandas as pd
from faker import Faker
import random
import numpy as np

# Initialize Faker
fake = Faker()

# Output folder
OUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')
os.makedirs(OUT_DIR, exist_ok=True)
print(f"📂 Data folder ready at: {OUT_DIR}")

# Number of records
n_records = 100

# Violation categories
violation_types = ["Speeding", "Red Light", "No Helmet", "Parking", "Seat Belt"]
vehicle_types = ["Car", "Bike", "Truck", "Bus", "Auto"]
severity_levels = ["Low", "Medium", "High"]

data = []
for _ in range(n_records):
    # Randomly make some records incomplete to simulate inconsistencies
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
# Convert to DataFrame
df = pd.DataFrame(data)
# Write to CSV and JSON
csv_path = os.path.join(OUT_DIR, "violations.csv")
json_path = os.path.join(OUT_DIR, "violations.json")

df.to_csv(csv_path, index=False)
df.to_json(json_path, orient="records", lines=True)

print("✅ CSV file generated at:", csv_path)
print("✅ JSON file generated at:",json_path)