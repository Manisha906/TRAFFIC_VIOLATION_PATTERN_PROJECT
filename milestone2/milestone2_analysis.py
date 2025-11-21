from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.functions import col, to_timestamp, hour, dayofweek, month, expr

# ================================
# 🚀 1. Initialize Spark Session
# ================================
spark = SparkSession.builder \
    .appName("Traffic_Violation_Analysis") \
    .getOrCreate()

print("🚀 Spark Session Started")

# ================================
# 📂 2. Load Cleaned Data
# ================================
input_path = r"C:\TRAFFIC_VIOLATION_PATTERN_PROJECT\milestone1\Data\cleaned_violations\csv_output"

df = spark.read.csv(input_path, header=True, inferSchema=True)
print("✅ Data Loaded Successfully")

df.printSchema()
df.show(5)

# ================================
# 🕒 3. Clean + Feature Engineering
# ================================
# Convert timestamp safely
df = df.withColumn("Timestamp", expr("try_to_timestamp(Timestamp, 'yyyy-MM-dd HH:mm:ss')"))

# Remove invalid rows
df = df.filter(col("Timestamp").isNotNull())
print("✅ Invalid timestamp rows removed")

# Extract hour, day, month
df = df.withColumn("hour", hour(col("Timestamp")))
df = df.withColumn("day_of_week", dayofweek(col("Timestamp")))
df = df.withColumn("month", month(col("Timestamp")))
print("✅ Time features extracted")

# ================================
# 📊 WEEK 3: Violation Type vs Hour
# ================================
crosstab = df.crosstab("Violation_Type", "hour")
print("✅ Crosstab created successfully")
crosstab.show(5)

week3_parquet = r"C:\TRAFFIC_VIOLATION_PATTERN_PROJECT\milestone2\Output\week3_time_analysis.parquet"
week3_csv = r"C:\TRAFFIC_VIOLATION_PATTERN_PROJECT\milestone2\Output\week3_time_analysis.csv"

crosstab.write.mode("overwrite").parquet(week3_parquet)
crosstab.write.mode("overwrite").option("header", True).csv(week3_csv)

print("✅ Week 3 results saved successfully")

# ================================
# 📍 WEEK 4: Aggregations by Location
# ================================
if "Location" in df.columns:
    # ✅ Total violations per location
    total_by_location = df.groupBy("Location").agg(F.count("*").alias("Total_Violations"))

    # ✅ Top 10 locations with highest violations
    top_n_locations = total_by_location.orderBy(F.desc("Total_Violations")).limit(10)

    print("✅ Total Violations by Location")
    total_by_location.show(5)

    print("✅ Top 10 Locations with Highest Violations")
    top_n_locations.show(10)

    # ✅ Save Week 4 outputs (CSV + Parquet)
    week4_total_parquet = r"C:\TRAFFIC_VIOLATION_PATTERN_PROJECT\\milestone2\Output\week4_total_by_location.parquet"
    week4_total_csv = r"C:\TRAFFIC_VIOLATION_PATTERN_PROJECT\milestone2\Output\week4_total_by_location.csv"

    week4_top_parquet = r"C:\TRAFFIC_VIOLATION_PATTERN_PROJECT\milestone2\Output\week4_top_locations.parquet"
    week4_top_csv = r"C:\TRAFFIC_VIOLATION_PATTERN_PROJECT\milestone2\Output\week4_top_locations.csv"

    total_by_location.write.mode("overwrite").parquet(week4_total_parquet)
    total_by_location.write.mode("overwrite").option("header", True).csv(week4_total_csv)

    top_n_locations.write.mode("overwrite").parquet(week4_top_parquet)
    top_n_locations.write.mode("overwrite").option("header", True).csv(week4_top_csv)

    print("✅ Week 4 results saved successfully (CSV + Parquet)")
else:
    print("⚠ Column 'Location' not found in dataset — please verify your data header names.")

# ================================
# 🛑 Stop Spark Session
# ================================
spark.stop()
print("🟢 Spark Session Stopped Successfully")
