from pyspark.sql import SparkSession
from pyspark.sql.functions import col, coalesce, expr

# -------------------------------
# 1. CREATE SPARK SESSION
# -------------------------------
print("\n🚀 Starting Spark Session...")
spark = SparkSession.builder \
    .appName("Week 5 Violation Analysis") \
    .getOrCreate()
print("✔ Spark Session Started\n")

# -------------------------------
# 2. LOAD CLEANED PARQUET FILE
# -------------------------------
print("📥 Loading cleaned violations data...")
input_path = "./milestone1/data/cleaned_violations/parquet_output"
df = spark.read.parquet(input_path)

total_rows = df.count()
print(f"✔ Total rows loaded: {total_rows}\n")
df.show(5, truncate=False)

# -------------------------------
# 3. SAFE TIMESTAMP PARSING (with try_to_timestamp)
# -------------------------------
print("\n⏳ Parsing timestamps safely using try_to_timestamp...")

df = df.withColumn(
    "Timestamp",
    coalesce(
        expr("try_to_timestamp(Timestamp, 'yyyy-MM-dd HH:mm:ss')"),
        expr("try_to_timestamp(Timestamp, 'dd/MM/yyyy HH:mm:ss')"),
        expr("try_to_timestamp(Timestamp, 'MM/dd/yyyy HH:mm:ss')")
    )
)

# Drop rows where Timestamp could not be parsed
df = df.filter(col("Timestamp").isNotNull())

print("✔ Timestamp parsing completed\n")
df.select("Timestamp").show(5, truncate=False)

# -------------------------------
# 4. ANALYSIS – BASIC STATISTICS
# -------------------------------
print("\n📊 Total number of violations:")
df.groupBy().count().show()

print("\n📌 Violations by type:")
violation_type_df = df.groupBy("Violation_Type").count()
violation_type_df.show()

print("\n🔥 Violations by severity:")
severity_df = df.groupBy("Severity").count()
severity_df.show()

print("\n🚗 Violations by vehicle type:")
vehicle_df = df.groupBy("Vehicle_Type").count()
vehicle_df.show()

# -------------------------------
# 5. SAVE OUTPUTS
# -------------------------------
output_csv = "./milestone3/week5_output/csv_output"
output_parquet = "./milestone3/week5_output/parquet_output"

print("\n💾 Saving analysis outputs...")

# CSV
violation_type_df.write.mode("overwrite").csv(output_csv + "/violation_type")
severity_df.write.mode("overwrite").csv(output_csv + "/severity")
vehicle_df.write.mode("overwrite").csv(output_csv + "/vehicle")

# Parquet
violation_type_df.write.mode("overwrite").parquet(output_parquet + "/violation_type")
severity_df.write.mode("overwrite").parquet(output_parquet + "/severity")
vehicle_df.write.mode("overwrite").parquet(output_parquet + "/vehicle")

print("✔ Output Saved Successfully!")

# -------------------------------
# 6. STOP SPARK
# -------------------------------
spark.stop()
print("\n✨ Week 5 Analysis Completed Successfully!")