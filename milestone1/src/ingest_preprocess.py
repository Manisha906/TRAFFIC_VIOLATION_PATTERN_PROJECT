import os
from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, StringType, DoubleType
from pyspark.sql.functions import col, when, trim
# ---------------------------------------------------------
# ⿡ Create Spark Session
# ---------------------------------------------------------
spark = (
    SparkSession.builder \
    .appName("Violation Data Ingestion and Cleaning") \
    .master("local[*]") \
    .config("spark.sql.debug.maxToStringFields", "200") \
    .getOrCreate()
)
spark.sparkContext.setLogLevel("ERROR")
print("✅ Spark session started successfully!")
# ---------------------------------------------------------
# ⿢ Define schema  
# ---------------------------------------------------------
from pyspark.sql.types import *
schema = StructType([
    StructField("Violation_ID", StringType(), True),
    StructField("Timestamp", StringType(), True),
    StructField("Location", StringType(), True),
    StructField("Violation_Type", StringType(), True),
    StructField("Vehicle_Type", StringType(), True),
    StructField("Severity", StringType(), True),
    StructField("Fine_Amount", StringType(),True)
])
# ---------------------------------------------------------
# ⿣ Read CSV file
# ---------------------------------------------------------
file_path = os.path.join("milestone1","Data", "violations.csv")
if not os.path.exists(file_path):
    raise FileNotFoundError(f"❌ File not found: {file_path}")
df_raw = spark.read.csv(file_path, header=True, schema=schema)
print(f"📊 Total records (raw): {df_raw.count()}")
# ---------------------------------------------------------
# ⿤ Data cleaning / preprocessing
# ---------------------------------------------------------
for c in df_raw.columns:
    df_raw = df_raw.withColumn(c, trim(col(c)))
df_clean = df_raw.withColumn(
    "Fine_Amount",
    when(col("Fine_Amount").rlike("^[0-9]+(\\.[0-9]+)?$"), col("Fine_Amount").cast(DoubleType()))
    .otherwise(None)
)
# Separate valid and invalid rows
good_data = df_clean.filter(col("Fine_Amount").isNotNull())
bad_data = df_clean.filter(col("Fine_Amount").isNull())
print(f"✅ Valid rows (good): {good_data.count()}")
print(f"⚠ Invalid rows (bad Fine_Amount): {bad_data.count()}")
print("\n✅ Sample of cleaned data:")
good_data.show(5, truncate=False)
# ⿦ Write cleaned data (CSV + Parquet)
output_base = os.path.join("milestone1", "Data", "cleaned_violations")
output_csv_path = os.path.join(output_base, "csv_output")
output_parquet_path = os.path.join(output_base, "parquet_output")

try:
    # Write CSV
    good_data.write.mode("overwrite").option("header", True).csv(output_csv_path)
    print(f"✅ Cleaned CSV data written to: {output_csv_path}")

    # Write Parquet
    good_data.write.mode("overwrite").parquet(output_parquet_path)
    print(f"✅ Cleaned Parquet data written to: {output_parquet_path}")

except Exception as e:
    print(f"⚠ Warning: Could not write cleaned data due to: {e}")
# ---------------------------------------------------------
# ⿧ Stop Spark
# ---------------------------------------------------------
spark.stop()
print("✅ Spark session stopped successfully!")