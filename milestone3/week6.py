import os
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, count, lit, mean, stddev, hour, dayofweek, split, trim, expr
from pyspark.ml.clustering import KMeans
from pyspark.ml.feature import VectorAssembler
from functools import reduce

# -------------------------------
# 0. PYSPARK CONFIG
# -------------------------------
os.environ['PYSPARK_PYTHON'] = r"C:\Users\Admin\AppData\Local\Programs\Python\Python311\python.exe"
os.environ['PYSPARK_DRIVER_PYTHON'] = r"C:\Users\Admin\AppData\Local\Programs\Python\Python311\python.exe"

# -------------------------------
# 1. CREATE SPARK SESSION
# -------------------------------
print("\n🚀 Starting Spark Session for Week 6...")
spark = SparkSession.builder \
    .appName("Week 6 Hotspot Analysis") \
    .getOrCreate()
print("✔ Spark Session Started\n")

# -------------------------------
# 2. LOAD CLEANED PARQUET FILE
# -------------------------------
print("📥 Loading cleaned violations data...")
input_path = r"C:\TRAFFIC_VIOLATION_PATTERN_PROJECT\milestone1\Data\cleaned_violations\parquet_output"
df = spark.read.parquet(input_path)
df.show(5, truncate=False)

# -------------------------------
# 3. EXTRACT LATITUDE AND LONGITUDE
# -------------------------------
print("\n📍 Extracting latitude and longitude from Location column...")
df = df.withColumn("Latitude", trim(split(col("Location"), ",").getItem(0)).cast("double")) \
       .withColumn("Longitude", trim(split(col("Location"), ",").getItem(1)).cast("double"))

df = df.filter((col("Latitude").isNotNull()) & (col("Longitude").isNotNull()))
print(f"✔ Total rows with valid coordinates: {df.count()}\n")

# -------------------------------
# 4. AGGREGATE VIOLATIONS BY GRID
# -------------------------------
print("📊 Aggregating violations by grid cells...")
grid_size = 20.0  # increased to group nearby points
df = df.withColumn("lat_cell", (col("Latitude") / lit(grid_size)).cast("int") * lit(grid_size)) \
       .withColumn("lon_cell", (col("Longitude") / lit(grid_size)).cast("int") * lit(grid_size))

location_counts = df.groupBy("lat_cell", "lon_cell") \
                    .count() \
                    .withColumnRenamed("count", "violation_count") \
                    .orderBy(col("violation_count").desc())
location_counts.show(10)

# -------------------------------
# 5. HOTSPOT DETECTION (Z-SCORE OR COUNT THRESHOLD)
# -------------------------------
print("\n🔥 Identifying hotspots using z-score method...")
stats = location_counts.agg(
    mean("violation_count").alias("mean_count"),
    stddev("violation_count").alias("std_count")
).collect()[0]

mean_count = stats['mean_count']
std_count = stats['std_count']

if std_count is None or std_count == 0:
    hotspots = location_counts.filter(col("violation_count") > 1)
else:
    hotspots = location_counts.withColumn(
        "z_score",
        expr(f"try_divide(violation_count - {mean_count}, {std_count})")
    ).filter(col("z_score") > 1)  # reduced threshold to catch hotspots

print("✔ Hotspots identified:")
hotspots.show(10)

# -------------------------------
# 6. TOP VIOLATION TYPES & PEAK TIMES PER HOTSPOT
# -------------------------------
print("\n📌 Calculating top violation type and peak hour per hotspot...")

# Add hour and weekday
df = df.withColumn("hour", hour(col("Timestamp"))) \
       .withColumn("weekday", dayofweek(col("Timestamp")))

if hotspots.count() == 0:
    print("⚠ No hotspots found. Skipping top violation & peak hour analysis.")
    top_violation_df = spark.createDataFrame([], df.schema)
    peak_hour_df = spark.createDataFrame([], df.schema)
else:
    # Join only hotspot coordinates
    df_hotspot = df.alias("d").join(
        hotspots.alias("h"),
        (col("d.lat_cell") == col("h.lat_cell")) & (col("d.lon_cell") == col("h.lon_cell")),
        "inner"
    ).select(
        col("d.Latitude"),
        col("d.Longitude"),
        col("d.Timestamp"),
        col("d.Violation_Type"),
        col("d.hour"),
        col("h.lat_cell").alias("lat_cell"),
        col("h.lon_cell").alias("lon_cell"),
        col("h.violation_count"),
        col("h.z_score")
    )

    # Top violation type per hotspot
    top_violation_df = df_hotspot.groupBy("lat_cell", "lon_cell", "Violation_Type") \
        .count() \
        .withColumnRenamed("count", "violation_type_count")

    # Get the most frequent violation type per hotspot
    from pyspark.sql.window import Window
    from pyspark.sql.functions import row_number, desc

    w1 = Window.partitionBy("lat_cell", "lon_cell").orderBy(desc("violation_type_count"))
    top_violation_df = top_violation_df.withColumn("rn", row_number().over(w1)) \
                                       .filter(col("rn") == 1) \
                                       .drop("rn")

    # Peak hour per hotspot
    peak_hour_df = df_hotspot.groupBy("lat_cell", "lon_cell", "hour") \
        .count() \
        .withColumnRenamed("count", "hour_count")

    w2 = Window.partitionBy("lat_cell", "lon_cell").orderBy(desc("hour_count"))
    peak_hour_df = peak_hour_df.withColumn("rn", row_number().over(w2)) \
                               .filter(col("rn") == 1) \
                               .drop("rn")

    # Combine top violation and peak hour
    structured_hotspots = top_violation_df.join(
        peak_hour_df, on=["lat_cell", "lon_cell"], how="inner"
    ).join(
        hotspots, on=["lat_cell", "lon_cell"], how="inner"
    ).select(
        "lat_cell", "lon_cell", "violation_count", "z_score",
        "Violation_Type", "hour"
    )

print("✔ Structured hotspot data:")
structured_hotspots.show()
# -------------------------------
# 7. K-MEANS CLUSTERING
# -------------------------------
print("\n🧩 Running K-Means clustering on coordinates...")
assembler = VectorAssembler(inputCols=["Latitude", "Longitude"], outputCol="features")
df_features = assembler.transform(df)

k = 5
kmeans = KMeans(k=k, seed=42, featuresCol="features", predictionCol="cluster")
model = kmeans.fit(df_features)
df_clusters = model.transform(df_features)

cluster_counts = df_clusters.groupBy("cluster").count().withColumnRenamed("count", "violation_count")
print("✔ Cluster counts:")
cluster_counts.show()

# -------------------------------
# 8. COMBINED OUTPUT
# -------------------------------
print("\n📂 Creating combined output for hotspots + top violation + peak hour...")

dfs_to_join = [hotspots, top_violation_df, peak_hour_df]

def join_multiple(dfs, on=["lat_cell", "lon_cell"]):
    return reduce(lambda left, right: left.join(right, on=on, how='outer'), dfs)

combined_df = join_multiple(dfs_to_join).orderBy("lat_cell", "lon_cell")
combined_df.show(10)

# -------------------------------
# 9. SAVE ALL OUTPUTS
# -------------------------------
print("\n💾 Saving all outputs...")

hotspot_csv = "./milestone3/week6_output/csv/hotspots"
hotspot_parquet = "./milestone3/week6_output/parquet/hotspots"
top_violation_csv = "./milestone3/week6_output/csv/top_violation"
peak_hour_csv = "./milestone3/week6_output/csv/peak_hour"
cluster_csv = "./milestone3/week6_output/csv/clusters"
cluster_parquet = "./milestone3/week6_output/parquet/clusters"
combined_csv = "./milestone3/week6_output/csv/combined_hotspots"

hotspots.write.mode("overwrite").csv(hotspot_csv)
hotspots.write.mode("overwrite").parquet(hotspot_parquet)
top_violation_df.write.mode("overwrite").csv(top_violation_csv)
peak_hour_df.write.mode("overwrite").csv(peak_hour_csv)
cluster_counts.write.mode("overwrite").csv(cluster_csv)
cluster_counts.write.mode("overwrite").parquet(cluster_parquet)
combined_df.write.mode("overwrite").csv(combined_csv)

print("✔ All outputs saved successfully!")

# -------------------------------
# 10. STOP SPARK
# -------------------------------
spark.stop()
print("\n✨ Week 6 Hotspot Analysis Completed Successfully!")