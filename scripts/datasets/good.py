from pyspark.sql import SparkSession

# Initialize Spark session
spark = SparkSession.builder \
    .appName("Clean Data") \
    .config("spark.driver.memory", "20g") \
    .getOrCreate()

# Load data
df = spark.read.parquet(f"../../tmp/datasets/simplified")

# Drop columns that are mostly null
columns_to_drop = [
    'acq_date',
    'acq_time',
    'Neighbour_frp',
    'Neighbour_acq_time',
    'Shape',
    'Neighbour_Shape',
    'TEMP_ave',
    'Neighbour_frp',
    'Neighbour_c_longitude',
]
df = df.drop(*columns_to_drop)

# Coalesce the DataFrame into a single partition
df = df.coalesce(1)

# Save the cleaned datasets as a Parquet file
df.write.mode("overwrite").parquet('../../tmp/datasets/good')

# Stop the Spark session
spark.stop()
