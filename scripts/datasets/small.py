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

# Count the number of rows with an empty value in any column
num_rows_with_empty_value = df.where(" or ".join([f"{c} IS NULL" for c in df.columns])).count()

print(f'# of rows with an empty value in any column: {num_rows_with_empty_value:,}')
print(f'# of rows without an empty value in any column: {df.count() - num_rows_with_empty_value:,}')
print(f'Percentage of rows with an empty value in any column: {num_rows_with_empty_value / df.count() * 100.0:,}')

# Remove the rows with an empty value in any column
df = df.dropna(subset=None)

# Save the cleaned datasets as a Parquet file
df.write.mode("overwrite").parquet('../../tmp/datasets/small')

# Stop the Spark session
spark.stop()
