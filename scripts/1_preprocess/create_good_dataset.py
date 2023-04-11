from pyspark.sql import SparkSession
from pyspark.sql.functions import col

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

# # Count the number of rows with an empty value in any column
# num_rows_with_empty_value = df.where(" or ".join([f"{c} IS NULL" for c in df.columns])).count()
#
# print("# of rows with an empty value in any column: ", num_rows_with_empty_value)
# print("# of rows without an empty value in any column: ", df.count() - num_rows_with_empty_value)
# print("Percentage of rows with an empty value in any column: ", num_rows_with_empty_value / df.count() * 100.0)
#
# # Remove the rows with an empty value in any column
# df = df.dropna(subset=None)

# Save the cleaned dataset as a Parquet file
df.write.mode("overwrite").parquet('../../tmp/datasets/good')

# Stop the Spark session
spark.stop()
