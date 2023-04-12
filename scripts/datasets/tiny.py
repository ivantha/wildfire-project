from pyspark.sql import SparkSession
from pyspark.sql.functions import rand

# Initialize Spark session
spark = SparkSession.builder \
    .appName("Clean Data") \
    .config("spark.driver.memory", "20g") \
    .getOrCreate()

# Load data
df = spark.read.parquet(f"../../tmp/datasets/small")

# Randomly sample the dataset
sampled_df = df.orderBy(rand()).limit(int(df.count() * 0.1))  # sampling 10% of the data

# Coalesce the DataFrame into a single partition
sampled_df = sampled_df.coalesce(1)

# Show the new number of rows
print("Number of rows after sampling:", sampled_df.count())

# Save the cleaned datasets as a Parquet file
df.write.mode("overwrite").parquet('../../tmp/datasets/tiny')

# Stop the Spark session
spark.stop()
