from pyspark.sql import SparkSession

# Create a SparkSession
spark = SparkSession.builder.appName("Convert Data Types").getOrCreate()

# Read the CSV file into a DataFrame
df = spark.read.format("csv").option("header", "true").option("inferSchema", "true").option("sep", "	").load("../dataset/features_array.csv")

# Coalesce the DataFrame into a single partition
df = df.coalesce(1)

# Save the new DataFrame as a Parquet file
df.write.mode("overwrite").parquet("../tmp/datasets/original")
