from pyspark.sql import SparkSession

# Initialize the Spark session
spark = SparkSession.builder \
    .appName("Convert Data to DataFrame") \
    .getOrCreate()

# Define the data
data = [("Java", "20000"), ("Python", "100000"), ("Scala", "3000")]

# Convert the data to a DataFrame
df = spark.createDataFrame(data, schema=["Language", "Users"])

# Show the DataFrame
df.show()

# Stop the Spark session
spark.stop()
