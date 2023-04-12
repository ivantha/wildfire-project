from pyspark.sql import SparkSession
import matplotlib.pyplot as plt

spark = SparkSession.builder \
    .appName("Random Forest Regressor") \
    .config("spark.driver.memory", "20g") \
    .getOrCreate()

# Load data
df = spark.read.parquet(f"../../tmp/datasets/pca_2")

# Split the pca_features column into separate columns
df = df.select(df['pca_features'].getItem(0).alias('x'), df['pca_features'].getItem(1).alias('y'))

# Convert Spark DataFrame to Pandas DataFrame
pdf = df.toPandas()

print(pdf.head())

# Plot data in a scatter plot
plt.scatter(pdf['x'], pdf['y'])
plt.xlabel('Column 1')
plt.ylabel('Column 2')
plt.title('Scatter Plot')
plt.show()