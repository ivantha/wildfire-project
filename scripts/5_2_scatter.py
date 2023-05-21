from pyspark.sql import SparkSession
import matplotlib.pyplot as plt
import numpy as np

spark = SparkSession.builder \
    .appName("Random Forest Regressor") \
    .config("spark.driver.memory", "20g") \
    .getOrCreate()

# Load data
df = spark.read.parquet(f"../tmp/datasets/pca_2")

# Convert Spark DataFrame to Pandas DataFrame
df = df.toPandas()

# Calculate the mean of the 'frp' column
mean_frp = np.mean(df['frp'])

# Create two classes based on the mean of the 'frp' column
df['class'] = np.where(df['frp'] < mean_frp, 1, 2)

# Sample a subset of the data points
sample_fraction = 0.01  # Adjust this value to control the number of points in the plot
df = df.sample(frac=sample_fraction)

# Plot data in a scatter plot
colors = {1: 'red', 2: 'blue'}
plt.scatter(df['pca_0'], df['pca_1'], c=df['class'].apply(lambda x: colors[x]), marker='.', s=1)
plt.xlabel('pca_0')
plt.ylabel('pca_1')
plt.title('Visualization of data after PCA')
# plt.show()

# Save the plot as a high-resolution image
plt.savefig('../tmp/scatter_pca_2.png', dpi=300)

# Stop the Spark session
spark.stop()
