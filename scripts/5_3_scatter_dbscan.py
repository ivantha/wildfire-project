import matplotlib.pyplot as plt
import numpy as np
from pyspark.sql import SparkSession
from sklearn.cluster import DBSCAN

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

# Use DBSCAN to cluster nearby points
eps_value = 0.8  # Adjust this value to control the distance threshold for merging points
min_samples_value = 5  # Adjust this value to control the minimum number of samples required to form a cluster
dbscan = DBSCAN(eps=eps_value, min_samples=min_samples_value)
df['cluster'] = dbscan.fit_predict(df[['pca_0', 'pca_1']])

# Calculate the centroid of each cluster
df_centroids = df.groupby(['cluster', 'class']).agg({'pca_0': 'mean', 'pca_1': 'mean'}).reset_index()

# Plot data in a scatter plot
colors = {1: 'red', 2: 'blue'}
plt.scatter(df_centroids['pca_0'], df_centroids['pca_1'], c=df_centroids['class'].apply(lambda x: colors[x]), marker='.', s=1)
plt.xlabel('pca_0')
plt.ylabel('pca_1')
plt.title('Visualization of data after PCA with merged points')
# plt.show()

# Save the plot as a high-resolution image
plt.savefig('../tmp/scatter_pca_2_dbscan.png', dpi=300)

# Stop the Spark session
spark.stop()
