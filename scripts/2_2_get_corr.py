import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.stat import Correlation
from pyspark.sql import SparkSession

# Initialize Spark session
spark = SparkSession.builder \
    .appName("Correlation Matrix") \
    .config("spark.driver.memory", "20g") \
    .getOrCreate()

# Load data
df = spark.read.parquet(f"../tmp/datasets/original")

# Filter out string type columns
num_cols = [col for col, dtype in zip(df.columns, df.dtypes) if dtype[1] != "string" and dtype[1] != "timestamp"]

# Assemble the features into a single vector
vector_assembler = VectorAssembler(inputCols=num_cols, outputCol="features", handleInvalid="skip")
df_transformed = vector_assembler.transform(df).select("features")

# Calculate the correlation matrix
corr_matrix = Correlation.corr(df_transformed, "features")
corr_array = corr_matrix.collect()[0][0].toArray()

# Convert the correlation matrix to a Pandas DataFrame
corr_matrix_df = pd.DataFrame(corr_array, index=num_cols, columns=num_cols)

# Create a DataFrame to store the column pairs and their correlations
column1 = []
column2 = []
correlations = []

for i, col1 in enumerate(num_cols):
    for j, col2 in enumerate(num_cols[i+1:], start=i+1):
        column1.append(col1)
        column2.append(col2)
        correlations.append(corr_matrix_df.loc[col1, col2])

correlation_pairs_df = pd.DataFrame({"Column1": column1, "Column2": column2, "Correlation": correlations})

# Sort the DataFrame by correlation in descending order
correlation_pairs_df = correlation_pairs_df.sort_values(by="Correlation", ascending=False)

# Save the sorted correlation pairs DataFrame to a CSV file
correlation_pairs_df.to_csv("../tmp/correlation_pairs.csv", index=False)

# Plot the correlation matrix using seaborn
sns.set(style="white")
fig, ax = plt.subplots(figsize=(100, 100))
sns.heatmap(corr_matrix_df, annot=True, fmt=".2f", linewidths=.5, cmap="coolwarm", ax=ax)

# Save the plot as a high-resolution image
fig.savefig('../tmp/correlation_matrix.png', dpi=300)

# Stop the Spark session
spark.stop()
