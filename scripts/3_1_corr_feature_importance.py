import pandas as pd
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.stat import Correlation
from pyspark.sql import SparkSession

# Create a SparkSession object
spark = SparkSession.builder \
    .appName("Correlation-based Feature Selection") \
    .config("spark.driver.memory", "20g") \
    .getOrCreate()

# Load data
df = spark.read.parquet(f"../tmp/datasets/original")

# Drop unnecessary columns
df = df.drop(
    "Polygon_ID",
    "frp",
    'acq_date',
    'acq_time',
    'Neighbour_frp',
    'Neighbour_acq_time',
    'Shape',
    'Neighbour_Shape',
    'TEMP_ave',
    'Neighbour_frp',
    'Neighbour_c_longitude',
)

# Convert the features to a vector column
assembler = VectorAssembler(inputCols=df.columns[:-1], outputCol="features", handleInvalid="skip")
df = assembler.transform(df)

# Calculate the correlation matrix between the features and the target variable
correlation_matrix = Correlation.corr(df, "features", "pearson").head()[0].toArray()

# Extract the correlation coefficients for the target variable
correlations = correlation_matrix[-1][:-1]

# Create a list of tuples containing the feature name and its correlation coefficient
feature_importance = [(col, corr) for col, corr in zip(df.columns[:-1], correlations)]

# Sort the features by their correlation coefficient in descending order
feature_importance = sorted(feature_importance, key=lambda x: abs(x[1]), reverse=True)

# Create a pandas dataframe with feature and importance
df_importance = pd.DataFrame(feature_importance, columns=['feature', 'correlation_coefficient'])

# Save the feature importance to a CSV file
df_importance.to_csv("../tmp/feature_importance.csv", index=True)
