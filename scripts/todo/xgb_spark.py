import pyspark.sql.functions as F
from pyspark.ml import Pipeline
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml.regression import GBTRegressor
from pyspark.sql import SparkSession

from util.data import process_value_list_str

# Initialize Spark session
spark = SparkSession.builder.master("local") \
    .appName("XGBoostRegression") \
    .getOrCreate()

# Load data
df = spark.read.parquet(f"../../tmp/datasets/good")

# Process the 'frp' column
df = df.withColumn("frp", process_value_list_str(F.col("frp")))

df = df.drop(
    "Polygon_ID"
)

# Convert pandas DataFrame to PySpark DataFrame
spark_df = spark.createDataFrame(df)

# Split data into training and testing sets
train, test = spark_df.randomSplit([0.9, 0.1], seed=42)

# Define input and output columns
input_cols = [col for col in train.columns if col != 'frp']
output_col = 'frp'

# Create the pipeline for data preprocessing and training
vector_assembler = VectorAssembler(inputCols=input_cols, outputCol="unscaled_features")
scaler = StandardScaler(inputCol="unscaled_features", outputCol="features")
gbt = GBTRegressor(labelCol=output_col, featuresCol="features", seed=42, maxDepth=60, maxIter=100)
pipeline = Pipeline(stages=[vector_assembler, scaler, gbt])

# Train the model
model = pipeline.fit(train)

# Make predictions on the testing set
predictions = model.transform(test)

# Calculate evaluation metrics
evaluator = RegressionEvaluator(labelCol=output_col, predictionCol="prediction", metricName="mse")
mse = evaluator.evaluate(predictions)
rmse = evaluator.evaluate(predictions, {evaluator.metricName: "rmse"})
r2 = evaluator.evaluate(predictions, {evaluator.metricName: "r2"})

# Print evaluation metrics
print('Mean Squared Error:', mse)
print('Root Mean Squared Error:', rmse)
print('R-squared:', r2)

# Stop Spark session
spark.stop()
