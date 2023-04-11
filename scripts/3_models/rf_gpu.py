from pyspark.ml import Pipeline
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import RandomForestRegressor
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.sql import SparkSession
import pyspark.sql.functions as F

from pyspark.sql.functions import udf
from pyspark.sql.types import FloatType

# Define a UDF to process the 'frp' column
@udf(returnType=FloatType())
def process_frp(frp):
    values = list(map(float, frp.split(',')))
    return sum(values) / len(values)

# Initialize Spark session with RAPIDS configuration
spark = SparkSession.builder \
    .appName("Random Forest Regressor") \
    .config("spark.plugins", "com.nvidia.spark.SQLPlugin") \
    .config("spark.rapids.sql.concurrentGpuTasks", "2") \
    .config("spark.executor.memory", "20g") \
    .config("spark.executor.resource.gpu.amount", "1") \
    .config("spark.task.resource.gpu.amount", "0.5") \
    .config("spark.driver.memory", "20g") \
    .getOrCreate()

# Load data
df = spark.read.parquet(f"../../tmp/datasets/good")

# Process the 'frp' column
df = df.withColumn("frp", process_frp(F.col("frp")))

# Drop unnecessary columns
df = df.drop("Polygon_ID")

# Split data into training and testing sets
train, test = df.randomSplit([0.9, 0.1], seed=42)

# Define X and y
feature_cols = [col for col in df.columns if col != "frp"]
assembler = VectorAssembler(inputCols=feature_cols, outputCol="features", handleInvalid="keep")

# Set Random Forest parameters
params = {
    'numTrees': 10,
    'maxDepth': 40,
    'minInstancesPerNode': 2,
    'minInfoGain': 0.0,
    'seed': 42
}

# Train the model
rf = RandomForestRegressor(labelCol="frp", featuresCol="features", **params)
pipeline = Pipeline(stages=[assembler, rf])
model = pipeline.fit(train)

# Make predictions on the testing set
predictions = model.transform(test)

# Calculate evaluation metrics
evaluator = RegressionEvaluator(labelCol="frp", predictionCol="prediction", metricName="mse")
mse = evaluator.evaluate(predictions)
rmse = mse ** 0.5
evaluator_r2 = RegressionEvaluator(labelCol="frp", predictionCol="prediction", metricName="r2")
r2 = evaluator_r2.evaluate(predictions)

# Print evaluation metrics
print('Mean Squared Error:', mse)
print('Root Mean Squared Error:', rmse)
print('R-squared:', r2)

# Stop the Spark session
spark.stop()
