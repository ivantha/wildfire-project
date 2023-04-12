import pyspark.sql.functions as F
from pyspark.ml import Pipeline
from pyspark.ml.feature import VectorAssembler, PCA
from pyspark.sql import SparkSession
from pyspark.sql.functions import udf
from pyspark.sql.types import ArrayType, DoubleType

from util.data import process_frp


def do_pca(spark, num_cols, dataset_name):
    # Load data
    df = spark.read.parquet(f"../../tmp/datasets/small")

    # Calculate average of the 'frp' column
    df = df.withColumn("frp", process_frp(F.col("frp")))

    # Drop unnecessary columns
    columns_to_drop = [
        'Polygon_ID',
    ]
    df = df.drop(*columns_to_drop)

    # Get a list of all the columns, except the "frp" column
    input_cols = [col for col in df.columns if col != "frp"]

    # Assemble the features into a single vector
    assembler = VectorAssembler(inputCols=input_cols, outputCol="features")

    # Perform PCA on the dataset and reduce the number of columns to a given value n
    n = num_cols
    pca = PCA(k=n, inputCol="features", outputCol="pca_features")

    # Create a pipeline to process the data
    pipeline = Pipeline(stages=[assembler, pca])

    # Fit the pipeline to the data
    model = pipeline.fit(df)

    # Transform the data
    result = model.transform(df)

    # Define a UDF to convert an array of doubles to separate columns
    def to_columns(array):
        return [float(x) for x in array]

    to_columns_udf = udf(to_columns, ArrayType(DoubleType()))

    # Apply the UDF to create multiple PCA columns
    result = result.withColumn("pca_columns", to_columns_udf("pca_features"))

    # Create a new dataset with n+1 columns: the PCA features and the "frp" column
    pca_columns = [f"pca_{i}" for i in range(n)]
    for i in range(n):
        result = result.withColumn(pca_columns[i], result["pca_columns"].getItem(i))

    selected_cols = ["frp"] + pca_columns
    result_selected = result.select(selected_cols)

    # Save the resulting DataFrame to a Parquet file
    result_selected.write.parquet(f"../../tmp/datasets/{dataset_name}", mode="overwrite")


if __name__ == '__main__':
    # Initialize the Spark session
    spark = SparkSession.builder \
        .appName("PCA") \
        .config("spark.driver.memory", "20g") \
        .getOrCreate()

    do_pca(spark, num_cols=100, dataset_name="pca_100")
    do_pca(spark, num_cols=75, dataset_name="pca_75")
    do_pca(spark, num_cols=50, dataset_name="pca_50")
    do_pca(spark, num_cols=25, dataset_name="pca_25")
    do_pca(spark, num_cols=2, dataset_name="pca_2")

    # Stop Spark session
    spark.stop()
