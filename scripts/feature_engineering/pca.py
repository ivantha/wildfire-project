import pyspark.sql.functions as F
from pyspark.ml import Pipeline
from pyspark.ml.feature import PCA, StandardScaler, VectorAssembler
from pyspark.sql import SparkSession

from util.data import process_frp


def do_pca(spark, num_cols, dataset_name):
    # Load data
    df = spark.read.parquet(f"../../tmp/datasets/good")

    # Calculate average of the 'frp' column
    df = df.withColumn("frp", process_frp(F.col("frp")))

    # Drop unnecessary columns
    columns_to_drop = [
        'Polygon_ID',
    ]
    df = df.drop(*columns_to_drop)

    # Prepare the features and target columns
    feature_columns = [col for col in df.columns if col != 'frp']
    target_column = 'frp'

    # Assemble the feature columns into a single vector
    assembler = VectorAssembler(inputCols=feature_columns, outputCol="features", handleInvalid="skip")

    # Standardize the features (important for PCA)
    scaler = StandardScaler(inputCol="features", outputCol="scaled_features", withStd=True, withMean=True)

    # Choose the number of components to keep
    n_components = num_cols

    # Apply PCA
    pca = PCA(k=n_components, inputCol="scaled_features", outputCol="pca_features")

    # Create a pipeline with the assembler, scaler, and PCA stages
    pipeline = Pipeline(stages=[assembler, scaler, pca])

    # Fit and transform the data
    model = pipeline.fit(df)
    principal_df = model.transform(df)

    # Select the PCA features and the target column 'frp'
    principal_df = principal_df.select("pca_features", target_column)

    # Save the resulting DataFrame to a Parquet file
    principal_df.write.parquet(f"../../tmp/datasets/{dataset_name}", mode="overwrite")


if __name__ == '__main__':
    # Initialize Spark session
    spark = SparkSession.builder.master("local").appName("PCA").getOrCreate()

    do_pca(spark, num_cols=100, dataset_name="pca_100")
    # do_pca(spark, num_cols=75, dataset_name="pca_75")
    # do_pca(spark, num_cols=50, dataset_name="pca_50")
    # do_pca(spark, num_cols=2, dataset_name="pca_2")

    # Stop Spark session
    spark.stop()
