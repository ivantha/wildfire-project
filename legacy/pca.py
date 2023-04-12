import pandas as pd
from pyspark.sql import SparkSession
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


def do_pca(spark, num_cols, dataset_name):
    # Load data
    df = spark.read.parquet(f"../../tmp/datasets/small")

    df = df.toPandas()

    df['frp'] = df['frp'].apply(lambda x: sum(map(float, x.split(','))) / len(x.split(',')))

    df = df.drop([
        'Polygon_ID',
        'acq_date',
        'acq_time',
        # 'Neighbour',
        # 'Neighbour_frp',
        'CH_mean',
        'Neighbour_CH_mean'
    ], axis=1)

    # Separate the 'frp' target column from the feature columns
    features_df = df.drop('frp', axis=1)
    target_df = df['frp']

    # Standardize the features (important for PCA)
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features_df)

    # Number of components
    n_components = num_cols

    # Apply PCA
    pca = PCA(n_components=n_components)
    principal_components = pca.fit_transform(scaled_features)

    # Create a new DataFrame with the principal components
    columns = [f'PC{i + 1}' for i in range(n_components)]

    principal_df = pd.DataFrame(data=principal_components, columns=columns)

    # Add the target column 'frp' to the principal_df
    principal_df['frp'] = target_df

    principal_df = spark.createDataFrame(principal_df)

    # Save the resulting DataFrame to a Parquet file
    principal_df.write.parquet(f"../../tmp/datasets/{dataset_name}", mode="overwrite")


if __name__ == '__main__':
    # Initialize Spark session
    spark = SparkSession.builder.master("local").appName("PCA") \
        .config("spark.driver.memory", "20g") \
        .config("spark.driver.maxResultSize", "10g") \
        .getOrCreate()

    do_pca(spark, num_cols=100, dataset_name="pca_100")
    do_pca(spark, num_cols=75, dataset_name="pca_75")
    do_pca(spark, num_cols=50, dataset_name="pca_50")
    do_pca(spark, num_cols=25, dataset_name="pca_25")
    do_pca(spark, num_cols=2, dataset_name="pca_2")

    # Stop Spark session
    spark.stop()
