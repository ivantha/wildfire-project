import csv
import os

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, count, when


def create_summary(spark, dataset_name):
    # Load data
    df = spark.read.parquet(f"../../tmp/datasets/{dataset_name}")

    # create a new DataFrame with column names and data types
    columns_df = df.dtypes
    columns_df = spark.createDataFrame(columns_df, ['column_name', 'data_type'])
    columns_df = columns_df.toPandas()

    # Initialize the new columns for summary statistics
    columns_df['mean'] = None
    columns_df['stddev'] = None
    columns_df['min'] = None
    columns_df['max'] = None
    columns_df['null_count'] = None
    columns_df['null_percentage'] = None

    # Calculate total number of rows
    total_rows = df.count()

    # Iterate through the columns and calculate summary statistics
    for index, row in columns_df.iterrows():
        if row['data_type'] in ('int', 'bigint', 'smallint', 'float', 'double', 'decimal'):
            summary = df.selectExpr(f"mean({row['column_name']})", f"stddev({row['column_name']})", f"min({row['column_name']})", f"max({row['column_name']})").collect()[0]
            columns_df.at[index, 'mean'] = summary[0]
            columns_df.at[index, 'stddev'] = summary[1]
            columns_df.at[index, 'min'] = summary[2]
            columns_df.at[index, 'max'] = summary[3]

        # Calculate null count and null percentage for each column
        null_count = df.select(count(when(col(row['column_name']).isNull(), 1))).collect()[0][0]
        null_percentage = (null_count / total_rows) * 100

        columns_df.at[index, 'null_count'] = null_count
        columns_df.at[index, 'null_percentage'] = null_percentage

    # save the columns_df to a CSV file
    columns_df.to_csv(f'../../tmp/summary/{dataset_name}.csv', quoting=csv.QUOTE_ALL, index=False, float_format='%.2f', sep=',')


if __name__ == '__main__':
    # Create a SparkSession object
    spark = SparkSession.builder \
        .appName("MyApp") \
        .config("spark.driver.memory", "20g") \
        .getOrCreate()

    # Create the output directory
    summary_directory_path = '../../tmp/summary'
    if not os.path.exists(summary_directory_path):
        os.mkdir(summary_directory_path)

    create_summary(spark, 'simplified')
    create_summary(spark, 'good')
    create_summary(spark, 'small')
    create_summary(spark, 'pca_100')
    create_summary(spark, 'pca_75')
    create_summary(spark, 'pca_50')
    create_summary(spark, 'pca_25')
    create_summary(spark, 'pca_2')

    # Stop the Spark session
    spark.stop()
