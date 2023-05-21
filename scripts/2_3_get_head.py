import os

from pyspark.sql import SparkSession


def save_head_as_csv(spark, dataset_name):
    # Load data
    df = spark.read.parquet(f"../tmp/datasets/{dataset_name}")

    # List of all columns to select
    columns_to_select = [
        'frp',
        'acq_date',
        'acq_time',
        'Neighbour_frp',
        'Neighbour_acq_time',
        'Shape',
        'Neighbour_Shape',

        'acq_year',
        'acq_month',
        'acq_day',
        'acq_day_of_the_week',
        'acq_day_of_the_month',
        'acq_spring',
        'acq_summer',
        'acq_fall',
        'acq_winter'
    ]

    # Only keep columns that actually exist in the DataFrame
    columns_to_select = [col for col in columns_to_select if col in df.columns]

    # Select the existing columns and limit the DataFrame
    head_df = df.select(*columns_to_select).limit(100)

    # Save as CSV file
    head_directory_path = '../tmp/head'
    if not os.path.exists(head_directory_path):
        os.mkdir(head_directory_path)

    head_df.toPandas().to_csv(f'../tmp/head/{dataset_name}.csv', index=False)


if __name__ == '__main__':
    # Create a SparkSession object
    spark = SparkSession.builder \
        .appName("MyApp") \
        .config("spark.driver.memory", "20g") \
        .getOrCreate()

    save_head_as_csv(spark, 'original')
    save_head_as_csv(spark, 'processed')

    # Stop the Spark session
    spark.stop()
