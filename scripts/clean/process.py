import pyspark.sql.functions as F
from pyspark.sql import SparkSession
from pyspark.sql.functions import col
from pyspark.sql.types import DoubleType, FloatType, IntegerType, LongType

from util.data import process_frp
from util.timer import timeit


def remove_outliers(dataframe, columns):
    no_outliers_df = dataframe
    for column in columns:
        # Calculate Q1, Q3, and IQR
        quantiles = no_outliers_df.approxQuantile(column, [0.25, 0.75], 0.05)
        q1, q3 = quantiles[0], quantiles[1]
        iqr = q3 - q1

        # Define the lower and upper range for non-outliers
        lower_range = q1 - 1.5 * iqr
        upper_range = q3 + 1.5 * iqr

        # Remove outliers
        no_outliers_df = no_outliers_df.filter((col(column) >= lower_range) & (col(column) <= upper_range))

    return no_outliers_df


@timeit
def main():
    # Initialize Spark session
    spark = SparkSession.builder \
        .appName("Clean Data") \
        .config("spark.driver.memory", "20g") \
        .getOrCreate()

    # Load data
    df = spark.read.parquet(f"../../tmp/datasets/tiny")

    # Calculate average of the 'frp' column
    df = df.withColumn("frp", process_frp(F.col("frp")))

    print(f"# of rows before removing outliers: {df.count()}:,")

    # Manual list of numerical columns
    numerical_columns = ["frp"]

    # Automatically filter numerical columns based on DataFrame schema
    # numerical_columns = [field.name for field in df.schema.fields if field.dataType in (FloatType(), DoubleType(), IntegerType(), LongType())]

    cleaned_df = remove_outliers(df, numerical_columns)

    # Coalesce the DataFrame into a single partition
    cleaned_df = cleaned_df.coalesce(1)

    # Show the new number of rows
    print(f"# of rows after removing outliers: {cleaned_df.count()}:,")

    # Save the cleaned datasets as a Parquet file
    cleaned_df.write.mode("overwrite").parquet('../../tmp/datasets/processed')

    # Stop the Spark session
    spark.stop()


if __name__ == '__main__':
    main()
