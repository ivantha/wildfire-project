import pyspark.sql.functions as F
from pyspark.sql import SparkSession
from scipy import rand

from util.data import process_value_list_str


def drop_unnecessary_columns(df):
    columns_to_drop = [
        'TEMP_ave',  # Entirely empty

        'Shape',
        'Neighbour_Shape',
    ]
    df = df.drop(*columns_to_drop)

    return df


def process_timestamp(df):
    # Assuming acq_date is already in date format. If not, convert it using to_date function
    df = df.withColumn('acq_date', F.to_date(F.col('acq_date'), 'yyyy-MM-dd'))

    # Extract year, month, day
    df = df.withColumn('acq_year', F.year('acq_date'))
    df = df.withColumn('acq_month', F.month('acq_date'))
    df = df.withColumn('acq_day', F.dayofmonth('acq_date'))

    # Extract day_of_the_week (1 - Sunday, 7 - Saturday)
    df = df.withColumn('acq_day_of_the_week', F.dayofweek('acq_date'))

    # Extract day_of_the_month
    df = df.withColumn('acq_day_of_the_month', F.dayofmonth('acq_date'))

    # Create separate columns for each season
    df = df.withColumn('acq_spring', F.when((F.col('acq_month').between(3, 5)), 1).otherwise(0))
    df = df.withColumn('acq_summer', F.when((F.col('acq_month').between(6, 8)), 1).otherwise(0))
    df = df.withColumn('acq_fall', F.when((F.col('acq_month').between(9, 11)), 1).otherwise(0))
    df = df.withColumn('acq_winter', F.when((F.col('acq_month').isin([12, 1, 2])), 1).otherwise(0))

    # Drop the original 'acq_date' column
    df = df.drop('acq_date')

    return df


def process_list_str_columns(df):
    df = df.withColumn("frp", process_value_list_str(F.col("frp")))
    df = df.withColumn("acq_time", process_value_list_str(F.col("acq_time")))
    df = df.withColumn("Neighbour_frp", process_value_list_str(F.col("Neighbour_frp")))
    df = df.withColumn("Neighbour_acq_time", process_value_list_str(F.col("Neighbour_acq_time")))

    return df


def remove_outliers(df):
    print(f"# of rows before removing outliers: {df.count():,}")

    # Manual list of numerical columns
    numerical_columns = ["frp"]

    # Automatically filter numerical columns based on DataFrame schema
    # numerical_columns = [field.name for field in df.schema.fields if field.dataType in (FloatType(), DoubleType(), IntegerType(), LongType())]

    no_outliers_df = df
    for column in numerical_columns:
        # Calculate Q1, Q3, and IQR
        quantiles = no_outliers_df.approxQuantile(column, [0.25, 0.75], 0.05)
        q1, q3 = quantiles[0], quantiles[1]
        iqr = q3 - q1

        # Define the lower and upper range for non-outliers
        lower_range = q1 - 1.5 * iqr
        upper_range = q3 + 1.5 * iqr

        # Remove outliers
        no_outliers_df = no_outliers_df.filter((F.col(column) >= lower_range) & (F.col(column) <= upper_range))

    print(f"# of rows after removing outliers: {df.count():,}")

    return no_outliers_df


def drop_empty_rows(df):
    # Count the number of rows with an empty value in any column
    num_rows_with_empty_value = df.where(" or ".join([f"{c} IS NULL" for c in df.columns])).count()

    print(f'# of rows with an empty value in any column: {num_rows_with_empty_value:,}')
    print(f'# of rows without an empty value in any column: {df.count() - num_rows_with_empty_value:,}')
    print(f'Percentage of rows with an empty value in any column: {num_rows_with_empty_value / df.count() * 100.0:,}')

    # Remove the rows with an empty value in any column
    df = df.dropna(subset=None)

    return df


def randomly_sample(df):
    # Randomly sample the dataset
    df = df.orderBy(rand()).limit(int(df.count() * 0.01))  # sampling

    # Show the new number of rows
    print("Number of rows after sampling:", df.count())

    return df


if __name__ == '__main__':
    # Initialize Spark session
    spark = SparkSession.builder \
        .appName("Clean Data") \
        .config("spark.driver.memory", "20g") \
        .getOrCreate()

    # Load data
    df = spark.read.parquet(f"../tmp/datasets/original")

    df = drop_unnecessary_columns(df)
    df = process_list_str_columns(df)
    df = process_timestamp(df)
    df = remove_outliers(df)
    df = drop_empty_rows(df)
    # df = randomly_sample(df)

    # Coalesce the DataFrame into a single partition
    df = df.coalesce(1)

    # Save the cleaned datasets as a Parquet file
    df.write.mode("overwrite").parquet('../tmp/datasets/processed')

    # Stop the Spark session
    spark.stop()
