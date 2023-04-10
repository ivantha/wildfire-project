import csv

import pandas as pd

from util.data import read_dataset

# Load data
df = read_dataset()

# create a new DataFrame with column names and data types
columns_df = pd.DataFrame({
	'column_name': df.columns,
	'data_type': df.dtypes
})

# get the summary statistics of the numerical columns in the DataFrame
summary_df = df.describe()

# transpose the summary statistics DataFrame to make it easier to merge with the columns DataFrame
summary_df = summary_df.transpose()

summary_df = summary_df.reset_index().rename(columns={'index': 'column_name'})

# merge the columns DataFrame with the summary statistics DataFrame on the column name
merged_df = pd.merge(columns_df, summary_df, on='column_name', how='left')

# save the merged DataFrame to a CSV file
merged_df.to_csv('../tmp/original_summary.csv', quoting=csv.QUOTE_ALL, index=False, float_format='%.2f', sep=',')
