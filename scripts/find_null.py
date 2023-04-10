import pandas as pd

from util.data import read_dataset

# Load data
df = read_dataset()

# count the number of null values in each column
null_count = df.isnull().sum()

# calculate the percentage of null values in each column
null_percentage = null_count / len(df) * 100

# create a dataframe with column_name, null count, and null percentage
df = pd.DataFrame({'column_name': null_count.index, 'null_count': null_count.values, 'null_percentage': null_percentage.values})

# save the dataframe to a CSV file
df.to_csv('../tmp/null_stats.csv', index=False)