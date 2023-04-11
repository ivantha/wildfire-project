import csv

from legacy.data import read_dataset

# Load data
df = read_dataset()

# drop columns that are mostly null
df = df.drop(columns=[
	'Neighbour',
	'Neighbour_frp',
	'Neighbour_c_longitude',
	'WCOMP',
	'WDIR_ave',
], axis=1)

# count the number of rows with an empty value in any column
num_rows_with_empty_value = df.isna().any(axis=1).sum()

print("# of rows with an empty value in any column: ", num_rows_with_empty_value)
print("# of rows without an empty value in any column: ", len(df) - num_rows_with_empty_value)
print("Percentage of rows with an empty value in any column: ", num_rows_with_empty_value / len(df) * 100.0)

# remove the rows with an empty value in any column
df = df.dropna(subset=None)

df.to_csv('../tmp/complete_dataset.csv', quoting=csv.QUOTE_ALL, index=False)
