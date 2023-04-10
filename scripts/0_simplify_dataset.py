import csv
import json

import numpy as np
import pandas as pd

df = pd.read_csv('../dataset/features_array.csv', sep="	")

# find columns of type float64 that can be converted to float32
convertible_cols = []
non_convertible_cols = []

float32_min = np.finfo(np.float32).min
float32_max = np.finfo(np.float32).max

float64_cols = df.select_dtypes(include=['float64']).columns
for col in float64_cols:
	if df[col].min() >= float32_min and df[col].max() <= float32_max:
		convertible_cols.append(col)
	else:
		non_convertible_cols.append(col)

print(f"Columns of type float64 that can be converted to float32: {convertible_cols}")
print(f"Columns of type float64 that cannot be converted to float32: {non_convertible_cols}")

# for col in convertible_cols:
#     df[col] = df[col].astype('float32')

# save the new dataframe as a CSV file
df.to_csv('../tmp/dataset.csv', quoting=csv.QUOTE_ALL, index=False)

# save list of columns with their column types to a json file.
columns_info = {}

for col in df.columns:
	columns_info[col] = str(df[col].dtype)

with open('../tmp/column_info.json', 'w') as fp:
	json.dump(columns_info, fp)
