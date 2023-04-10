import pandas as pd

from util.data import read_dataset

# Load data
df = read_dataset()

# Calculate the correlations between the features
correlations = df.corr()

# Stack the correlations into a series
corr_series = correlations.stack()

# Drop the self-correlations (i.e., where the feature is correlated with itself)
corr_series = corr_series[corr_series.index.get_level_values(0) != corr_series.index.get_level_values(1)]

# Sort the correlations in descending order
corr_series_sorted = corr_series.sort_values(ascending=False)

# Create a DataFrame with the sorted correlations and feature pairs
corr_df = pd.DataFrame({'Feature 1': corr_series_sorted.index.get_level_values(0),
                        'Feature 2': corr_series_sorted.index.get_level_values(1),
                        'Correlation': corr_series_sorted.values})

# Save the output to a CSV file
corr_df.to_csv('../tmp/corr_list.csv', index_label='Rank')
