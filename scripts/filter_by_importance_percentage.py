import pandas as pd

from util.data import read_dataset, read_complete_dataset

# Load data
df = read_dataset()

feature_importance_df = pd.read_csv("../tmp/feature_importance.csv")

# Set the percentage of rows (columns) to keep based on importance
percentage_to_keep = 0.4  # 50%

# Calculate the number of rows to keep based on the given percentage
n_rows_to_keep = int(len(feature_importance_df) * percentage_to_keep)

# Sort the feature_importance_df by importance in descending order and keep the top rows
top_features_df = feature_importance_df.nlargest(n_rows_to_keep, "importance")

# Extract the filtered columns from the main DataFrame
filtered_columns = top_features_df["feature"].tolist()
filtered_main_df = df[filtered_columns]

# Add the target column 'frp' to the principal_df
filtered_main_df['frp'] = df['frp']

filtered_main_df.to_csv('../tmp/importance_dataset.csv', index=False)
