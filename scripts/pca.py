import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from legacy.data import read_complete_dataset

# Load data
df = read_complete_dataset()

df['frp'] = df['frp'].apply(lambda x: sum(map(float, x.split(','))) / len(x.split(',')))

df = df.drop([
	'Polygon_ID',
	'acq_date',
	'acq_time',
	# 'Neighbour',
	# 'Neighbour_frp',
	'CH_mean',
	'Neighbour_CH_mean'
], axis=1)

# Separate the 'frp' target column from the feature columns
features_df = df.drop('frp', axis=1)
target_df = df['frp']

# Standardize the features (important for PCA)
scaler = StandardScaler()
scaled_features = scaler.fit_transform(features_df)

# Choose the number of components you want to keep
n_components = 50  # Change this to the desired number of components

# Apply PCA
pca = PCA(n_components=n_components)
principal_components = pca.fit_transform(scaled_features)

# Create a new DataFrame with the principal components
columns = [f'PC{i+1}' for i in range(n_components)]

principal_df = pd.DataFrame(data=principal_components, columns=columns)

# Add the target column 'frp' to the principal_df
principal_df['frp'] = target_df

# Save the resulting DataFrame to a CSV file
principal_df.to_csv('../tmp/pca.csv', index=False)
