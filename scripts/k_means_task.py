import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

from util.data import read_complete_dataset

# Load data
df = read_complete_dataset()

# remove any columns that are not useful for clustering
df = df.drop([
	'Polygon_ID',
	'acq_date',
	'acq_time',
	# 'Neighbour',
	# 'Neighbour_frp',
	'CH_mean',
	'Neighbour_CH_mean'
], axis=1)

df['frp'] = df['frp'].apply(lambda x: sum(map(float, x.split(','))) / len(x.split(',')))

# standardize the data
scaler = StandardScaler()
data = scaler.fit_transform(df)

# set the number of clusters
num_clusters = 3

# perform clustering
kmeans = KMeans(n_clusters=num_clusters, random_state=42)
kmeans.fit(data)

# add cluster labels to the dataframe
df['cluster'] = kmeans.labels_

# create scatter plots of the data, colored by cluster
for i in range(data.shape[1]):
    for j in range(i+1, data.shape[1]):
        plt.scatter(data[:, i], data[:, j], c=df['cluster'], cmap='viridis')
        plt.xlabel(df.columns[i])
        plt.ylabel(df.columns[j])
        plt.title(f'Clusters ({df.columns[i]} vs {df.columns[j]})')
        plt.savefig(f'../tmp/clusters_{df.columns[i]}_vs_{df.columns[j]}.png')
        plt.show()
