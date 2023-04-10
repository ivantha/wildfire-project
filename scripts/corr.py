import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from util.data import read_dataset

# Load data
df = read_dataset()

# calculate the correlation matrix
corr_matrix = df.corr()

# plot the correlation matrix using seaborn
sns.set(style="white")
fig, ax = plt.subplots(figsize=(100, 100))
sns.heatmap(corr_matrix, annot=True, fmt=".2f", linewidths=.5, cmap="coolwarm", ax=ax)

# save the plot as a high-resolution image
fig.savefig('../tmp/correlation_matrix.png', dpi=300)
