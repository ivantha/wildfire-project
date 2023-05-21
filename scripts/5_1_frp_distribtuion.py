import pandas as pd
import matplotlib.pyplot as plt

# Load data
df = pd.read_parquet(f"../tmp/datasets/processed")

# df['frp'] = df['frp'].apply(lambda x: sum(map(float, x.split(','))) / len(x.split(',')))

# Select the 'frp' column
frp_col = df['frp']

# Create a figure with three subplots
fig, axs = plt.subplots(1, 2, figsize=(8, 4))

# Plot a histogram of the 'frp' column on the first subplot
axs[0].hist(frp_col, bins=50)
axs[0].set_xlabel('FRP')
axs[0].set_ylabel('Frequency')
axs[0].set_title('Histogram of FRP')

# Plot a box plot of the 'frp' column on the third subplot
axs[1].boxplot(frp_col)
axs[1].set_ylabel('FRP')
axs[1].set_title('Box Plot of FRP')

# Adjust the layout of the subplots to avoid overlapping
plt.tight_layout()

# Save the plot to disk as a PNG image
plt.savefig('../tmp/frp_distribution_plots.png')
