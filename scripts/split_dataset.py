import os

import pandas as pd

if __name__ == '__main__':
	f_path = '../dataset/features_array.csv'

	# Create tmp directory if doesnt exist
	if not os.path.exists('../tmp/dataset'):
		os.makedirs('../tmp/dataset')

	chunksize = 10 ** 6
	with pd.read_csv(f_path, sep="	", chunksize=chunksize) as reader:
		i = 0
		for chunk in reader:
			print(f'Processing chunk #{i}')
			chunk.to_csv(f'../tmp/dataset/chunk_{i}', index=False)
			i += 1
