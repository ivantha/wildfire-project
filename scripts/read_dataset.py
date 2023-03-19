import pandas as pd

from util.fun_time import timeit


@timeit
def read_dataset():
	df = pd.read_csv('dataset/features_array.csv', sep="	")
	print('Length of dataset: ', len(df))
	print('Columns:')
	print(df.columns)
	print('Head:')
	print(df.head())


if __name__ == '__main__':
	read_dataset()