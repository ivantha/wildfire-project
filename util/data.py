import json

import pandas as pd

from util.timer import timeit


@timeit
def read_dataset():
	# load columns_info.json
	with open('../tmp/column_info.json', 'r') as f:
		column_info = json.load(f)

	# create dtype dictionary for pd.read_csv()
	dtype_dict = {col: dtype for col, dtype in column_info.items()}

	# load the csv file with the specified dtype
	df = pd.read_csv('../tmp/dataset.csv', dtype=dtype_dict)

	return df


@timeit
def read_complete_dataset():
	# load columns_info.json
	with open('../tmp/column_info.json', 'r') as f:
		column_info = json.load(f)

	# create dtype dictionary for pd.read_csv()
	dtype_dict = {col: dtype for col, dtype in column_info.items()}

	# load the csv file with the specified dtype
	df = pd.read_csv('../tmp/complete_dataset.csv', dtype=dtype_dict)

	return df


def read_importance_dataset():
	# load columns_info.json
	with open('../tmp/column_info.json', 'r') as f:
		column_info = json.load(f)

	# create dtype dictionary for pd.read_csv()
	dtype_dict = {col: dtype for col, dtype in column_info.items()}

	# load the csv file with the specified dtype
	df = pd.read_csv('../tmp/importance_dataset.csv', dtype=dtype_dict)

	return df


@timeit
def read_pca():
	df = pd.read_csv('../tmp/pca.csv')

	return df