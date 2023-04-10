import pandas as pd

from util.data import read_dataset

# Load data
df = read_dataset()

# check for duplicates
df.duplicated()
