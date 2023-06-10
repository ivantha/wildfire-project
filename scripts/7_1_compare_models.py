import os
import glob

MODEL_LIST = [
    "lr",
    "dt",
    "rf",
    "xgb",
    "nn"
]

MODEL_OUTPUT_DIRECTORY_PATH = '../tmp/models/'

# list all text files in the directory
txt_files = glob.glob(os.path.join(MODEL_OUTPUT_DIRECTORY_PATH, '*.txt'))

# create a dictionary to hold all metrics
metrics = {}

# create a list to hold all unique metrics in their original order
all_metrics = []

# read each file
for txt_file in txt_files:
    with open(txt_file, 'r') as f:
        lines = f.readlines()
        # get the model name from the file name
        model_name = os.path.basename(txt_file).split('.')[0]
        # store metrics in the dictionary
        metrics[model_name] = [(line.strip().split(':')[0], line.strip().split(':')[1].strip()) for line in lines]
        # add the metrics to the list of all metrics if they are not already there
        for metric, _ in metrics[model_name]:
            if metric not in all_metrics:
                all_metrics.append(metric)

# define column width
col_width = 20

# print model names in the top row
print(f'{"Metrics":<{col_width}}', end='')
for model_name in MODEL_LIST:
    print(f'{model_name:<{col_width}}', end='')
print()

# print metrics and their values for each model
for metric in all_metrics:
    print(f'{metric:<{col_width}}', end='')
    for model_name in MODEL_LIST:
        print(f'{dict(metrics.get(model_name, [])).get(metric, "N/A"):<{col_width}}', end='')
    print()
