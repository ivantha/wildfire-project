import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from util.data import read_dataset, read_complete_dataset

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
], axis=1, errors='ignore')

# Split data into training and testing sets
train, test = train_test_split(df, test_size=0.2, random_state=42)

# Standardize the training set
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(train.drop('frp', axis=1))
y_train = train['frp']

# Standardize the testing set using the same scaler
X_test_scaled = scaler.transform(test.drop('frp', axis=1))
y_test = test['frp']

# Define X and y
X_train = train.drop('frp', axis=1)
y_train = train['frp']
X_test = test.drop('frp', axis=1)
y_test = test['frp']

# Set XGBoost parameters
params = {
	'objective': 'reg:squarederror',
	'eval_metric': 'rmse',
	'n_estimators': 100,
	'max_depth': 40,
	# 'learning_rate': 0.01,
	'min_child_weight': 1,
	'subsample': 0.5,
	# 'colsample_bytree': 0.8,
	'seed': 42,
	'tree_method': 'gpu_hist',
	# 'n_jobs': -1,
	'verbosity': 1
}

# Train the model
model = xgb.XGBRegressor(**params)
model.fit(X_train, y_train)

# Make predictions on the testing set
y_pred = model.predict(X_test)

# Calculate evaluation metrics
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

# Print evaluation metrics
print('Mean Squared Error:', mse)
print('Root Mean Squared Error:', rmse)
print('R-squared:', r2)

# get feature importance
importance = model.feature_importances_
feature_importance_df = pd.DataFrame({'feature': X_train.columns, 'importance': importance})
feature_importance_df = feature_importance_df.sort_values('importance', ascending=False)

# sort the DataFrame by the 'importance' column in descending order
feature_importance_df = feature_importance_df.sort_values('importance', ascending=False)

feature_importance_df.to_csv('../tmp/feature_importance.csv', index=False)