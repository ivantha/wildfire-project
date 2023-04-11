import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

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

# Split data into training and testing sets
train, test = train_test_split(df, test_size=0.2, random_state=42)

# Define X and y
X_train = train.drop('frp', axis=1)
y_train = train['frp']
X_test = test.drop('frp', axis=1)
y_test = test['frp']

# Set Random Forest parameters
params = {
    'n_estimators': 100,
    'max_depth': 20,
    'min_samples_split': 2,
    'min_samples_leaf': 1,
    'random_state': 42,
    'n_jobs': -1,
    'verbose': 1
}

# Train the model
model = RandomForestRegressor(**params)
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
