import os
import time

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def main():
    start_time = time.time()

    # Load data
    df = pd.read_parquet(f"../tmp/datasets/processed")

    df = df.drop([
        'Polygon_ID',
    ], axis=1, errors='ignore')

    # Split data into training and testing sets
    train, test = train_test_split(df, test_size=0.1, random_state=42)

    # Define X and y
    X_train = train.drop('frp', axis=1)
    y_train = train['frp']
    X_test = test.drop('frp', axis=1)
    y_test = test['frp']

    # Scale the data
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Set XGBoost parameters
    params = {
        'objective': 'reg:squarederror',
        'eval_metric': 'rmse',
        'n_estimators': 10,
        'max_depth': 30,
        # 'learning_rate': 0.01,
        'min_child_weight': 1,
        'subsample': 0.5,
        # 'colsample_bytree': 0.8,
        'seed': 42,
        # 'tree_method': 'gpu_hist',
        'n_jobs': -1,
        'verbosity': 1
    }

    # Train the model
    model = xgb.XGBRegressor(**params)
    model.fit(X_train_scaled, y_train)

    # Make predictions on the testing set
    y_pred = model.predict(X_test_scaled)

    # Calculate evaluation metrics
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
    r2 = r2_score(y_test, y_pred)

    # Calculate execution time
    execution_time = time.time() - start_time

    # Print evaluation metrics
    print(f'MSE: {mse:,.2f}')
    print(f'RMSE: {rmse:,.2f}')
    print(f'MAE: {mae:,.2f}')
    print(f'MAPE: {mape:,.2f}')
    print(f'R2: {r2:,.2f}')
    print(f'Execution time: {execution_time:,.2f} seconds')

    # Write evaluation metrics to a text file
    os.makedirs('../tmp/models', exist_ok=True)
    with open('../tmp/models/xgb.txt', 'w') as f:
        f.write(f'MSE: {mse:,.2f}\n')
        f.write(f'RMSE: {rmse:,.2f}\n')
        f.write(f'MAE: {mae:,.2f}\n')
        f.write(f'MAPE: {mape:,.2f}\n')
        f.write(f'R2: {r2:,.2f}\n')
        f.write(f'Execution time: {execution_time:,.2f} seconds\n')
        f.flush()


if __name__ == '__main__':
    main()
