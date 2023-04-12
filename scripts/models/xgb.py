import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from util.timer import timeit


@timeit
def main():
    # Load data
    df = pd.read_parquet(f"../../tmp/datasets/tiny")

    df['frp'] = df['frp'].apply(lambda x: sum(map(float, x.split(','))) / len(x.split(',')))

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
        'tree_method': 'gpu_hist',
        # 'n_jobs': -1,
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
    r2 = r2_score(y_test, y_pred)

    # Print evaluation metrics
    print('Mean Squared Error:', mse)
    print('Root Mean Squared Error:', rmse)
    print('Mean Absolute Error:', mae)
    print('R-squared:', r2)


if __name__ == '__main__':
    main()
