import os
import time

import numpy as np
import pandas as pd
import tensorflow as tf
from keras import Sequential
from keras.callbacks import EarlyStopping
from keras.layers import Dense
from keras.metrics import MeanAbsolutePercentageError, MeanSquaredError, RootMeanSquaredError, MeanAbsoluteError
from keras.optimizers import Adam
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def check_gpu_availability():
    physical_devices = tf.config.list_physical_devices('GPU')
    if len(physical_devices) > 0:
        print("GPU is available")
        tf.config.experimental.set_memory_growth(physical_devices[0], True)
    else:
        print("GPU is not available. Using CPU instead.")


def main():
    start_time = time.time()

    check_gpu_availability()

    # Load data
    df = pd.read_parquet(f"../tmp/datasets/processed")

    df = df.drop([
        'Polygon_ID',
    ], axis=1)

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

    # Define the deep neural network model
    model = Sequential()
    # model.add(Dense(512, activation='relu', input_shape=(X_train_scaled.shape[1],)))
    # model.add(Dense(256, activation='relu', input_shape=(X_train_scaled.shape[1],)))
    model.add(Dense(128, activation='relu', input_shape=(X_train_scaled.shape[1],)))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(1))

    # Compile the model
    model.compile(
        optimizer=Adam(lr=0.001),
        loss='mean_squared_error',
        metrics=[
            MeanSquaredError(name='mse'),
            RootMeanSquaredError(name='rmse'),
            MeanAbsoluteError(name='mae'),
            MeanAbsolutePercentageError(name='mape'),
            # R2Score()
        ]
    )

    # Define callbacks
    early_stopping = EarlyStopping(monitor='val_loss', patience=10)

    # Train the model
    history = model.fit(
        X_train_scaled,
        y_train,
        epochs=2000,
        batch_size=32,
        verbose=1,
        validation_data=(X_test_scaled, y_test),
        callbacks=[early_stopping]
    )

    # Make predictions on the testing set
    y_pred = model.predict(X_test_scaled).flatten()

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
    with open('../tmp/models/nn.txt', 'w') as f:
        f.write(f'MSE: {mse:,.2f}\n')
        f.write(f'RMSE: {rmse:,.2f}\n')
        f.write(f'MAE: {mae:,.2f}\n')
        f.write(f'MAPE: {mape:,.2f}\n')
        f.write(f'R2: {r2:,.2f}\n')
        f.write(f'Execution time: {execution_time:,.2f} seconds\n')
        f.flush()


if __name__ == '__main__':
    main()
