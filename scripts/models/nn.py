import numpy as np
import pandas as pd
from keras import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
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
    model.add(Dense(128, activation='relu', input_shape=(X_train_scaled.shape[1],)))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(1))

    # Compile the model
    model.compile(optimizer=Adam(lr=0.001), loss='mean_squared_error')

    # Train the model
    model.fit(X_train_scaled, y_train, epochs=100, batch_size=32, verbose=1)

    # Make predictions on the testing set
    y_pred = model.predict(X_test_scaled).flatten()

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
