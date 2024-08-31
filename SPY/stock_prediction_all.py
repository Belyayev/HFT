# stock_prediction.py

# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
import matplotlib.pyplot as plt
import tensorflow as tf

# Check if GPU is available
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

# Load and preprocess data
def load_data(file_path):
    data = pd.read_csv(file_path)
    data.fillna(method='ffill', inplace=True)
    return data

def preprocess_data(data):
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data[['Open', 'High', 'Low', 'Close', 'Volume']])
    return scaled_data, scaler

# Create training and test sets
def create_datasets(scaled_data, look_back=60):
    X, y = [], []
    for i in range(look_back, len(scaled_data)):
        X.append(scaled_data[i-look_back:i])
        y.append(scaled_data[i])  # Include all columns as targets
    X, y = np.array(X), np.array(y)
    train_size = int(len(X) * 0.8)
    return X[:train_size], X[train_size:], y[:train_size], y[train_size:]

# Build and train the model
def build_model(input_shape):
    model = Sequential()
    model.add(Input(shape=input_shape))  # Use Input layer to define the input shape
    model.add(LSTM(units=50, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(units=50, return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(units=input_shape[1]))  # Output layer with the same number of units as features
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

def train_model(model, X_train, y_train, epochs=50, batch_size=32):
    model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size)

# Evaluate the model
def evaluate_model(model, X_test, y_test, scaler, original_data):
    predicted_prices = model.predict(X_test)
    
    # Inverse transform the predicted prices
    predicted_prices = scaler.inverse_transform(predicted_prices)
    
    # Plot the results
    plt.figure(figsize=(14, 5))
    plt.plot(original_data['Close'][len(original_data) - len(y_test):], color='blue', label='Actual Close Prices')
    plt.plot(predicted_prices[:, 3], color='red', label='Predicted Close Prices')  # Assuming 'Close' is the 4th column
    plt.title('Stock Price Prediction')
    plt.xlabel('Time')
    plt.ylabel('Price')
    plt.legend()
    plt.show()

# Main function to run the steps
def main():
    file_path = 'SPY_1993_2024.csv'
    data = load_data(file_path)
    scaled_data, scaler = preprocess_data(data)
    X_train, X_test, y_train, y_test = create_datasets(scaled_data)
    model = build_model((X_train.shape[1], X_train.shape[2]))
    train_model(model, X_train, y_train)
    
    # Save the trained model
    model.save('stock_prediction_model_all_values.h5')
    
    evaluate_model(model, X_test, y_test, scaler, data)

if __name__ == "__main__":
    main()
