# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping

# Check if GPU is available
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

# Load and preprocess data
def load_data(file_path):
    data = pd.read_csv(file_path)
    data.ffill(inplace=True)  # Updated to use ffill()
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
        y.append(scaled_data[i, 3])  # Assuming 'Close' is the target
    X, y = np.array(X), np.array(y).reshape(-1, 1)  # Reshape y to match model output
    train_size = int(len(X) * 0.8)
    return X[:train_size], X[train_size:], y[:train_size], y[train_size:]

def add_technical_indicators(data):
    data['SMA_20'] = data['Close'].rolling(window=20).mean()
    data['SMA_50'] = data['Close'].rolling(window=50).mean()
    data['RSI'] = compute_rsi(data['Close'])
    data.ffill(inplace=True)  # Updated to use ffill()
    return data

def compute_rsi(series, period=14):
    delta = series.diff(1)
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

# Build and train the model
def build_model(input_shape):
    model = Sequential()
    model.add(Input(shape=input_shape))
    model.add(LSTM(units=100, return_sequences=True))
    model.add(Dropout(0.3))
    model.add(LSTM(units=100, return_sequences=False))
    model.add(Dropout(0.3))
    model.add(Dense(units=input_shape[1]))
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

def train_model(model, X_train, y_train, epochs=100, batch_size=32):
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_split=0.2, callbacks=[early_stopping])

# Evaluate the model
def evaluate_model(model, X_test, y_test, scaler, original_data):
    predicted_prices = model.predict(X_test)
    
    # Create a placeholder array with the same shape as the scaled data
    predicted_prices_full = np.zeros((predicted_prices.shape[0], scaler.n_features_in_))
    
    # Insert the predicted prices into the appropriate column (assuming 'Close' is the 4th column)
    predicted_prices_full[:, 3] = predicted_prices[:, 0]
    
    # Inverse transform the entire array
    predicted_prices_full = scaler.inverse_transform(predicted_prices_full)
    
    # Extract the 'Close' prices
    predicted_prices = predicted_prices_full[:, 3]
    
    # Plot the results
    plt.plot(original_data['Close'][len(original_data) - len(y_test):], color='blue', label='Actual Prices')
    plt.plot(predicted_prices, color='red', label='Predicted Prices')
    plt.title('Stock Price Prediction')
    plt.xlabel('Time')
    plt.ylabel('Price')
    plt.legend()
    plt.show()
    
    # Export to Excel
    results = pd.DataFrame({
        'Date': original_data['Date'][len(original_data) - len(y_test):].values,
        'Actual': original_data['Close'][len(original_data) - len(y_test):].values,
        'Predicted': predicted_prices
    })
    results.to_excel('predictions.xlsx', index=False)

# Main function to run the steps
def main():
    file_path = 'SPY_1993_2024.csv'
    data = load_data(file_path)
    scaled_data, scaler = preprocess_data(data)
    X_train, X_test, y_train, y_test = create_datasets(scaled_data)
    model = build_model((X_train.shape[1], X_train.shape[2]))
    train_model(model, X_train, y_train)
    
    # Save the trained model
    model.save('stock_prediction_model.h5')
    
    evaluate_model(model, X_test, y_test, scaler, data)

if __name__ == "__main__":
    main()
