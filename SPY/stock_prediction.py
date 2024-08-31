import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from tensorflow.keras.callbacks import EarlyStopping
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
import matplotlib.pyplot as plt
import tensorflow as tf

# Check if GPU is available
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

# Load and preprocess data
def load_and_preprocess_data(file_path):
    data = pd.read_csv(file_path)
    data = add_technical_indicators(data)
    data.dropna(inplace=True)  # Ensure no missing values
    return data

def preprocess_data(data):
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data[['Open', 'High', 'Low', 'Close', 'Volume']])
    return scaled_data, scaler

# Create training and test sets
def create_datasets(data, look_back=60):
    X, y = [], []
    for i in range(look_back, len(data)):
        X.append(data.iloc[i-look_back:i].values)
        y.append(data.iloc[i, data.columns.get_loc('Close')])
    X, y = np.array(X), np.array(y)
    X = X.reshape(X.shape[0], -1)  # Flatten the input for Random Forest
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

def train_model(X_train, y_train):
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model

# Evaluate the model
def evaluate_model(model, X_test, y_test, scaler, original_data):
    predictions = model.predict(X_test)
    mse = mean_squared_error(y_test, predictions)
    print(f'Mean Squared Error: {mse}')
    
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
    
    results.to_csv('predictions.csv', index=False)

# Main function to run the steps
def main():
    file_path = 'SPY_1993_2024.csv'
    data = load_and_preprocess_data(file_path)
    X_train, X_test, y_train, y_test = create_datasets(data)
    model = train_model(X_train, y_train)
    
    # Save the trained model
    import joblib
    joblib.dump(model, 'stock_prediction_model.pkl')
    
    evaluate_model(model, X_test, y_test)

if __name__ == "__main__":
    main()
