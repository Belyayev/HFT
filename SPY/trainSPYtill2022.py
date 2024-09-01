import pandas as pd
from ta.trend import SMAIndicator, MACD
from ta.momentum import RSIIndicator
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import pickle

# Load the data
file_path = 'CSV/SPY_till_2022.csv'
data = pd.read_csv(file_path)

# Ensure the date column is in datetime format
data['Date'] = pd.to_datetime(data['Date'])

# Calculate technical indicators
data['SMA20'] = SMAIndicator(close=data['Close'], window=20).sma_indicator()
data['SMA50'] = SMAIndicator(close=data['Close'], window=50).sma_indicator()
data['RSI'] = RSIIndicator(close=data['Close'], window=14).rsi()
data['MACD'] = MACD(close=data['Close']).macd()

# Drop rows with NaN values
data.dropna(inplace=True)

# Features and target
features = data[['Date', 'Open', 'High', 'Low', 'Close', 'SMA20', 'SMA50', 'RSI', 'MACD', 'Volume']]
target = data[['Open', 'High', 'Low', 'Close']]

# Split the data
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

# Initialize the model
model = RandomForestRegressor(n_estimators=100, random_state=42)

# Train the model
model.fit(X_train.drop(columns=['Date']), y_train)

# Save the model to disk
with open('Models/random_forest_model_SPY.pkl', 'wb') as file:
    pickle.dump(model, file)

print("Model saved to random_forest_model_SPY.pkl")
