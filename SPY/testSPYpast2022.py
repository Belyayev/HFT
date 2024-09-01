import pandas as pd
from ta.trend import SMAIndicator, MACD
from ta.momentum import RSIIndicator
import pickle

# Load the saved model
with open('Models/random_forest_model_SPY.pkl', 'rb') as file:
    model = pickle.load(file)

# Load new data
new_data_path = 'CSV/SPY_start_2022.csv'
new_data = pd.read_csv(new_data_path)

# Ensure the date column is in datetime format
new_data['Date'] = pd.to_datetime(new_data['Date'])

# Calculate technical indicators
new_data['SMA20'] = SMAIndicator(close=new_data['Close'], window=20).sma_indicator()
new_data['SMA50'] = SMAIndicator(close=new_data['Close'], window=50).sma_indicator()
new_data['RSI'] = RSIIndicator(close=new_data['Close'], window=14).rsi()
new_data['MACD'] = MACD(close=new_data['Close']).macd()

# Drop rows with NaN values
new_data.dropna(inplace=True)

# Features for prediction
new_features = new_data[['Date', 'Open', 'High', 'Low', 'Close', 'SMA20', 'SMA50', 'RSI', 'MACD', 'Volume']]

# Make predictions
new_predictions = model.predict(new_features.drop(columns=['Date']))

# Create a DataFrame with actual and predicted values
new_results = new_features.copy()
new_results['Predicted_Open'] = new_predictions[:, 0]
new_results['Predicted_High'] = new_predictions[:, 1]
new_results['Predicted_Low'] = new_predictions[:, 2]
new_results['Predicted_Close'] = new_predictions[:, 3]

# Save to CSV
new_results.to_csv('CSV/new_predictions_with_dates.csv', index=False)

print("New predictions saved to new_predictions_with_dates.csv")
