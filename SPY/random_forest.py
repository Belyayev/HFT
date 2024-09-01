import pandas as pd
from ta.trend import SMAIndicator, MACD
from ta.momentum import RSIIndicator
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

# Load the data
file_path = 'SPY_1993_2024.csv'
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
features = data[['Date', 'Open', 'High', 'Low', 'Close', 'SMA20', 'SMA50', 'RSI', 'MACD']]
target = data[['Open', 'High', 'Low', 'Close']]

# Split the data
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

# Initialize the model
model = RandomForestRegressor(n_estimators=100, random_state=42)

# Train the model
model.fit(X_train.drop(columns=['Date']), y_train)

# Make predictions
predictions = model.predict(X_test.drop(columns=['Date']))

# Create a DataFrame with actual and predicted values
results = X_test.copy()
results['Actual_Open'] = y_test['Open'].values
results['Predicted_Open'] = predictions[:, 0]
results['Actual_High'] = y_test['High'].values
results['Predicted_High'] = predictions[:, 1]
results['Actual_Low'] = y_test['Low'].values
results['Predicted_Low'] = predictions[:, 2]
results['Actual_Close'] = y_test['Close'].values
results['Predicted_Close'] = predictions[:, 3]

# Save to CSV
results.to_csv('predictions_with_dates.csv', index=False)

print("Predictions saved to predictions_with_dates.csv")
