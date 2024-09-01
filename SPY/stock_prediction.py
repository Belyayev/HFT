import logging
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import joblib
import matplotlib.pyplot as plt

# Function to load and preprocess data
def load_and_preprocess_data(file_path):
    data = pd.read_csv(file_path)
    data['Date'] = pd.to_datetime(data['Date'])
    data.set_index('Date', inplace=True)
    
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(data[['Close']])
    
    return data, scaled_data, scaler

# Function to create datasets
def create_datasets(data):
    X = data[:-1]
    y = data[1:]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    return X_train, X_test, y_train, y_test

# Function to train the model
def train_model(X_train, y_train):
    model = RandomForestRegressor(n_estimators=100)
    model.fit(X_train, y_train)
    return model

# Function to evaluate the model
def evaluate_model(model, X_test, y_test, scaler=None, original_data=None):
    predictions = model.predict(X_test)
    if scaler:
        predictions = scaler.inverse_transform(predictions.reshape(-1, 1))
        y_test = scaler.inverse_transform(y_test.reshape(-1, 1))
    
    plt.figure(figsize=(14, 7))
    plt.plot(original_data.index[-len(y_test):], y_test, color='blue', label='Actual')
    plt.plot(original_data.index[-len(predictions):], predictions, color='red', label='Predicted')
    plt.title('Stock Price Prediction')
    plt.xlabel('Date')
    plt.ylabel('Stock Price')
    plt.legend()
    plt.show()
    
    mse = mean_squared_error(y_test, predictions)
    logging.info(f'Mean Squared Error: {mse}')
    return mse

# Main function to run the steps
def main():
    logging.info("Loading and preprocessing data...")
    file_path = 'SPY_1993_2024.csv'
    original_data, data, scaler = load_and_preprocess_data(file_path)
    
    logging.info("Creating datasets...")
    X_train, X_test, y_train, y_test = create_datasets(data)
    
    logging.info("Training model...")
    model = train_model(X_train, y_train)
    
    logging.info("Saving the trained model...")
    joblib.dump(model, 'stock_prediction_model.pkl')
    
    logging.info("Evaluating model...")
    evaluate_model(model, X_test, y_test, scaler, original_data)
    
    logging.info("Process completed.")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
