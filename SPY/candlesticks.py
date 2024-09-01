import pandas as pd
import plotly.graph_objects as go

# Load the data
file_path = 'predictions_with_dates_sorted.csv'
data = pd.read_csv(file_path)

# Ensure the date column is in datetime format
data['Date'] = pd.to_datetime(data['Date'])

# Create the candlestick chart for actual values
fig = go.Figure(data=[go.Candlestick(
    x=data['Date'],
    open=data['Actual_Open'],
    high=data['Actual_High'],
    low=data['Actual_Low'],
    close=data['Actual_Close'],
    name='Actual'
)])

# Add the predicted values as a line chart
fig.add_trace(go.Scatter(
    x=data['Date'],
    y=data['Predicted_Close'],
    mode='lines',
    name='Predicted Close',
    line=dict(color='blue')
))

# Add the predicted values as a line chart
fig.add_trace(go.Scatter(
    x=data['Date'],
    y=data['Predicted_Open'],
    mode='lines',
    name='Predicted Open',
    line=dict(color='green')
))

# Update layout
fig.update_layout(
    title='Stock Market Candlestick Chart with Predicted vs Actual Values',
    xaxis_title='Date',
    yaxis_title='Price',
    xaxis_rangeslider_visible=False
)

# Show the plot
fig.show()
