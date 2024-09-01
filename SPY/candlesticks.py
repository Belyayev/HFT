import pandas as pd
import plotly.graph_objects as go

# Load the data
file_path = 'CSV/new_predictions_with_dates.csv'
data = pd.read_csv(file_path)

# Ensure the date column is in datetime format
data['Date'] = pd.to_datetime(data['Date'])

# Offset the predicted dates by a small amount
data['Predicted_Date'] = data['Date'] + pd.Timedelta(days=0.3)

# Create the candlestick chart for actual values
fig = go.Figure(data=[go.Candlestick(
    x=data['Date'],
    open=data['Open'],
    high=data['High'],
    low=data['Low'],
    close=data['Close'],
    name='Actual'
)])

# Add the candlestick chart for predicted values
fig.add_trace(go.Candlestick(
    x=data['Predicted_Date'],
    open=data['Predicted_Open'],
    high=data['Predicted_High'],
    low=data['Predicted_Low'],
    close=data['Predicted_Close'],
    name='Predicted',
    increasing=dict(line=dict(color='darkgrey', width=1), fillcolor='lightgreen'),
    decreasing=dict(line=dict(color='darkgrey', width=1), fillcolor='lightcoral'),
    opacity=0.3
))

# Add the predicted values as a line chart
# fig.add_trace(go.Scatter(
#     x=data['Date'],
#     y=data['Predicted_Open'],
#     mode='lines',
#     name='Predicted Open',
#     line=dict(color='green')
# ))

# Update layout
fig.update_layout(
    title='Stock Market Candlestick Chart with Predicted vs Actual Values',
    xaxis_title='Date',
    yaxis_title='Price',
    xaxis_rangeslider_visible=False
)

# Show the plot
fig.show()
