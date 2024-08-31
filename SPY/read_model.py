from tensorflow.keras.models import load_model

# Load the model
model = load_model('stock_prediction_model.h5')

# Print the model architecture
model.summary()

# Access the model weights
weights = model.get_weights()
