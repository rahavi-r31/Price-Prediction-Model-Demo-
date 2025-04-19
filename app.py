# app.py
from flask import Flask, render_template, request, jsonify
import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import io
import base64
from datetime import datetime, timedelta
import os

app = Flask(__name__)

# Load model on startup
model = tf.keras.models.load_model("model.h5")

# Global variables - to be populated from your dataset
min_price = 0
max_price = 10000  # Default values, should be replaced with actual min/max
timesteps = 20  # Adjust based on your model architecture
n_features = 8  # Adjust based on your model architecture

# Load historical data and update min/max prices
def load_historical_data():
    global min_price, max_price
    try:
        # Replace with your actual data loading code
        historical_data = pd.read_csv("historical_data.csv")
        min_price = historical_data["Modal_Price"].min()
        max_price = historical_data["Modal_Price"].max()
        return historical_data
    except Exception as e:
        print(f"Error loading historical data: {e}")
        # Return dummy data for demo if file doesn't exist
        return create_dummy_data()

def create_dummy_data():
    """Create dummy data for demo purposes if real data is not available"""
    dates = pd.date_range(start='2024-01-01', periods=100)
    data = {
        'Arrival_Date': dates,
        'Modal_Price': np.random.uniform(3000, 5000, 100),
        'Commodity_Code': [13] * 100,
        'Commodity': ['Soyabean'] * 100
    }
    return pd.DataFrame(data)

# Function to get most recent test sequence
def get_recent_sequence():
    # Replace this with code to get your actual test sequence
    # This is a placeholder that creates random data
    return np.random.random((timesteps, n_features))

# Generate predictions for the next 7 days
def predict_future_prices(input_sequence, days=7):
    future_predictions = []
    
    # Make a copy of the input sequence to avoid modifying the original
    current_sequence = input_sequence.copy()
    
    for _ in range(days):
        # Reshape for model input (batch_size, timesteps, features)
        current_sequence_reshaped = current_sequence.reshape(1, timesteps, n_features)
        
        # Predict next value
        next_prediction = model.predict(current_sequence_reshaped, verbose=0)
        
        # Extract scalar value
        next_prediction_value = next_prediction.item()
        
        # Store prediction
        future_predictions.append(next_prediction_value)
        
        # Update sequence for next iteration (remove oldest, add newest)
        current_sequence = np.roll(current_sequence, -1, axis=0)
        current_sequence[-1, -1] = next_prediction_value  # Assume last feature is the target
    
    # Convert predictions back to original scale
    future_predictions = np.array(future_predictions).reshape(-1, 1)
    future_predictions_real = future_predictions * (max_price - min_price) + min_price
    
    # Ensure no negative or NaN values
    future_predictions_real = np.nan_to_num(np.maximum(future_predictions_real, 0))
    
    return future_predictions_real.flatten()

# Create a plot and return as base64 encoded image
def create_prediction_plot(dates, prices, commodity_name):
    plt.figure(figsize=(10, 5))
    plt.plot(dates, prices, marker="o", linestyle="dashed", color="red", 
             label=f"Predicted Prices for {commodity_name} (₹)")
    plt.xlabel("Date")
    plt.ylabel("Price (₹)")
    plt.title(f"Predicted Prices for {commodity_name} for Next 7 Days")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    
    # Save plot to a bytes buffer
    buffer = io.BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    
    # Convert to base64 for embedding in HTML
    image_base64 = base64.b64encode(buffer.read()).decode('utf-8')
    plt.close()
    
    return image_base64

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Get the most recent sequence for prediction
        input_sequence = get_recent_sequence()
        
        # Get commodity details from request or use default
        commodity_code = request.json.get("commodity_code", 13)
        commodity_name = request.json.get("commodity_name", "Soyabean")
        
        # Generate future dates
        last_date = datetime.now().date()
        future_dates = [(last_date + timedelta(days=i+1)).strftime('%Y-%m-%d') for i in range(7)]
        
        # Make predictions
        predictions = predict_future_prices(input_sequence)
        
        # Create plot
        plot_img = create_prediction_plot(future_dates, predictions, commodity_name)
        
        # Create prediction dataframe for table display
        prediction_data = {
            "dates": future_dates,
            "prices": [round(float(p), 2) for p in predictions],
            "commodity_code": commodity_code,
            "commodity_name": commodity_name
        }
        
        return jsonify({
            "success": True,
            "prediction_data": prediction_data,
            "plot": plot_img
        })
    
    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        })

@app.route("/commodities")
def get_commodities():
    try:
        # Load historical data
        historical_data = load_historical_data()
        
        # Get unique commodities
        commodities = historical_data[["Commodity_Code", "Commodity"]].drop_duplicates().to_dict('records')
        
        return jsonify({
            "success": True,
            "commodities": commodities
        })
    
    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        })

if __name__ == "__main__":
    # Make sure to load data before starting the app
    historical_data = load_historical_data()
    app.run(debug=True)
