# app.py
from flask import Flask, render_template, request, jsonify
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import io
import base64
from datetime import datetime, timedelta
import os
import tensorflow as tf

app = Flask(__name__)

# Create a simple LSTM model instead of loading a custom one
def create_simple_model(input_shape):
    model = tf.keras.Sequential([
        tf.keras.layers.LSTM(64, return_sequences=True, input_shape=input_shape),
        tf.keras.layers.LSTM(32),
        tf.keras.layers.Dense(16, activation='relu'),
        tf.keras.layers.Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    return model

# Global variables
timesteps = 10
n_features = 1
model = create_simple_model((timesteps, n_features))

# Generate synthetic data for demonstration
def generate_demo_data():
    # Create date range
    end_date = datetime.now().date()
    start_date = end_date - timedelta(days=100)
    date_range = pd.date_range(start=start_date, end=end_date)
    
    # Generate prices with some trend and seasonality
    base_price = 3500  # Base price
    trend = np.linspace(0, 500, len(date_range))  # Upward trend
    seasonality = 200 * np.sin(np.linspace(0, 6*np.pi, len(date_range)))  # Seasonal component
    noise = np.random.normal(0, 100, len(date_range))  # Random noise
    
    prices = base_price + trend + seasonality + noise
    prices = np.maximum(prices, 1000)  # Ensure minimum price
    
    # Create the dataframe
    data = {
        'Arrival_Date': date_range,
        'Modal_Price': prices,
        'Commodity_Code': [13] * len(date_range),
        'Commodity': ['Soyabean'] * len(date_range)
    }
    
    # Add more commodities
    commodities = [
        {'code': 1, 'name': 'Rice', 'base_price': 2500, 'factor': 0.9},
        {'code': 2, 'name': 'Wheat', 'base_price': 2000, 'factor': 1.1},
        {'code': 3, 'name': 'Corn', 'base_price': 1800, 'factor': 1.2},
        {'code': 4, 'name': 'Potato', 'base_price': 1200, 'factor': 0.8}
    ]
    
    for commodity in commodities:
        commodity_prices = commodity['base_price'] + commodity['factor'] * (trend + seasonality + np.random.normal(0, 100, len(date_range)))
        commodity_prices = np.maximum(commodity_prices, 800)
        
        temp_data = {
            'Arrival_Date': date_range,
            'Modal_Price': commodity_prices,
            'Commodity_Code': [commodity['code']] * len(date_range),
            'Commodity': [commodity['name']] * len(date_range)
        }
        
        # Append to existing data
        for key in data:
            data[key] = np.append(data[key], temp_data[key])
    
    return pd.DataFrame(data)

# Load or generate data
def get_data():
    try:
        if os.path.exists("historical_data.csv"):
            return pd.read_csv("historical_data.csv")
        else:
            # Generate and save demo data
            data = generate_demo_data()
            data.to_csv("demo_data.csv", index=False)
            return data
    except Exception as e:
        print(f"Error loading data: {e}")
        return generate_demo_data()

# Get dataset stats for scaling
def get_scaling_params(data, commodity_code):
    commodity_data = data[data['Commodity_Code'] == commodity_code]
    min_price = commodity_data['Modal_Price'].min()
    max_price = commodity_data['Modal_Price'].max()
    return min_price, max_price

# Get input sequence for a specific commodity
def get_input_sequence(data, commodity_code):
    # Filter data for the selected commodity
    commodity_data = data[data['Commodity_Code'] == commodity_code]
    
    # Sort by date
    commodity_data = commodity_data.sort_values('Arrival_Date')
    
    # Get the last timesteps prices
    latest_prices = commodity_data['Modal_Price'].tail(timesteps).values
    
    # Get scaling parameters
    min_price, max_price = get_scaling_params(data, commodity_code)
    
    # Scale the prices
    scaled_prices = (latest_prices - min_price) / (max_price - min_price)
    
    # Reshape for LSTM input: [samples, timesteps, features]
    return scaled_prices.reshape(1, timesteps, n_features), min_price, max_price

# Generate predictions for the next 7 days
def predict_future_prices(data, commodity_code, days=7):
    # Get input sequence and scaling parameters
    sequence, min_price, max_price = get_input_sequence(data, commodity_code)
    
    # Initialize array to store predictions
    predictions = np.zeros(days)
    current_sequence = sequence.copy()
    
    # Make predictions one day at a time
    for i in range(days):
        # Predict next day
        next_price = model.predict(current_sequence, verbose=0)[0, 0]
        predictions[i] = next_price
        
        # Update sequence for next prediction
        current_sequence = np.roll(current_sequence, -1, axis=1)
        current_sequence[0, -1, 0] = next_price
    
    # Scale predictions back to original range
    predictions = predictions * (max_price - min_price) + min_price
    
    return predictions

# Create a plot and return as base64 encoded image
def create_prediction_plot(dates, prices, commodity_name):
    plt.figure(figsize=(10, 5))
    plt.plot(dates, prices, marker="o", linestyle="-", color="blue", 
             label=f"Predicted Prices for {commodity_name}")
    
    # Formatting
    plt.xlabel("Date")
    plt.ylabel("Price (₹)")
    plt.title(f"Predicted Prices for {commodity_name} for Next 7 Days")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend()
    
    # Format y-axis as currency
    plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'₹{int(x):,}'))
    
    # Rotate date labels for better readability
    plt.xticks(rotation=45)
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
        # Load data
        data = get_data()
        
        # Get commodity details from request or use default
        commodity_code = int(request.json.get("commodity_code", 13))
        commodity_name = request.json.get("commodity_name", "Soyabean")
        
        # Generate future dates
        last_date = datetime.now().date()
        future_dates = [(last_date + timedelta(days=i+1)).strftime('%Y-%m-%d') for i in range(7)]
        
        # Make predictions
        predictions = predict_future_prices(data, commodity_code)
        
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
        print(f"Error in prediction: {str(e)}")
        return jsonify({
            "success": False,
            "error": str(e)
        })

@app.route("/commodities")
def get_commodities():
    try:
        # Load data
        data = get_data()
        
        # Get unique commodities
        commodities = data[["Commodity_Code", "Commodity"]].drop_duplicates().to_dict('records')
        
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
    app.run(debug=True)
