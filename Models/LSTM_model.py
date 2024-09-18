import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dropout, Dense
import datetime
import joblib 
import keras

# Memoization cache to store previous predictions
cache = {}

# Function to generate the next time step's features (day, month, year, etc.) based on current datetime.
def get_next_time_features(current_datetime):
    next_datetime = current_datetime + pd.Timedelta(minutes=5)  # Increment time by 5 minutes
    day = next_datetime.day
    month = next_datetime.month
    year = next_datetime.year
    day_of_week = next_datetime.weekday() + 1  # Monday=0, Sunday=6
    hour = next_datetime.hour
    minute = next_datetime.minute
    
    return [day, month, year, day_of_week, hour, minute], next_datetime

# Function to predict future values based on previous data
def predict_future_values(sequence_length, scaler_X, scaler_y, model, df, end_datetime):
    global cache

    predictions = []
    prediction_times = []

    # Collect the last available sequences from the dataset
    last_12_sequences = df[['day', 'month', 'year', 'day_of_week', 'hour', 'minute']].values[-sequence_length:]

    # Ensure there are enough initial sequences
    if len(last_12_sequences) < sequence_length:
        raise ValueError("Not enough data points to start predictions.")

    # Set current_datetime to the last available datetime in the dataset
    current_datetime = df['datetime'].values[-1]

    # Iteratively predict values until end_datetime is reached
    while current_datetime <= end_datetime:
        # Reshape the last sequences for prediction
        X_predict = np.array(last_12_sequences).reshape((1, sequence_length, 6))  # 6 features

        # Scale the input
        X_predict_scaled = scaler_X.transform(X_predict.reshape(-1, 6)).reshape(1, sequence_length, 6)

        # Predict the next value
        predicted_value_scaled = model.predict(X_predict_scaled)
        predicted_value = scaler_y.inverse_transform(predicted_value_scaled)[0][0]

        # Store the prediction and corresponding time
        predictions.append(predicted_value)
        prediction_times.append(current_datetime)

        # Generate the next time step's features dynamically
        next_time_features, current_datetime = get_next_time_features(current_datetime)

        # Update the sequence with the new time step's features
        last_12_sequences = np.vstack((last_12_sequences[1:], next_time_features))

    # Combine predictions and corresponding times into a DataFrame for clarity
    prediction_df = pd.DataFrame({'datetime': prediction_times, 'predicted_value': predictions})

    return prediction_df

# Load and preprocess your dataset
def load_and_preprocess_data(file_path):
    # Load dataset
    df = pd.read_csv(file_path)
    
    # Create a datetime column from day, month, year, hour, and minute
    df['datetime'] = pd.to_datetime(df[['year', 'month', 'day', 'hour', 'minute']])
    
    return df

# Build and compile the LSTM model with two LSTM layers of 64 units each
def build_lstm_model(sequence_length):
    model = Sequential()
    
    # First LSTM layer with 64 units, returning sequences for the next LSTM layer
    model.add(LSTM(units=64, return_sequences=True, input_shape=(sequence_length, 6)))  # 6 features (excluding value)
    model.add(Dropout(0.2))  # Regularization

    # Second LSTM layer with 64 units
    model.add(LSTM(units=64, return_sequences=False))
    model.add(Dropout(0.2))  # Regularization
    
    # Output layer for prediction
    model.add(Dense(units=1))  

    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

# Prepare the data for LSTM
def prepare_lstm_data(df, sequence_length):
    scaler_X = MinMaxScaler(feature_range=(0, 1))
    scaler_y = MinMaxScaler(feature_range=(0, 1))
    
    # Use all features except 'value' and 'datetime'
    X = df[['day', 'month', 'year', 'day_of_week', 'hour', 'minute']].values
    y = df['value'].values

    X_scaled = scaler_X.fit_transform(X)
    y_scaled = scaler_y.fit_transform(y.reshape(-1, 1))

    X_seq, y_seq = [], []
    for i in range(sequence_length, len(X_scaled)):
        X_seq.append(X_scaled[i-sequence_length:i])
        y_seq.append(y_scaled[i])

    X_seq, y_seq = np.array(X_seq), np.array(y_seq)
    
    return X_seq, y_seq, scaler_X, scaler_y

# Main function to train the model, predict future values, and save/load the model
def main():
    # Define paths for saving the model and scalers
    model_file = 'E:/APPLICATIONS/VSC/SIH/Models/lstm_model_1.h5'
    scaler_X_file = 'E:/APPLICATIONS/VSC/SIH/Models/main/scalers/scaler_x.pkl'
    scaler_y_file = 'E:/APPLICATIONS/VSC/SIH/Models/main/scalers/scaler_y.pkl'
    
    # Load and preprocess the data
    df = pd.read_csv('C:/Users/arya0/Downloads/updated_sldc_data_without_time.csv')
    df['datetime'] = pd.to_datetime(df[['year', 'month', 'day', 'hour', 'minute']])
    
    # Define sequence length
    sequence_length = 6

    # Prepare the dataset for LSTM
    X_train, y_train, scaler_X, scaler_y = prepare_lstm_data(df, sequence_length)

    # Load the model if it exists, else build and train a new one
    if os.path.exists(model_file):
        model = load_model(model_file)
        print("Model loaded from disk.")
        scaler_X = joblib.load(scaler_X_file)
        scaler_y = joblib.load(scaler_y_file)
        print("Scalers loaded from disk.")
    else:
        # Build the LSTM model
        model = build_lstm_model(sequence_length)

        # Train the LSTM model
        model.fit(X_train, y_train, epochs=50, batch_size=32)

        # Save the model
        model.save(model_file)
        print("Model trained and saved to disk.")
        
        # Save the scalers
        joblib.dump(scaler_X, scaler_X_file)
        joblib.dump(scaler_y, scaler_y_file)
        print("Scalers saved to disk.")

    # Predict all values for 1st September 2024
    start_datetime = pd.to_datetime('2024-09-01 00:00')
    end_datetime = pd.to_datetime('2024-09-01 23:55')

    # Predict future values with corresponding times
    prediction_df = predict_future_values(sequence_length, scaler_X, scaler_y, model, df, end_datetime)

    # Display the predictions with corresponding times in the format "Time: Load"
    print(f"Predictions for 1st September 2024:\n")
    for idx, row in prediction_df.iterrows():
        print(f"Time: {row['datetime']}, Load: {row['predicted_value']}")

if __name__ == '__main__':
    main()
