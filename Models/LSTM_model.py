import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dropout, Dense 
from sklearn.metrics import mean_squared_error, mean_absolute_error
import datetime 
import keras
import joblib


# Load the dataset
df = pd.read_csv('C:/Users/arya0/OneDrive/Desktop/SLDC_Data_with_Day_of_Week.csv')

# Convert 'time' to an integer hour and minute values (assuming time is in %H:%M format)
df['hour'] = pd.to_datetime(df['time'], format='%H:%M').dt.hour
df['minute'] = pd.to_datetime(df['time'], format='%H:%M').dt.minute

# Use the columns 'year', 'month', 'day', 'hour', 'minute', and 'day_of_week' as features
features = df[['year', 'month', 'day', 'hour', 'minute', 'day_of_week']].values
target = df['value'].values

# Scaling the features and target
# Scaling the features and target
scaler_X = MinMaxScaler()
scaler_y = MinMaxScaler()

# Fit the scaler on all 6 features: 'year', 'month', 'day', 'hour', 'minute', 'day_of_week'
X_scaled = scaler_X.fit_transform(features)
y_scaled = scaler_y.fit_transform(target.reshape(-1, 1))

# Save the scaler for later use in prediction
joblib.dump(scaler_X, 'E:/APPLICATIONS/VSC/SIH/Models/main/scalers/scaler_X.pkl')  # Update the path to save the scaler
joblib.dump(scaler_y, 'E:/APPLICATIONS/VSC/SIH/Models/main/scalers/scaler_y.pkl')

# Continue with your model training...


# Set the sequence length to 12 (you can adjust this)
sequence_length = 12

# Function to create sequences for training
def create_sequences(X, y, seq_length):
    X_seq = []
    y_seq = []
    for i in range(len(X) - seq_length):
        X_seq.append(X[i:i+seq_length])
        y_seq.append(y[i+seq_length])
    return np.array(X_seq), np.array(y_seq)

# Create sequences
X_seq, y_seq = create_sequences(X_scaled, y_scaled, sequence_length)

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_seq, y_seq, test_size=0.2, random_state=42)

# Path to save the model
model_path = 'lstm_load_prediction_model_with_seq6.h5'

# Check if the model already exists
if os.path.exists(model_path):
    # Load the saved model
    model = load_model(model_path)
    print("Model loaded from disk.")
else:
    # Initialize the Sequential model
    model = Sequential()

    # First LSTM layer
    model.add(LSTM(units=64, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])))
    model.add(Dropout(0.2))

    # Second LSTM layer
    model.add(LSTM(units=64, return_sequences=False))
    model.add(Dropout(0.2))

    # Output layer
    model.add(Dense(units=1))

    # Compile the model
    model.compile(optimizer='adam', loss='mean_squared_error')

    # Train the model
    model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test), verbose=1)

    # Save the trained model
    model.save(model_path)
    print("Model trained and saved to disk.") 
# real_values = {
#     "00:00": 6072.950, "00:05": 6044.530, "00:10": 6022.250, "00:15": 5980.490,
#     "00:20": 5956.720, "00:25": 5923.450, "00:30": 5918.460, "00:35": 5873.230,
#     "00:40": 5844.400, "00:45": 5794.770, "00:50": 5771.530, "00:55": 5743.110,
#     "01:00": 5715.550, "01:05": 5686.970, "01:10": 5660.640, "01:15": 5641.690,
#     "01:20": 5609.380, "01:25": 5570.010, "01:30": 5549.360, "01:35": 5504.070,
#     "01:40": 5466.210, "01:45": 5443.290, "01:50": 5433.520, "01:55": 5413.280,
#     "02:00": 5397.380, "02:05": 5372.940, "02:10": 5344.180, "02:15": 5309.340,
#     "02:20": 5305.670, "02:25": 5298.440, "02:30": 5256.220, "02:35": 5228.050,
#     "02:40": 5209.670, "02:45": 5181.730, "02:50": 5164.760, "02:55": 5146.980,
#     "03:00": 5115.920, "03:05": 5094.020, "03:10": 5077.020, "03:15": 5057.690,
#     "03:20": 5018.240, "03:25": 5003.210, "03:30": 4983.070, "03:35": 4965.710,
#     "03:40": 4949.660, "03:45": 4923.440, "03:50": 4908.980, "03:55": 4888.230,
#     "04:00": 4865.520, "04:05": 4839.640, "04:10": 4838.920, "04:15": 4828.510,
#     "04:20": 4830.990, "04:25": 4794.890, "04:30": 4770.610, "04:35": 4749.450,
#     "04:40": 4752.930, "04:45": 4733.580, "04:50": 4720.590, "04:55": 4710.990,
#     "05:00": 4690.910, "05:05": 4672.360, "05:10": 4683.800, "05:15": 4662.220,
#     "05:20": 4657.820, "05:25": 4645.560, "05:30": 4626.380, "05:35": 4616.340,
#     "05:40": 4613.930, "05:45": 4560.220, "05:50": 4562.470, "05:55": 4546.290,
#     "06:00": 4502.510, "06:05": 4516.330, "06:10": 4495.050, "06:15": 4511.210,
#     "06:20": 4531.070, "06:25": 4490.730, "06:30": 4500.380, "06:35": 4483.030,
#     "06:40": 4475.450, "06:45": 4472.340, "06:50": 4464.440, "06:55": 4449.850,
#     "07:00": 4439.930, "07:05": 4453.900, "07:10": 4457.280, "07:15": 4426.200,
#     "07:20": 4437.780, "07:25": 4456.980, "07:30": 4424.940, "07:35": 4417.660,
#     "07:40": 4426.560, "07:45": 4418.700, "07:50": 4427.930, "07:55": 4387.870,
#     "08:00": 4397.310, "08:05": 4365.800, "08:10": 4397.270, "08:15": 4424.480,
#     "08:20": 4384.220, "08:25": 4423.560, "08:30": 4396.670, "08:35": 4404.470,
#     "08:40": 4443.330, "08:45": 4409.030, "08:50": 4418.390, "08:55": 4440.200,
#     "09:00": 4454.160, "09:05": 4455.180, "09:10": 4522.450, "09:15": 4567.100,
#     "09:20": 4583.110, "09:25": 4610.140, "09:30": 4628.980, "09:35": 4642.530,
#     "09:40": 4725.400, "09:45": 4743.240, "09:50": 4778.040, "09:55": 4815.010,
#     "10:00": 4823.140, "10:05": 4844.180, "10:10": 4880.600, "10:15": 4901.430,
#     "10:20": 4783.900, "10:25": 4924.390, "10:30": 4949.270, "10:35": 4973.250,
#     "10:40": 5000.970, "10:45": 5055.190, "10:50": 5034.300, "10:55": 5097.250,
#     "11:00": 5166.990, "11:05": 5113.110, "11:10": 5171.540, "11:15": 5183.810,
#     "11:20": 5200.510, "11:25": 5199.960, "11:30": 5244.690, "11:35": 5278.490,
#     "11:40": 5284.590, "11:45": 5305.640, "11:50": 5280.880, "11:55": 5351.820,
#     "12:00": 5378.860, "12:05": 5372.890, "12:10": 5432.590, "12:15": 5411.240,
#     "12:20": 5399.490, "12:25": 5456.890, "12:30": 5482.580, "12:35": 5446.530,
#     "12:40": 5468.430, "12:45": 5497.760, "12:50": 5527.870, "12:55": 5499.540,
#     "13:00": 5446.870, "13:05": 5449.680, "13:10": 5415.810, "13:15": 5430.670,
#     "13:20": 5445.860, "13:25": 5441.700, "13:30": 5478.010, "13:35": 5491.410,
#     "13:40": 5554.720, "13:45": 5561.380, "13:50": 5637.950, "13:55": 5642.020,
#     "14:00": 5635.480, "14:05": 5658.980, "14:10": 5688.120, "14:15": 5725.620,
#     "14:20": 5781.510, "14:25": 5772.120, "14:30": 5813.200, "14:35": 5834.400,
#     "14:40": 5852.490, "14:45": 5869.120, "14:50": 5821.210, "14 :55" : 5839}


# Predict for all 5-minute intervals from 00:00 to 23:55 on 1st September 2024
predictions = {}

# Define the target date for prediction
year, month, day = 2024, 9, 1

# Get the last 12 rows of available data for prediction sequence
last_12_intervals = df.tail(sequence_length)

# Prepare the input for prediction
X_predict = last_12_intervals[['year', 'month', 'day', 'hour', 'minute', 'day_of_week']].values

# Scale the prediction input
X_predict_scaled = scaler_X.transform(X_predict)

# Loop over every hour (00:00 to 23:00) and every 5 minutes within each hour
for hour in range(14):  # Loop through all hours
    for minute in range(0, 60, 5):  # Loop through every 5 minutes within each hour
        # Reshape the input for the LSTM
        X_predict_scaled = np.reshape(X_predict_scaled, (1, X_predict_scaled.shape[0], X_predict_scaled.shape[1]))

        # Predict the value for the current time interval
        predicted_value_scaled = model.predict(X_predict_scaled)
        predicted_value = scaler_y.inverse_transform(predicted_value_scaled)

        # Store the predicted value for the current time interval
        time_str = f"{hour:02d}:{minute:02d}"
        predictions[time_str] = predicted_value[0][0]

        # Update the sequence for the next interval
        new_minute = (minute + 5) % 60
        new_hour = hour if new_minute != 0 else (hour + 1) % 24  # Increment hour if needed

        # Add the new hour, minute, and day_of_week to the input sequence, replacing the oldest entry
        next_input = np.array([[year, month, day, new_hour, new_minute, last_12_intervals['day_of_week'].iloc[-1]]])
        next_input_scaled = scaler_X.transform(next_input)

        # Shift the sequence to the left and append the new scaled input
        X_predict_scaled = np.vstack([X_predict_scaled[0, 1:], next_input_scaled])

# Output the predicted value for each 5-minute interval from 00:00 to 23:55 on 1st September 2024
# Convert the real values and predicted values into lists of the same length
real_list = []
pred_list = []

# for time, real_value in real_value.items():
#     if time in predictions:  # Ensure the prediction exists for the corresponding time
#         real_list.append(real_value)
#         pred_list.append(predictions[time])

# Calculate Mean Absolute Error (MAE) and Mean Squared Error (MSE)
# mae = mean_absolute_error(real_list, pred_list)
# mse = mean_squared_error(real_list, pred_list)

# print(f"\nMean Absolute Error (MAE): {mae:.2f}")
# print(f"Mean Squared Error (MSE): {mse:.2f}")


    
model.save('LSTM_model.h5')