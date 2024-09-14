from django.shortcuts import render
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from tensorflow.keras.models import load_model
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import os
from datetime import datetime, timedelta
import joblib

# Load the model
model_path = 'E:/APPLICATIONS/VSC/SIH/Models/LSTM_model.h5'  
model = load_model(model_path)


# Load the dataset
data_path = 'C:/Users/arya0/OneDrive/Desktop/SLDC_Data_with_Day_of_Week.csv'  
df = pd.read_csv(data_path)
# Ensure 'hour' and 'minute' columns are created if not present
if 'hour' not in df.columns or 'minute' not in df.columns:
    df['hour'] = pd.to_datetime(df['time'], format='%H:%M').dt.hour
    df['minute'] = pd.to_datetime(df['time'], format='%H:%M').dt.minute

# Load the saved scalers
scaler_X_path = 'E:\APPLICATIONS\VSC\SIH\Models/main/scalers/scaler_X.pkl'
scaler_y_path = 'E:\APPLICATIONS\VSC\SIH\Models/main/scalers/scaler_y.pkl'
scaler_X = joblib.load(scaler_X_path)
scaler_y = joblib.load(scaler_y_path)



class PredictLoadAPIView(APIView):
    def post(self, request, *args, **kwargs):
        select_date = request.data.get('selectDate')
        select_month = request.data.get('selectMonth')
        select_year = request.data.get('selectYear')

        try:
            # Convert the received date into a datetime object
            input_date = datetime(int(select_year), int(select_month), int(select_date))

            # Prepare the input for the model (get the last 12 intervals of available data)
            last_12_intervals = df.tail(12)
            X_predict = last_12_intervals[['year', 'month', 'day', 'hour', 'minute', 'day_of_week']].values

            # Scale the input
            X_predict_scaled = scaler_X.transform(X_predict)

            # Predict for all 5-minute intervals of the selected date
            predictions = {}
            for hour in range(24):
                for minute in range(0, 60, 5):
                    X_predict_scaled = np.reshape(X_predict_scaled, (1, X_predict_scaled.shape[0], X_predict_scaled.shape[1]))
                    predicted_value_scaled = model.predict(X_predict_scaled)
                    predicted_value = scaler_y.inverse_transform(predicted_value_scaled)
                    time_str = f"{hour:02d}:{minute:02d}"
                    predictions[time_str] = predicted_value[0][0]

                    # Update the sequence for the next interval
                    new_minute = (minute + 5) % 60
                    new_hour = hour if new_minute != 0 else (hour + 1) % 24
                    next_input = np.array([[input_date.year, input_date.month, input_date.day, new_hour, new_minute, last_12_intervals['day_of_week'].iloc[-1]]])
                    next_input_scaled = scaler_X.transform(next_input)

                    X_predict_scaled = np.vstack([X_predict_scaled[0, 1:], next_input_scaled])

            return Response({'predictions': predictions}, status=status.HTTP_200_OK)

        except Exception as e:
            return Response({'error': str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

def load_frontend(request):
    return render(request, 'SIH_HTML.html')
