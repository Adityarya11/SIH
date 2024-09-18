import os
from django.shortcuts import render
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model  # type: ignore
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime
import joblib
from django.views.decorators.csrf import csrf_exempt
from django.utils.decorators import method_decorator
from dateutil import parser

# Load the model and scalers
model_path = os.path.join('E:/APPLICATIONS/VSC/SIH/Models', 'lstm_model_1.h5')
scaler_X_path = os.path.join('E:/APPLICATIONS/VSC/SIH/Models/main/scalers', 'scaler_x.pkl')
scaler_y_path = os.path.join('E:/APPLICATIONS/VSC/SIH/Models/main/scalers', 'scaler_y.pkl')

if os.path.exists(model_path):
    model = load_model(model_path)
else:
    raise FileNotFoundError(f"Model file not found at {model_path}")

if os.path.exists(scaler_X_path) and os.path.exists(scaler_y_path):
    scaler_X = joblib.load(scaler_X_path)
    scaler_y = joblib.load(scaler_y_path)
else:
    raise FileNotFoundError("Scalers not found.")

# Load and preprocess the dataset
data_path = os.path.join('C:/Users/arya0/Downloads', 'updated_sldc_data_without_time.csv')
if os.path.exists(data_path):
    df = pd.read_csv(data_path)
    df['datetime'] = pd.to_datetime(df[['year', 'month', 'day', 'hour', 'minute']])
    if 'hour' not in df.columns or 'minute' not in df.columns:
        if 'time' in df.columns:
            df['hour'] = pd.to_datetime(df['time'], format='%H:%M').dt.hour
            df['minute'] = pd.to_datetime(df['time'], format='%H:%M').dt.minute
        else:
            raise ValueError("The dataset must have either 'time' or 'datetime' column to extract hour and minute.")
else:
    raise FileNotFoundError(f"Data file not found at {data_path}")

# Prediction function
def predict_for_target_datetime(target_datetime_str, sequence_length, scaler_X, scaler_y, model, df):
    try:
        # Use dateutil.parser to handle various formats
        target_datetime = parser.parse(target_datetime_str)

        # Filter rows less than the target date and time
        last_intervals = df.loc[
            (df['year'] <= target_datetime.year) &
            (df['month'] <= target_datetime.month) &
            (df['day'] <= target_datetime.day) &
            (df['hour'] <= target_datetime.hour) &
            (df['minute'] < target_datetime.minute)
        ].tail(sequence_length)

        # Ensure sufficient data is available
        if len(last_intervals) < sequence_length:
            return None  # Handle this case if required

        # Extract features from the last intervals (year, month, day, hour, minute, day_of_week)
        X_predict = last_intervals[['year', 'month', 'day', 'hour', 'minute', 'day_of_week']].values

        # Scale the prediction input
        X_predict_scaled = scaler_X.transform(X_predict)

        # Reshape the input for the LSTM model
        X_predict_scaled = np.reshape(X_predict_scaled, (1, X_predict_scaled.shape[0], X_predict_scaled.shape[1]))

        # Predict the value for the target date and time
        predicted_value_scaled = model.predict(X_predict_scaled)

        # Inverse transform to get the actual value
        predicted_value = scaler_y.inverse_transform(predicted_value_scaled)

        return predicted_value[0][0]  # Return the actual predicted value
    except Exception as e:
        raise ValueError(f"Error during prediction: {str(e)}")

def predict_for_nearest_datetime(target_datetime_str, sequence_length, scaler_X, scaler_y, model, df):
    try:
        # Use dateutil.parser to handle various formats
        target_datetime = parser.parse(target_datetime_str)

        # Find the nearest row in the DataFrame to the given date and time
        df['datetime'] = pd.to_datetime(df[['year', 'month', 'day', 'hour', 'minute']])
        df['time_diff'] = (df['datetime'] - target_datetime).abs()

        # Get the closest row
        closest_row = df.loc[df['time_diff'].idxmin()]

        # Ensure the nearest sequence is of valid length
        last_intervals = df.loc[df['datetime'] <= closest_row['datetime']].tail(sequence_length)
        if len(last_intervals) < sequence_length:
            return None  # Handle this case if there is not enough historical data

        # Extract features from the last intervals
        X_predict = last_intervals[['year', 'month', 'day', 'hour', 'minute', 'day_of_week']].values
        X_predict_scaled = scaler_X.transform(X_predict)
        X_predict_scaled = np.reshape(X_predict_scaled, (1, X_predict_scaled.shape[0], X_predict_scaled.shape[1]))

        predicted_value_scaled = model.predict(X_predict_scaled)
        predicted_value = scaler_y.inverse_transform(predicted_value_scaled)

        return predicted_value[0][0]  # Return the nearest predicted value
    except Exception as e:
        raise ValueError(f"Error during prediction: {str(e)}")


@method_decorator(csrf_exempt, name='dispatch')
class PredictLoadAPIView(APIView):
    def post(self, request, *args, **kwargs):
        select_date = request.data.get('selectDate')
        select_time = request.data.get('selectTime')

        try:
            # Combine date and time
            target_datetime_str = f"{select_date} {select_time}"

            # Predict the load for the target date and time
            predicted_value = predict_for_target_datetime(target_datetime_str, sequence_length=12, scaler_X=scaler_X, scaler_y=scaler_y, model=model, df=df)

            if predicted_value is None:
                return Response({'error': 'Not enough historical data to make a prediction.'}, status=status.HTTP_400_BAD_REQUEST)

            return Response({'prediction': predicted_value}, status=status.HTTP_200_OK)

        except Exception as e:
            return Response({'error': str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


# View for rendering the frontend
def load_frontend(request):
    return render(request, 'SIH_HTML.html')