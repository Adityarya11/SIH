_Delhi Load Curve Prediction_  
This project aims to predict the electricity load for the Delhi region at specific dates and times using a machine learning model (LSTM). Users can input a specific date and time, and the model will return the predicted electricity load for that time slot. This project is built using Django for the backend and TensorFlow/Keras for the LSTM model, with the frontend powered by HTML and JavaScript.

**Project Overview**
The main goal of this project is to develop an accurate, time-based electricity load forecasting system. The system processes historical data and makes predictions for future loads using machine learning.

**Features**
1. Users can input a specific date and time within a 24-hour cycle.
2. The model provides real-time electricity load predictions for the specified date and time.
3. Predictions are based on historical data provided by the SLDC (State Load Dispatch Center).

  *Technologies Used*
1. Backend: Django, Django REST Framework
2. Frontend: HTML, CSS, JavaScript
3. Machine Learing: Tensorflow/Keras(LSTM Model)
4. Data Processing: Pandas, Scikit-Learn
5. Deployment: (To be deployed on the web server)

  *Project Status*
 **Current Status**
1. Model Training: The LSTM model has been trained on historical SLDC data.
2. Frontend Integration: Users can select both a date and time, and predictions are displayed on the frontend.
3. Backend Processing: The backend successfully processes user input, feeds it into the model, and returns predictions. Model is reloaded for each request to avoid caching of results.
4. Error Handling: The backend returns proper error messages in case of invalid inputs or processing failures.

 **Known Issues**
1. Data Refresh: The system relies on a static dataset. There is no real-time data collection from the SLDC at this point.
2. Prediction Accuracy: The current LSTM model provides reasonably accurate results, but further tuning may be required for enhanced accuracy.
3. Performance: Loading the model for each request can impact performance. Model optimization or caching approaches might improve this in future iterations.

**Future Approaches**
1. Real-Time Data Collection
   1. Objective: Integrate real-time data from official SLDC websites or APIs to keep the model updated with the latest information.
   2. Approach: Implement a web scraping service or API integration to gather fresh data periodically and update the model's training dataset.
2. Model Performance Optimization
   1. Objective: Improve the efficiency of the LSTM model and reduce the time taken for predictions.
   2. Approach: Implement model caching mechanisms or explore using a more efficient deployment method (e.g., using TensorFlow Serving or converting the model to TensorFlow Lite).
3. Model Retraining and Tuning
   1. Objective: Continuously improve the model's accuracy.
   2. Approach: Implement automated model retraining pipelines based on new data inputs and hyperparameter tuning.
4. Improved Error Handling
   1. Objective: Provide more informative error messages to users in case of invalid inputs or prediction failures.
   2. Approach: Expand backend error handling and validation mechanisms.
5. Deployment and Scaling
   1. Objective: Deploy the project on a scalable platform such as AWS, Heroku, or DigitalOcean.
   2. Approach: Optimize the backend for scalable deployment, add containerization support using Docker, and ensure proper load balancing for concurrent requests.
6. UI Enhancements: Allow for more interactive visualizations of predicted vs. actual load.

_**How to Run the Project Locally**_
**Clone the repository**: 
-> git clone https://github.com/Adityarya11/delhi-load-curve-prediction.git

**Navigate into the project directory**:
-> cd delhi-load-curve-prediction

**Install dependencies: Make sure you have Python and Django installed. Then, install the project requirements** :
-> pip install -r requirements.txt

**Set up the database and run migrations**:
-> python manage.py migrate

**Start the Django development server**:
-> python manage.py runserver


























