
# Voltage Value Prediction Model

This project is a Python script for training and using a machine learning model to predict voltage values based on the 'voltage.pulse' and 'voltage.spikes' features. The script utilizes an ensemble of Linear Regression and Random Forest models for prediction. Replace 'voltage' with your required paramater and provide it with suitable csv reflecting the change

## Features

- Loads data from a CSV file
- Handles missing values and infinite values in the data
- Removes outliers from the data
- Splits the data into training and test sets
- Uses a pipeline to scale the data, select best features, and fit the model
- Performs 5-fold cross-validation
- Evaluates the model using Mean Squared Error
- Saves the trained model for future use
- Loads the trained model from disk
- Makes predictions on new data

## How to Use

1. Prepare your data in a CSV file with columns 'voltage.pulse', 'voltage.spikes', and 'voltage'.
2. Replace 'data.csv' in the script with the path to your CSV file.
3. Run the script in your Python environment.
4. Check 'data_processing.log' for information about the script's progress and the model's performance.
5. Replace the 'new_data' DataFrame with your new data to make predictions.

## Requirements

- Python 3.6 or later
- pandas
- sklearn
- scipy
- joblib
- numpy

## Author

Harshad Joshi

## GitHub Link

[GitHub](https://github.com/hj91/industrial-machine-preventive-maintenance-using-ml)
