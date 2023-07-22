'''

 examples/frequency.py - Copyright 2023 Harshad Joshi and Bufferstack.IO Analytics Technology LLP, Pune

 Licensed under the GNU General Public License, Version 3.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

 https://www.gnu.org/licenses/gpl-3.0.html

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.


'''

import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from scipy import stats
import joblib
import numpy as np
import logging

# Set up logging
logging.basicConfig(filename='data_processing.log', level=logging.INFO)

# Function to load data from a CSV file
def load_data(file_path):
    data = pd.read_csv(file_path)
    return data

# Load data from CSV
data = load_data('frequency.csv')  # Replace with your actual file path

# Check for missing values
if data.isnull().values.any():
    logging.info("Data contains missing values. Filling with mean...")
    data.fillna(data.mean(), inplace=True)

# Check for infinite values in numeric columns
numeric_columns = ['pulse', 'spikes', 'value']
if (np.abs(data[numeric_columns].values) >= np.finfo(np.float64).max).any():
    logging.info("Data contains infinite values. Replacing with max float64...")
    data[numeric_columns] = data[numeric_columns].replace([np.inf, -np.inf], np.finfo(np.float64).max)

# Handle outliers
z_scores = stats.zscore(data[numeric_columns])
abs_z_scores = np.abs(z_scores)
filtered_entries = (abs_z_scores < 3).all(axis=1)
data = data[filtered_entries]

# Prepare data for machine learning
X = data[['pulse', 'spikes']]  # Features
y = data['value']  # Target

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the pipeline
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('model', LinearRegression())
])

# Train the model
pipeline.fit(X_train, y_train)

# Perform cross-validation
cv = KFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(pipeline, X, y, cv=cv, scoring='neg_mean_squared_error')
logging.info(f'Cross-validation MSE: {-np.mean(scores)}')

# Make predictions on the test set
y_pred = pipeline.predict(X_test)

# Calculate the mean squared error of the model
mse = mean_squared_error(y_test, y_pred)
logging.info(f'Mean Squared Error: {mse}')

# Save the trained pipeline for future use
joblib.dump(pipeline, 'pipeline.pkl')

# Load the pipeline from the file
pipeline = joblib.load('pipeline.pkl')

# New data to predict
new_data = pd.DataFrame({'voltage.pulse': [0, 1], 'voltage.spikes': [12, 13]})

# Check for missing values in new_data
if new_data.isnull().values.any():
    logging.info("New data contains missing values. Filling with mean...")
    new_data.fillna(new_data.mean(), inplace=True)

# Check for infinite values in new_data
if (np.abs(new_data.values) >= np.finfo(np.float64).max).any():
    logging.info("New data contains infinite values. Replacing with max float64...")
    new_data = new_data.replace([np.inf, -np.inf], np.finfo(np.float64).max)

# Make predictions on the new data
new_pred = pipeline.predict(new_data)
logging.info(f'New Predictions: {new_pred}')

