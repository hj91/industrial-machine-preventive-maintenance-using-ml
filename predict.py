'''

 predict.py - Copyright 2023 Harshad Joshi and Bufferstack.IO Analytics Technology LLP, Pune

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
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.model_selection import KFold, GridSearchCV, train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, VotingRegressor
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, f_regression
from scipy import stats
import joblib
import numpy as np
import logging

# Set up logging
logging.basicConfig(filename='data_processing.log', level=logging.INFO)

def load_data(file_path):
    """Load data from a CSV file."""
    return pd.read_csv(file_path)

def fill_missing_values(data):
    """Fill missing values with the mean of each respective column."""
    if data.isnull().values.any():
        logging.info("Data contains missing values. Filling with mean...")
        data.fillna(data.mean(), inplace=True)

def handle_infinite_values(data, numeric_columns):
    """Replace infinite values with the maximum finite representable number."""
    if (np.abs(data[numeric_columns].values) >= np.finfo(np.float64).max).any():
        logging.info("Data contains infinite values. Replacing with max float64...")
        data[numeric_columns] = data[numeric_columns].replace([np.inf, -np.inf], np.finfo(np.float64).max)

def remove_outliers(data, numeric_columns):
    """Remove rows with a Z-score greater than 3 in any of the numeric columns."""
    z_scores = stats.zscore(data[numeric_columns])
    abs_z_scores = np.abs(z_scores)
    filtered_entries = (abs_z_scores < 3).all(axis=1)
    return data[filtered_entries]

def create_pipeline():
    """Create a pipeline that scales the data, selects best features, and fits a model."""
    model1 = LinearRegression()
    model2 = RandomForestRegressor()  # Replace with your actual second model
    model = VotingRegressor([('lr', model1), ('rf', model2)])  # Ensemble
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('feature_selection', SelectKBest(score_func=f_regression, k=2)),
        ('model', model)
    ])
    return pipeline

def perform_cross_validation(pipeline, X, y):
    """Perform cross-validation and log the result."""
    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(pipeline, X, y, cv=cv, scoring='neg_mean_squared_error')
    logging.info(f'Cross-validation MSE: {-np.mean(scores)}')

# Load and preprocess data
data = load_data('data.csv')
fill_missing_values(data)
handle_infinite_values(data, ['voltage.pulse', 'voltage.spikes', 'voltage'])
data = remove_outliers(data, ['voltage.pulse', 'voltage.spikes', 'voltage'])

# Prepare data for machine learning
X = data[['voltage.pulse', 'voltage.spikes']]  # Features
y = data['voltage']  # Target

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define and train the pipeline
pipeline = create_pipeline()
pipeline.fit(X_train, y_train)

# Perform cross-validation
perform_cross_validation(pipeline, X, y)

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

# Check for missing values and infinite values in new_data
fill_missing_values(new_data)
handle_infinite_values(new_data, ['voltage.pulse', 'voltage.spikes'])

# Make predictions on the new data
new_pred = pipeline.predict(new_data)
logging.info(f'New Predictions: {new_pred}')

