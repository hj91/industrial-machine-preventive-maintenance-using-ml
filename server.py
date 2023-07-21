'''


 server.py - Copyright 2023 Harshad Joshi and Bufferstack.IO Analytics Technology LLP, Pune

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


from flask import Flask, request, jsonify
import pandas as pd
import joblib

app = Flask(__name__)

# Load the pipeline from the file
pipeline = joblib.load('pipeline.pkl')

# Functions required by the pipeline
def fill_missing_values(data):
    # Fill missing values function goes here
    pass

def handle_infinite_values(data, cols):
    # Handle infinite values function goes here
    pass

@app.route('/predict', methods=['POST'])
def predict():
    # Get data from POST request
    data = request.json

    # Map the data to DataFrame columns
    new_data = pd.DataFrame({'voltage.pulse': [data['pulse']], 'voltage.spikes': [data['spikes']]})

    # Check for missing values and infinite values in new_data
    fill_missing_values(new_data)
    handle_infinite_values(new_data, ['voltage.pulse', 'voltage.spikes'])

    # Make predictions on the new data
    new_pred = pipeline.predict(new_data)

    # Log the prediction
    app.logger.info(f'New Predictions: {new_pred}')

    # Return the prediction
    return jsonify({'prediction': new_pred.tolist()})

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=True)

