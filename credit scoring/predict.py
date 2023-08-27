import argparse
import os

from catboost import CatBoostClassifier
import numpy as np
import pandas as pd


MODELS_PATH = './models/'

# Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument('--csv', nargs=1, required=True)
args = parser.parse_args()
file = args.csv[0]

# Load and prepare data
data = pd.read_csv(file, index_col=0)
data.reset_index(inplace=True)
data.drop(columns='58', inplace=True)
data['9'].fillna(-1, inplace=True)

# Load model
model = CatBoostClassifier()
model.load_model(MODELS_PATH + 'catboost.cbm')

# Predict
result = model.predict(data, 'Class')

# Save result to file
dirname = os.path.dirname(os.path.abspath(__file__))
result_file = os.path.join(dirname, 'result.txt')
np.savetxt(result_file, result, '%d')
