#!/usr/bin/python
# chabar.py

################################################################################
# Imports
################################################################################
import os
import json
import pickle
import pandas as pd
# import seaborn as sns                 # Not implemented yet
# import matplotlib.pyplot as plt       # Not implemented yet

from sklearn.pipeline import Pipeline
from sklearn.datasets.base import Bunch
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, r2_score, mean_squared_error as mse
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVR
from sklearn.base import BaseEstimator, TransformerMixin


################################################################################
# Load into Pandas & Visualize for Feature Analysis
################################################################################
data = pd.read_csv('data/silas_israel_dataset.txt',sep='\t',index_col='ID')

# Drop the two columns that won't be predictive
data.drop('Citation', axis=1, inplace=True)
data.drop('Name', axis=1, inplace=True)


################################################################################
# Make a Scikit-Learn Bunch
################################################################################
meta = {
    'target_names': list(data.Type.unique()),
    'feature_names': list(data.columns),
    'categorical_features': {
        column: list(data[column].unique())
        for column in data.columns
        if data[column].dtype == 'object'
    },
}

with open('data/meta.json', 'w') as f:
    json.dump(meta, f, indent=2)

def load_data(root='data'):
    # Load the meta data from the file
    with open(os.path.join(root, 'meta.json'), 'r') as f:
        meta = json.load(f)

    names = meta['feature_names']

    # Load the readme information
    with open(os.path.join(root, 'README.md'), 'r') as f:
        readme = f.read()

    # Load the data
    dataset = pd.read_csv(os.path.join(root, 'silas_israel_dataset.txt'), sep='\t',index_col='ID')

    # Remove the target, name, and citation from the categorical features
    meta['categorical_features'].pop('Type')

    # print dataset['Type']

    # Return the bunch with the appropriate data chunked apart
    return Bunch(
        data = dataset[['Location', 'Date_BP', 'Date_error', u'RSL', 'RSL_error', 'RSL_min', 'RSL_min_error', 'RSL_max', 'RSL_max_error']],
        target = dataset['Type'],
        target_names = meta['target_names'],
        feature_names = meta['feature_names'],
        categorical_features = meta['categorical_features'],
        DESCR = readme,
    )

################################################################################
# Custom Label Encoder
################################################################################
class EncodeCategorical(BaseEstimator, TransformerMixin):
    """
    Encodes a specified list of columns or all columns if None.
    """

    def __init__(self, columns=None):
        self.columns  = columns
        self.encoders = None

    def fit(self, data, target=None):
        """
        Expects a data frame with named columns to encode.
        """
        # Encode all columns if columns is None
        if self.columns is None:
            self.columns = data.columns

        # Fit a label encoder for each column in the data frame
        self.encoders = {
            column: LabelEncoder().fit(data[column])
            for column in self.columns
        }
        return self

    def transform(self, data):
        """
        Uses the encoders to transform a data frame.
        """
        output = data.copy()
        for column, encoder in self.encoders.items():
            output[column] = encoder.transform(data[column])

        return output


################################################################################
# Pickle the Model for Future Use
################################################################################
def dump_model(model, path='data', name='classifier.pickle'):
    with open(os.path.join(path, name), 'wb') as f:
        pickle.dump(model, f)


if __name__ == '__main__':
    israel = load_data()
    israel.data = israel.data.fillna(0)

    # Encode our target data
    yencode = LabelEncoder().fit(israel.target)

    # Construct the pipeline
    sealevel = Pipeline([
            ('encoder',  EncodeCategorical(israel.categorical_features.keys())),
            # ('classifier', LogisticRegression())
            ('regressor', SVR())
        ])

    # Fit the pipeline
    y_true = yencode.transform(israel.target)
    sealevel.fit(israel.data, y_true)

    # Use the model to get the predicted value
    y_pred = sealevel.predict(israel.data)

    # execute classification report
    # print classification_report(y_true, y_pred, target_names=israel.target_names)

    # execute regression evaluators mse and coefficient of determination
    print "The mean squared error is: %s. " % mse(y_true,y_pred)
    print "The coefficient of determination is: %s. " % r2_score(y_true,y_pred)
    # Pickle the model for future use
    dump_model(sealevel, name='regressor.pickle')
