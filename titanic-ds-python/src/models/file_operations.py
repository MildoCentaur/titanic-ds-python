
import numpy as np
import pandas as pd
import os

from lightgbm import LGBMClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
import my_constants
import sklearn
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score
from sklearn.utils.multiclass import unique_labels
from sklearn.linear_model import LogisticRegression


def read_data(folder, filename, index_column='index'):
    # set the path of the raw data
    data_path = os.path.join(os.path.pardir, os.path.pardir, 'data', folder)
    file_path = os.path.join(data_path, filename)
    # read the data with all default parameters
    df = pd.read_csv(file_path, index_col=index_column)
    return df


def get_submission_file(model, filename, test_df):
    # converting to the matrix
    test_X = test_df.values.astype('float')
    # make predictions
    predictions = model.predict(test_X)
    # submission dataframe
    df_submission = pd.DataFrame({'PassengerId': test_df.index, 'Survived': predictions})
    # submission file
    submission_data_path = os.path.join(os.path.pardir, os.path.pardir, 'data', 'external')
    submission_file_path = os.path.join(submission_data_path, filename)
    # write to the file
    df_submission.to_csv(submission_file_path, index=False)


def get_submission_file_with_standardization(model,filename, scaler, test_df):
    # converting to the matrix
    test_X = test_df.values.astype('float')
    # standardization
    if (scaler!=None):
        test_X = scaler.transform(test_X)
    # make predictions
    predictions = model.predict(test_X)
    # submission dataframe
    df_submission = pd.DataFrame({'PassengerId': test_df.index, 'Survived' : predictions})
    # submission file
    submission_data_path = os.path.join(os.path.pardir, os.path.pardir, 'data', 'external')
    submission_file_path = os.path.join(submission_data_path, filename)
    # write to the file
    df_submission.to_csv(submission_file_path, index=False)


def write_logs(filename, message):
    file = open(filename, 'a+')
    file.write(message)
    file.write('\n')
    file.close()
