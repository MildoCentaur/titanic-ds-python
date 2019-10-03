# import kaggle

# kaggle.KaggleApi competitions download -c titanic


import numpy as np
import pandas as pd
import os

from lightgbm import LGBMClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
import my_constants
import file_operations
import sklearn
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score
from sklearn.utils.multiclass import unique_labels
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from lightgbm import LGBMClassifier
from sklearn.svm import SVC, LinearSVC


# Accuracy = (True Positives + True Negatives) / ( TN + FN + FP + TP)
# Presicion = True Positive / (TP+FP)
# Recall = True Positive/(TP+FP)
from sklearn.utils.testing import ignore_warnings
FILENAME='LGBM_Prediction_outcome.txt'

def calculate_metrics(model, X_test, y_test):
    score = model.score(X_test, y_test)
    precision = precision_score(y_test, model.predict(X_test))
    cm = confusion_matrix(y_test, model.predict(X_test))
    result = {
        'score': score,
        'precision': precision,
        'cm': cm
    }
    return result


def create_base_model(X_train, y_train, X_test, y_test):
    base_model = DummyClassifier(strategy='most_frequent', random_state=my_constants.RANDOM_VALUE)
    base_model.fit(X_train, y_train)
    metrics = calculate_metrics(base_model, X_test, y_test)
    return {
        'metrics': metrics,
        'model': base_model
    }


def do_generate_metrics_lgbm_optimazed_model(X_train, y_train, X_test, y_test, grid):
    file_operations.write_logs(FILENAME,"LGBM metrics calculation\n")
    model = LGBMClassifier(random_state=0)
    model.set_params(**grid.best_params_)
    model.fit(X_train, y_train)
    metrics = calculate_metrics(model, X_test, y_test)
    file_operations.write_logs(FILENAME,"Generated model params and results\n params:" + str(model.get_params()) + "\nscore " + str(model.score(X_test, y_test)))
    file_operations.write_logs(FILENAME,"Search grid best params and results\n params:"+ str(grid.best_params_) + "\nscore " + str(grid.best_score_))

    return model, metrics


def do_generate_lgbm_optimazed_model(X_train, y_train, parameters):
    file_operations.write_logs(FILENAME, 'Starting LGBM Grid Search with parameters:')
    file_operations.write_logs(FILENAME, str(parameters))
    model = LGBMClassifier(random_state=0)
    model = GridSearchCV(model, param_grid=parameters, cv=3, verbose=3, n_jobs=3)
    model.fit(X_train, y_train)
    file_operations.write_logs(FILENAME,"LGBM grid search completed")
    return model


def create_lgbm_optimized_model(X_train, y_train, X_test, y_test):
    parameters = {
        'learning_rate': [0.2, 0.3, 0.4, 0.5, 0.6],
        'n_estimators': [5, 10, 20, 50, 70, 100, 150, 200],
        'num_leaves': [10, 15, 20, 25, 30, 40, 50, 60],
        'subsample_for_bin': [10, 50, 100, 200],
        'reg_alpha': [0.1, 0.2, 0.5, 0.7,0.8],
        'reg_lambda': [0.1, 0.2, 0.5, 0.7, 0.8]
    }

    grid = do_generate_lgbm_optimazed_model(X_train, y_train, parameters)
    model, metrics = do_generate_metrics_lgbm_optimazed_model(X_train, y_train, X_test, y_test, grid)
    return {
        'metrics': metrics,
        'model': model
    }

""""
Generated model params and results
 params: {'boosting_type': 'gbdt', 'class_weight': None, 'colsample_bytree': 1.0, 'importance_type': 'split', 'learning_rate': 0.2, 'max_depth': -1, 'min_child_samples': 20, 'min_child_weight': 0.001, 'min_split_gain': 0.0, 'n_estimators': 20, 'n_jobs': -1, 'num_leaves': 15, 'objective': None, 'random_state': 0, 'reg_alpha': 0.1, 'reg_lambda': 0.5, 'silent': True, 'subsample': 1.0, 'subsample_for_bin': 200, 'subsample_freq': 0} 
score  0.8547486033519553
Search grid best params and results
 params: {'learning_rate': 0.2, 'n_estimators': 20, 'num_leaves': 15, 'reg_alpha': 0.1, 'reg_lambda': 0.5, 'subsample_for_bin': 200} 
score  0.8469101123595506
Metrics lgbm_model:  {'score': 0.8547486033519553, 'precision': 0.864406779661017, 'cm': array([[102,   8],
       [ 18,  51]])}
Creating lgbm  scaled model
Starting LGBM Grid Search with parameters: {'learning_rate': [0.2, 0.3, 0.4, 0.5, 0.6], 'n_estimators': [5, 10, 20, 50, 70, 100, 150, 200], 'num_leaves': [10, 15, 20, 25, 30, 40, 50, 60], 'subsample_for_bin': [10, 50, 100, 200], 'reg_alpha': [0.1, 0.2, 0.5, 0.7, 0.8], 'reg_lambda': [0.1, 0.2, 0.5, 0.7, 0.8]}
LGBM grid search completed
LGBM metrics calculation
Generated model params and results
 params: {'boosting_type': 'gbdt', 'class_weight': None, 'colsample_bytree': 1.0, 'importance_type': 'split', 'learning_rate': 0.3, 'max_depth': -1, 'min_child_samples': 20, 'min_child_weight': 0.001, 'min_split_gain': 0.0, 'n_estimators': 20, 'n_jobs': -1, 'num_leaves': 10, 'objective': None, 'random_state': 0, 'reg_alpha': 0.1, 'reg_lambda': 0.2, 'silent': True, 'subsample': 1.0, 'subsample_for_bin': 200, 'subsample_freq': 0} 
score  0.8603351955307262
Search grid best params and results
 params: {'learning_rate': 0.3, 'n_estimators': 20, 'num_leaves': 10, 'reg_alpha': 0.1, 'reg_lambda': 0.2, 'subsample_for_bin': 200} 
score  0.8469101123595506
Metrics lgbm_model_scaled:  {'score': 0.8603351955307262, 'precision': 0.8666666666666667, 'cm': array([[102,   8],
       [ 17,  52]])}

"""

def predictions():
    train_df = file_operations.read_data('processed', 'train.csv', 'PassengerId')
    competition_df = file_operations.read_data('processed', 'test.csv', 'PassengerId')

    X = train_df.loc[:, 'Age':].values.astype('float')
    y = train_df['Survived'].ravel()
    shape = X.shape
    if shape[0] == 891 & shape[1] > 36:
        file_operations.write_logs(FILENAME,"Dataset has ", shape[1], " and right amount amount of rows")

    file_operations.write_logs(FILENAME,'Creating test and train dataset')
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=my_constants.TEST_SIZE,
                                                        random_state=my_constants.RANDOM_VALUE)
    # Linear base dummy model
    file_operations.write_logs(FILENAME,'Creating linear model')
    base_model = create_base_model(X_train, y_train, X_test, y_test)
    file_operations.write_logs(FILENAME,"Metrics base_model: ")
    file_operations.write_logs(FILENAME, str(base_model['metrics']))
    file_operations.get_submission_file(base_model['model'], '01_base_model.csv', competition_df)

    file_operations.write_logs(FILENAME,'Creating lgbm  model')
    lgbm_model = create_lgbm_optimized_model(X_train, y_train, X_test, y_test)
    file_operations.write_logs(FILENAME,"Metrics lgbm_model: ")
    file_operations.write_logs(FILENAME, str(lgbm_model['metrics']))
    file_operations.get_submission_file(lgbm_model['model'], '04_lgbm_model_optimized.csv', competition_df)

    # Feature standarization
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    file_operations.write_logs(FILENAME,'Creating lgbm  scaled model')
    lgbm_model_scaled = create_lgbm_optimized_model(X_train_scaled, y_train, X_test_scaled, y_test)
    file_operations.write_logs(FILENAME, "Metrics lgbm_model_scaled: ")
    file_operations.write_logs(FILENAME, str(lgbm_model_scaled['metrics']))
    file_operations.get_submission_file(str(lgbm_model_scaled['model']), '04_lgbm_model_optimized_scaled.csv',
                                        competition_df)


predictions()