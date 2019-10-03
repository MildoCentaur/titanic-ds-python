# import kaggle

# kaggle.KaggleApi competitions download -c titanic

import numpy as np
import pandas as pd
import os

#from lightgbm import LGBMClassifier
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
#from lightgbm import LGBMClassifier
from sklearn.svm import SVC, LinearSVC
FILENAME='RF_Prediction_outcome.txt'

# Accuracy = (True Positives + True Negatives) / ( TN + FN + FP + TP)
# Presicion = True Positive / (TP+FP)
# Recall = True Positive/(TP+FP)
from sklearn.utils.testing import ignore_warnings


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


def create_logistic_simple_model(X_train, y_train, X_test, y_test):
    model = LogisticRegression(solver='liblinear', random_state=my_constants.RANDOM_VALUE)
    model.fit(X_train, y_train)
    metrics = calculate_metrics(model, X_test, y_test)
    return {
        'metrics': metrics,
        'model': model
    }


def do_generate_logistic_simple_model(X_train, y_train, parameters):
    file_operations.write_logs(FILENAME, "Calculate logistic simple model")
    model = LogisticRegression(random_state=my_constants.RANDOM_VALUE)
    model_grid = GridSearchCV(model, param_grid=parameters, cv=3, verbose=3, n_jobs=3)
    with ignore_warnings(category=ConvergenceWarning):
        model_grid.fit(X_train, y_train)

    file_operations.write_logs(FILENAME, "search grid")
    file_operations.write_logs(FILENAME, str(model_grid))
    return model_grid


def do_generate_metrics_logistic_simple_model(X_train, y_train, X_test, y_test, grid):
    file_operations.write_logs(FILENAME, "do_generate_metrics_logistic_simple_model")
    model = LogisticRegression(random_state=my_constants.RANDOM_VALUE)
    file_operations.write_logs(FILENAME, "grid Best params")
    file_operations.write_logs(FILENAME, str(grid.best_params_))

    model.set_params(**grid.best_params_)
    model.fit(X_train, y_train)
    metrics = calculate_metrics(model, X_test, y_test)
    file_operations.write_logs(FILENAME, 'model params:' + str(model.get_params()) + " model score:" + str(model.score))
    file_operations.write_logs(FILENAME, 'model grid.best_params_:' + str(model.get_params()) + " grid.best_score_:" + str(grid.best_score_))

    return model, metrics


def do_generate_rf_optimazed_model(X_train, y_train, parameters):
    file_operations.write_logs(FILENAME,'Starting RF Grid Search with parameters:' + str(parameters))
    model = RandomForestClassifier(random_state=my_constants.RANDOM_VALUE, oob_score=True)
    model_grid = GridSearchCV(model, param_grid=parameters, cv=3, verbose=3, n_jobs=3)
    with ignore_warnings(category=ConvergenceWarning):
        model_grid.fit(X_train, y_train)

    file_operations.write_logs(FILENAME,'RF Grid search completed')

    return model_grid


def do_generate_metrics_rf_optimazed_model(X_train, y_train, X_test, y_test, grid):
    file_operations.write_logs(FILENAME,'Starting metrics calculation')
    model = RandomForestClassifier(random_state=my_constants.RANDOM_VALUE, oob_score=True)
    model.set_params(**grid.best_params_)
    model.fit(X_train, y_train)
    metrics = calculate_metrics(model, X_test, y_test)
    file_operations.write_logs(FILENAME, "Generated model params and results\n params:" + str(model.get_params())
                               + "\nscore " + str(model.score(X_test, y_test)))
    file_operations.write_logs(FILENAME, "Search grid best params and results\n params:" + str(grid.best_params_)
                               + "\nscore " + str(grid.best_score_))

    return model, metrics


def create_logistic_optimazed_model(X_train, y_train, X_test, y_test):
    parameters = {'C': np.logspace(0, 3, 100),
                  'tol': np.logspace(-6, -3, 4),
                  'penalty': ['l1', 'l2'],
                  'class_weight': ['balanced', None],
                  'solver': ['liblinear'],
                  'max_iter': [100, 1000, 10000]}
    grid = do_generate_logistic_simple_model(X_train, y_train, parameters)
    model, metrics = do_generate_metrics_logistic_simple_model(X_train, y_train, X_test, y_test, grid)

    parameters2 = {'C': np.logspace(0, 3, 100),
                   'tol': np.logspace(-6, -3, 4),
                   'penalty': ['elasticnet', 'none'],
                   'class_weight': ['balanced', None],
                   'solver': ['saga'],
                   'max_iter': [100, 1000, 10000]}
    grid2 = do_generate_logistic_simple_model(X_train, y_train, X_test, y_test, parameters2)
    model2, metrics2 = do_generate_metrics_logistic_simple_model(X_train, y_train, X_test, y_test, grid2)

    return {
        'metrics': metrics if model.score > model2.score else metrics2,
        'model': model if model.score > model2.score else model2
    }


def create_rf_optimized_model(X_train, y_train, X_test, y_test):
    parameters = {'n_estimators': [10, 50, 100, 200, 1000],
                  'min_samples_leaf': [1, 2, 3, 5, 10, 20,50],
                  'min_samples_split': [2, 5, 10],
                  'max_features': ['auto', None, 'log2', 2 , 5],
                  'criterion': ['entropy','gini'],
                  'n_jobs': [None, 2, 3]
                  }
    grid = do_generate_rf_optimazed_model(X_train, y_train, parameters)
    model, metrics = do_generate_metrics_rf_optimazed_model(X_train, y_train, X_test, y_test, grid)
    return {
        'metrics': metrics,
        'model': model
    }


def predictions():
    train_df = file_operations.read_data('processed', 'train.csv', 'PassengerId')
    competition_df = file_operations.read_data('processed', 'test.csv', 'PassengerId')

    X = train_df.loc[:, 'Age':].values.astype('float')
    y = train_df['Survived'].ravel()
    shape = X.shape
    if shape[0] == 891 & shape[1] > 36:
        file_operations.write_logs(FILENAME, "Dataset has " + shape[1] + " and right amount amount of rows")

    file_operations.write_logs(FILENAME, 'Creating test and train dataset')
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=my_constants.TEST_SIZE,
                                                        random_state=my_constants.RANDOM_VALUE)
    # Linear base dummy model
    file_operations.write_logs(FILENAME,'Creating linear model')
    base_model = create_base_model(X_train, y_train, X_test, y_test)
    file_operations.write_logs(FILENAME, "Metrics base_model: " + str(base_model['metrics']))
    file_operations.get_submission_file(base_model['model'], '01_base_model.csv', competition_df)

    file_operations.write_logs(FILENAME,'Creating rf  model')
    rf_model_scaled = create_rf_optimized_model(X_train, y_train, X_test, y_test)
    file_operations.write_logs(FILENAME, "Metrics rf_model_scaled: " + rf_model_scaled['metrics'])
    file_operations.get_submission_file(rf_model_scaled['model'], '04_rf_model_optimized.csv', competition_df)

    # print('Creating lgbm  model')
    # lgbm_model = create_lgbm_optimized_model(X_train, y_train, X_test, y_test)
    # print("Metrics lgbm_model: ", lgbm_model['metrics'])
    # file_operations.get_submission_file(lgbm_model['model'], '04_lgbm_model_optimized.csv', competition_df)

    # print('Creating SVC scaled model')
    # svc_model = create_svc_optimized_model(X_train, y_train, X_test, y_test)
    # file_operations.get_submission_file(svc_model['model'], '05_svc_model_optimized_scaled.csv', competition_df)

    # Feature standarization
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    file_operations.write_logs(FILENAME, 'Creating rf scaled model')
    rf_model_scaled = create_rf_optimized_model(X_train_scaled, y_train, X_test_scaled, y_test)
    file_operations.write_logs(FILENAME, "Metrics rf_model_scaled: ", rf_model_scaled['metrics'])
    file_operations.get_submission_file_with_standardization(rf_model_scaled['model'],
                                                             '04_rf_model_optimized_scaled.csv',
                                                             scaler, competition_df)
    # print('Creating lgbm  scaled model')
    # lgbm_model_scaled = create_lgbm_optimized_model(X_train_scaled, y_train, X_test_scaled, y_test)
    # print("Metrics lgbm_model_scaled: ", lgbm_model_scaled['metrics'])
    # file_operations.get_submission_file(lgbm_model_scaled['model'], '04_lgbm_model_optimized_scaled.csv',
    #                                     competition_df)

    # print('Creating SVC scaled model')
    # svc_model_scaled = create_svc_optimized_model(X_train_scaled, y_train, X_test_scaled, y_test)
    # file_operations.get_submission_file(svc_model_scaled['model'], '05_svc_model_optimized_scaled.csv', competition_df)


predictions()

"""
from sklearn.svm import SVC, LinearSVC
estimador = SVC(random_state=0)
print (estimador.get_params().keys())
parameters = {  'kernel': ['rbf'],
    'C':     [7,8,9,10,20,25,30,70,100],
    'gamma': [0.0001,0.001,0.008,0.009,0.01,0.011],
    'probability': [True]
}
svc_grid = GridSearchCV(estimator=estimador, param_grid=parameters, cv=5, iid=False)
svc_grid.fit(X_train, y_train)
score = svc_grid.score(X_test, y_test)
print('%s: %.2f' % (estimador.__class__.__name__, score))
print(svc_grid.best_params_)
# get submission file
get_submission_file(svc_grid, '07_svc_hp_tunning.csv')
labels = [0, 1]
cm = confusion_matrix(y_test, svc_grid.predict(X_test), labels=labels)
print (cm)

"""


"""
RF Grid search completed
Starting metrics calculation
Generated model params and results
 params: {'bootstrap': True, 'class_weight': None, 'criterion': 'gini', 'max_depth': None, 'max_features': 'log2', 'max_leaf_nodes': None, 'min_impurity_decrease': 0.0, 'min_impurity_split': None, 'min_samples_leaf': 2, 'min_samples_split': 5, 'min_weight_fraction_leaf': 0.0, 'n_estimators': 50, 'n_jobs': None, 'oob_score': True, 'random_state': 0, 'verbose': 0, 'warm_start': False} 
score  0.8379888268156425
Search grid best params and results
 params: {'criterion': 'gini', 'max_features': 'log2', 'min_samples_leaf': 2, 'min_samples_split': 5, 'n_estimators': 50, 'n_jobs': None} 
score  0.8384831460674157
Metrics rf_model_scaled:  {'score': 0.8379888268156425, 'precision': 0.8448275862068966, 'cm': array([[101,   9],
       [ 20,  49]])}
Creating lgbm  scaled model
Starting LGBM Grid Search with parameters: {'learning_rate': [0.345, 0.35, 0.355, 0.36], 'n_estimators': [8, 10, 20], 'num_leaves': [25]}
LGBM grid search completed
LGBM metrics calculation
Generated model params and results
 params: {'boosting_type': 'gbdt', 'class_weight': None, 'colsample_bytree': 1.0, 'importance_type': 'split', 'learning_rate': 0.355, 'max_depth': -1, 'min_child_samples': 20, 'min_child_weight': 0.001, 'min_split_gain': 0.0, 'n_estimators': 8, 'n_jobs': -1, 'num_leaves': 25, 'objective': None, 'random_state': 0, 'reg_alpha': 0.0, 'reg_lambda': 0.0, 'silent': True, 'subsample': 1.0, 'subsample_for_bin': 200000, 'subsample_freq': 0} 
score  0.8379888268156425
Search grid best params and results
 params: {'learning_rate': 0.355, 'n_estimators': 8, 'num_leaves': 25} 
score  0.8426966292134831
Metrics lgbm_model_scaled:  {'score': 0.8379888268156425, 'precision': 0.8333333333333334, 'cm': array([[100,  10],
       [ 19,  50]])}

"""