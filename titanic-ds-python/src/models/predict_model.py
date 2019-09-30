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
# from sklearn.ensemble import RandomForestClassifier
# from lightgbm import LGBMClassifier
# from sklearn.svm import SVC, LinearSVC

# Accuracy = (True Positives + True Negatives) / ( TN + FN + FP + TP)
# Presicion = True Positive / (TP+FP)
# Recall = True Positive/(TP+FP)
from sklearn.utils.testing import ignore_warnings

FILENAME='Prediction_outcome.txt'

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
    model = LogisticRegression(random_state=my_constants.RANDOM_VALUE)
    model_grid = GridSearchCV(model, param_grid=parameters, cv=3)
    with ignore_warnings(category=ConvergenceWarning):
        model_grid.fit(X_train, y_train)
    file_operations.write_logs(FILENAME, "Calculate logistic simple model" + model_grid)

    return model_grid


def do_generate_metrics_logistic_simple_model(X_train, y_train, X_test, y_test, grid):
    model = LogisticRegression(random_state=my_constants.RANDOM_VALUE)
    file_operations.write_logs(FILENAME, "Calculate logistic simple model best params" + grid.best_params_)
    model.set_params(**grid.best_params_)
    model.fit(X_train, y_train)
    metrics = calculate_metrics(model, X_test, y_test)
    file_operations.write_logs(FILENAME, "model params" + model.get_params() + " scores:" + model.score)
    file_operations.write_logs(FILENAME, "Grid params" + grid.best_params_ + " scores:" + grid.best_score_)

    return model, metrics


def do_generate_rf_optimazed_model(X_train, y_train, parameters):
    model = RandomForestClassifier(random_state=my_constants.RANDOM_VALUE,oob_score=True)
    model_grid = GridSearchCV(model, param_grid=parameters, cv=3)
    with ignore_warnings(category=ConvergenceWarning):
        model_grid.fit(X_train, y_train)
    print(model_grid)

    return model_grid


def do_generate_metrics_rf_optimazed_model(X_train, y_train, X_test, y_test, grid):
    model = RandomForestClassifier(random_state=my_constants.RANDOM_VALUE,oob_score=True)
    model.set_params(**grid.best_params_ )
    model.fit(X_train, y_train)
    metrics = calculate_metrics(model, X_test, y_test)
    print(model.get_params(), " ", model.score)
    print(grid.best_params_, " ", grid.best_score_)

    return model, metrics


def do_generate_metrics_lgbm_optimazed_model(X_train, y_train, X_test, y_test, grid):
    model = LGBMClassifier(random_state=0)
    model.set_params(**grid.best_params_)
    model.fit(X_train, y_train)
    metrics = calculate_metrics(model, X_test, y_test)
    print(model.get_params(), " ", model.score)
    print(grid.best_params_, " ", grid.best_score_)

    return model, metrics


def do_generate_lgbm_optimazed_model(X_train, y_train, parameters):

    model = LGBMClassifier(random_state=0)
    model = GridSearchCV(model, param_grid=parameters, cv=3)
    model.fit(X_train, y_train)
    return model


def create_logistic_optimazed_model(X_train, y_train, X_test, y_test):
    # parameters = {'C': np.logspace(0, 3, 100),
    #               'tol': np.logspace(-6, -3, 4),
    #               'penalty': ['l1', 'l2','none'],
    #               'class_weight': ['balanced', None],
    #               'solver': ['liblinear'],
    #               'max_iter': [100, 1000, 10000]}
    # grid = do_generate_logistic_simple_model(X_train, y_train, parameters)
    # model, metrics = do_generate_metrics_logistic_simple_model(X_train, y_train, X_test, y_test, grid)

    parameters2 = {'C': np.logspace(0, 3, 100),
                   'tol': np.logspace(-6, -3, 4),
                   'penalty': ['elasticnet'],
                   'class_weight': ['balanced', None],
                   'solver': ['saga'],
                   'l1_ratio':[0.01,0.1,0.5,0.7,0.99],
                   'max_iter': [100, 1000, 10000]}
    grid2 = do_generate_logistic_simple_model(X_train, y_train, parameters2)
    model2, metrics2 = do_generate_metrics_logistic_simple_model(X_train, y_train, X_test, y_test, grid2)

    return {
        'metrics': metrics if model.score > model2.score else metrics2,
        'model': model if model.score > model2.score else model2
    }


def create_rf_optimized_model(X_train, y_train, X_test, y_test):
    parameters = {'n_estimators': [50, 100, 200, 1000],
                  'min_samples_leaf': [1, 5, 10, 50],
                  'max_features': ('auto', 'sqrt', 'log2'),
                  }
    grid = do_generate_rf_optimazed_model(X_train, y_train, parameters)
    model, metrics = do_generate_metrics_rf_optimazed_model(X_train, y_train, X_test, y_test, grid)
    return {
        'metrics': metrics,
        'model': model
    }


def create_lgbm_optimized_model(X_train, y_train, X_test, y_test):
    parameters = {
        'learning_rate': [0.345, 0.35, 0.355, 0.36],
        'n_estimators': [8, 10, 20],
        'num_leaves': [25]
    }

    grid = do_generate_lgbm_optimazed_model(X_train, y_train, parameters)
    model, metrics = do_generate_metrics_lgbm_optimazed_model(X_train, y_train, X_test, y_test, grid)
    return {
        'metrics': metrics,
        'model': model
    }




def do_generate_svc_optimazed_model(X_train, y_train, parameters):

    estimador = SVC(random_state=my_constants.RANDOM_VALUE)
    print(estimador.get_params().keys())
    model = GridSearchCV(estimator=estimador, param_grid=parameters, cv=3, iid=False)
    model.fit(X_train, y_train)
    return model

def do_generate_metrics_svc_optimazed_model(X_train, y_train, X_test, y_test, grid):
    pass


def create_svc_optimized_model(X_train, y_train, X_test, y_test):

    parameters = {'kernel': ['rbf'],
                  'C': [7, 8, 9, 10, 20, 25, 30, 70, 100],
                  'gamma': [0.0001, 0.001, 0.008, 0.009, 0.01, 0.011],
                  'probability': [True]
                  }
    grid = do_generate_svc_optimazed_model(X_train, y_train, X_test, y_test, parameters)
    model, metrics = do_generate_metrics_svc_optimazed_model(X_train, y_train, X_test, y_test, grid)
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
    file_operations.write_logs(FILENAME, 'Creating linear model')
    base_model = create_base_model(X_train, y_train, X_test, y_test)
    file_operations.write_logs(FILENAME, "Metrics BaseModel: " + base_model['metrics'])
    file_operations.get_submission_file(base_model['model'], '01_base_model.csv', competition_df)

    # Logistic regression model
    file_operations.write_logs(FILENAME,'Creating logistic simple model')
    lg_simple_model = create_logistic_simple_model(X_train, y_train, X_test, y_test)
    file_operations.write_logs(FILENAME, "Metrics lg_simple_model: " + lg_simple_model['metrics'])
    file_operations.get_submission_file(lg_simple_model['model'], '02_lg_model.csv', competition_df)

    # Logistic regression model with hyp optimization
    file_operations.write_logs(FILENAME,'Creating logistic optimazed model')
    lg_optimazed_model = create_logistic_optimazed_model(X_train, y_train, X_test, y_test)
    file_operations.write_logs(FILENAME, "Metrics lg_optimazed_model: " + lg_optimazed_model['metrics'])
    file_operations.get_submission_file(lg_optimazed_model['model'], '03_lg_model_optimized.csv', competition_df)

    # print('Creating rf  model')
    # rf_model_scaled = create_rf_optimized_model(X_train, y_train, X_test, y_test)
    # print("Metrics rf_model_scaled: ", rf_model_scaled['metrics'])
    # file_operations.get_submission_file(rf_model_scaled['model'], '04_rf_model_optimized.csv', competition_df)
    #
    # print('Creating lgbm  model')
    # lgbm_model = create_lgbm_optimized_model(X_train, y_train, X_test, y_test)
    # print("Metrics lgbm_model: ", lgbm_model['metrics'])
    # file_operations.get_submission_file(lgbm_model['model'], '04_lgbm_model_optimized.csv', competition_df)
    #
    # print('Creating SVC scaled model')
    # svc_model = create_svc_optimized_model(X_train, y_train, X_test, y_test)
    # file_operations.get_submission_file(svc_model['model'], '05_svc_model_optimized_scaled.csv', competition_df)

    # Feature standarization
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # linear base dummy model
    file_operations.write_logs(FILENAME,'Creating dummy scaled model')
    base_model_scaled = create_base_model(X_train_scaled, y_train, X_test_scaled, y_test)
    file_operations.write_logs(FILENAME,"Metrics base_model_scaled: " +  base_model_scaled['metrics'])
    file_operations.get_submission_file_with_standardization(base_model_scaled['model'], '01_base_model_scaled.csv', scaler,
                                                             competition_df)

    # Logistic regression model
    file_operations.write_logs(FILENAME,'Creating logitic optimazed scaled model')
    lg_simple_model_scaled = create_logistic_simple_model(X_train, y_train, X_test, y_test)
    file_operations.write_logs(FILENAME,"Metrics lg_simple_model_scaled: " + lg_simple_model_scaled['metrics'])
    file_operations.get_submission_file_with_standardization(lg_simple_model_scaled['model'], '02_lg_model_scaled.csv', scaler,
                                                             competition_df)

    # Logistic regression model with hyp optimization
    print('Creating logitic optimazed scaled model')
    lg_optimazed_model_scaled = create_logistic_optimazed_model(X_train, y_train, X_test, y_test)
    print("Metrics lg_optimazed_model_scaled: ", lg_optimazed_model_scaled['metrics'])
    file_operations.get_submission_file_with_standardization(lg_optimazed_model_scaled['model'],
                                                             '03_lg_model_optimized_scaled.csv',
                                                             scaler, competition_df)
    # print('Creating rf scaled model')
    # rf_model_scaled = create_rf_optimized_model(X_train_scaled, y_train, X_test_scaled, y_test)
    # print("Metrics rf_model_scaled: ", rf_model_scaled['metrics'])
    # file_operations.get_submission_file_with_standardization(rf_model_scaled['model'],
    #                                                          '04_rf_model_optimized_scaled.csv',
    #                                                          scaler, competition_df)
    # print('Creating lgbm  scaled model')
    # lgbm_model_scaled = create_lgbm_optimized_model(X_train_scaled, y_train, X_test_scaled, y_test)
    # print("Metrics lgbm_model_scaled: ", lgbm_model_scaled['metrics'])
    # file_operations.get_submission_file(lgbm_model_scaled['model'], '04_lgbm_model_optimized_scaled.csv',
    #                                     competition_df)
    #
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