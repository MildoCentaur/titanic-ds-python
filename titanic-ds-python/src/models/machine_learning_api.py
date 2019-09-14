
from flask import Flask, request
import pandas as pd
import numpy as np
import json
import pickle
import os

app = Flask(__name__)

# Load Model and Scaler Files
model_path = os.path.join('models')
model_filepath = os.path.join(model_path, 'lr_model.pkl')
scaler_filepath = os.path.join(model_path, 'lr_scaler.pkl')

scaler = pickle.load(open(scaler_filepath,'rb'))
model = pickle.load(open(model_filepath,'rb'))

# columns
columns = [ u'Age', u'Fare', u'FamilySize', \
       u'IsMother', u'IsMale', u'Deck_A', u'Deck_B', u'Deck_C', u'Deck_D', \
       u'Deck_E', u'Deck_F', u'Deck_G', u'Deck_Z', u'Pclass_1', u'Pclass_2', \
       u'Pclass_3', u'Title_Lady', u'Title_Master', u'Title_Miss', u'Title_Mr', \
       u'Title_Mrs', u'Title_Officer', u'Title_Sir', u'Fare_Bin_very_low', \
       u'Fare_Bin_low', u'Fare_Bin_high', u'Fare_Bin_very_high', u'Embarked_C', \
       u'Embarked_Q', u'Embarked_S', u'AgeState_Adult', u'AgeState_Child'] 


@app.route('/api', methods=['POST'])
def make_prediction():
    # read json object and conver to json string
    data = json.dumps(request.get_json(force=True))
    # create pandas dataframe using json string
    df = pd.read_json(data)
    # extract passengerIds
    passenger_ids = df['PassengerId'].ravel()
    # actual survived values
    actuals = df['Survived'].ravel()
    # extract required columns based and convert to matrix
    X = df[columns].values.astype('float')
    # transform the input
    X_scaled = scaler.transform(X)
    # make predictions
    predictions = model.predict(X_scaled)
    # create response dataframe
    df_response = pd.DataFrame({'PassengerId': passenger_ids, 'Predicted' : predictions, 'Actual' : actuals})
    # return json 
    return df_response.to_json()
 

if __name__ == '__main__':
    # host flask app at port 10001
    app.run(port=10001, debug=True)
