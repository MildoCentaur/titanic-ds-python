#import kaggle

#kaggle.KaggleApi competitions download -c titanic


import pandas as pd
import numpy as np
import os
# set the file paths
raw_data_path = os.path.join(os.path.pardir,os.path.pardir,'data','raw')
train_data_path = os.path.join(raw_data_path, 'train.csv')
test_data_path = os.path.join(raw_data_path, 'test.csv')
print (train_data_path)
open(train_data_path,"r").readline()


#read data with default parameters using panda df
train_df = pd.read_csv(train_data_path,index_col="PassengerId")
test_df = pd.read_csv(test_data_path,index_col="PassengerId")

#get the type of the DF to confirm that we have propertly loaded the dataframe
type(train_df)
type(test_df)
train_df.info()
print (" the size of the datasset is ", train_df.size, " and the dimension is", train_df.ndim, " Shape is ", train_df.shape)
print ("the amount of rows per column is\n",train_df.count())

train_df.describe()
test_df['Survived']=-888
df = pd.concat((test_df,train_df), axis=0, sort=True)
import matplotlib.pyplot as plt

plt.rcParams['figure.figsize'] = [15, 10]
# use hist to create histogram
df.Age.plot(kind='hist', title='histogram for Age', color='c',xticks=range(0,80,5), bins=16);

print(df.loc[(df.Age>=0) & (df.Age<5),].Age.count())
print (df.loc[(df.Age>=5) & (df.Age<10),].Age.count())
print(df.loc[(df.Age>=10) & (df.Age<15),].Age.count())
print (df.loc[(df.Age>=15) & (df.Age<20),].Age.count())

df.plot.scatter(x='Pclass', y='Fare', color='c', title='Scatter plot : Passenger class vs Fare', alpha=0.15, norm='Normilize');

df.loc[df['Age']<1,['Age']]
#df.loc[df['Age']<1,['Age']] = df.loc[df['Age']<1,['Age']]*100
## df.loc[df['age']==3, ['age-group']] = 'toddler'


print(df.groupby(pd.cut(df["Age"], range(0, 100, 5))))

test = pd.DataFrame({'days': [0,31,45]})
test['range'] = pd.cut(test.days, [0,30,60])


df.applymap(lambda x: x**2)

df["AgeRange2"]= df["Age"].map(lambda x : ceil(x/5) -1)

import math

from sklearn.model_selection import GridSearchCV

GridSearchCV.cv_results_