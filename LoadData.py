
import pandas   as pd
import numpy    as np

from sklearn.model_selection import train_test_split
from keras.utils import to_categorical

path1 = 'HR_test.csv'
path2 = 'HR_train.csv'

def ReadData(p1, p2):
    print("\n################            Looding Data          ################")
    print("\n     ===>    Reading CSV files")
    
    df1 = pd.read_csv(p1)
    df2 = pd.read_csv(p2)
    
    li = [df1, df2]
    df = pd.concat(li, axis=0, ignore_index=True)
    df.describe()
    print("\n             DataFrame dimensions: ", df.shape)
    
    print("\n     ===>    Encoding Categorical Variables")

    df['department'] = df['department'].astype('category')
    df['salary'] = df['salary'].astype('category')
    
    categorical = df.select_dtypes(['category']).columns        # = ['department','salary']
    df[categorical] = df[categorical].apply(lambda x: x.cat.codes)
    
    return df

def SetFeatures(df):
    print("\n     ===>    Setting up features")

    target_column = ['left']
    to_eliminate = ['left','last_evaluation','number_project','average_montly_hours','time_spend_company','satisfaction_level']
    predictors = list(set(list(df.columns))-set(target_column)) #eliminate the target
    df[predictors] = df[predictors]/df[predictors].max()        #normalization
    df.describe()

    X = df[predictors].values
    y = df[target_column].values
    
    return X, y, df, predictors, target_column

def SplitData(X, y):
    print("\n     ===>    Splitting Data")

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=40)
    print("\n             Train Data dimensions: ",X_train.shape)
    print("\n             Test Data dimensions: ",X_test.shape)
    return X_train, X_test, y_train, y_test

def EncodeOutputs(y_train, y_test):
    print("\n     ===>    Get one hot encoding")
    y_train = to_categorical(y_train)
    y_test = to_categorical(y_test)
    count_classes = y_test.shape[1]
    print("\n             Output classes numbre: ",count_classes)
    return y_train, y_test
