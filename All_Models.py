from sklearn.ensemble import RandomForestRegressor
from sklearn import svm
from sklearn.linear_model import LogisticRegression
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV

#############   LogisticRegression  #############
def FitLogistic(X_train,y_train):
    print("\n     ===>    Create a Logistic Regression Classifier")
    logmodel = LogisticRegression()
    
    print("\n     ===>    Fitting the model with the Training Data")
    logmodel.fit(X_train, y_train)
    
    return logmodel

#############   SVM     #############
def FitSVM(X_train, y_train):
    print("\n     ===>    Create a svm Classifier")
    clf = svm.SVC(kernel='linear')

    print("\n     ===>    Fitting the model with the Training Data")
    clf.fit(X_train, y_train)
    return clf

#############   RandomForrest   #############
def FitForrest(X_train, y_train):
    print("\n     ===>    Create a Random Forest Regressor")
    rf = RandomForestRegressor(n_estimators = 1000, random_state = 42)

    print("\n     ===>    Fitting the model with the Training Data")
    rf.fit(X_train, y_train)
    return rf

#############   NeuralNetworks  #############
def base_model():
    print("\n     ===>    Building the base model")
    model = Sequential()
    model.add(Dense(50, activation='relu', input_dim=9))
    model.add(Dense(50, activation='relu'))
    model.add(Dense(2, activation='sigmoid'))

    print("\n     ===>    Compile the model")
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

def FitNN(base_model, X_train, y_train):
    print("\n     ===>    Create a NN Classifier")
    model = KerasClassifier(build_fn=base_model, verbose=0)
    
    epochs = [50, 100, 150]
    batches = [5, 10, 20]
    param_grid = dict( epochs=epochs, batch_size=batches)
    grid = GridSearchCV(estimator=model, param_grid=param_grid)
    
    print("\n     ===>    Fitting the model with the Training Data")
    Result = grid.fit(X_train, y_train)
    return model

