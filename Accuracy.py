
from keras.wrappers.scikit_learn import KerasClassifier

def accurate(model, X_train, X_test, y_train, y_test):
    pred_train= model.predict(X_train)
    scores = model.score(X_train, y_train)

    print('Accuracy on training data: {}% \n Error on training data: {}'.format(scores, 1 - scores))
    
    pred_test= model.predict(X_test)
    scores2 = model.score(X_test, y_test)
    print('Accuracy on test data: {}% \n Error on test data: {}'.format(scores2, 1 - scores2))