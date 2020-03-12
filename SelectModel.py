from LoadData           import  ReadData, SetFeatures, SplitData, EncodeOutputs
from All_Models      import  base_model, FitNN, FitLogistic, FitSVM, FitForrest
'''
from NeuralNetwork      import  base_model, FitNN
from LogisticRegression import  FitLogistic
'''
from Accuracy           import  accurate
            
def ChooseN():
    n = 0
    while n not in range(1,5):
        n = int(input("\nPlease type the number of the model you want : "))
    return n

def confirm():
    y = str(input("\nIf you want to change it type Y, else type C : "))
    while y not in ['Y','C','y','c']:
        y = str(input("\nIf you want to change it type Y, else type C : "))
    return y

def chooseM(model, modeli):
    print("\nYour model is : " , model[modeli])

    our_model = model[modeli].replace(" ","")
    
    path1 = 'HR_test.csv'
    path2 = 'HR_train.csv'

    df = ReadData(path1, path2)

    X, y, df, predictors, target = SetFeatures(df)

    X_train, X_test, y_train, y_test = SplitData(X, y)

    l = [X, y, df, predictors, target, X_train, X_test, y_train, y_test]
    
    print("\n################         Training The Model       ################")
    if modeli == 1:
        model = FitLogistic(X_train, y_train)
    elif modeli == 2:
        model = FitSVM(X_train, y_train)
    elif modeli == 3 or modeli == 4:
        y_train, y_test = EncodeOutputs(y_train, y_test)
        if modeli == 3:
            model = FitForrest(X_train, y_train)
        if modeli == 4:
            model = FitNN(base_model, X_train, y_train)
    
    print("\n################          Accuracy Summary        ################")
    accurate(model, X_train, X_test, y_train, y_test)
    
