from SelectModel import ChooseN, confirm, chooseM

print("##################################################################")
print("#                                                                #")
print("#                HR RESIGNATION PREDICTION WITH ML               #")
print("#                                                                #")
print("#                ---------------------------------               #")
print("#                                                                #")
print("#                       Ahmed Yassine FAIK                       #")
print("#                                                                #")
print("##################################################################")


print("\n################         Choosing The Model       ################")

models = {   1 : "Logistic Regression", 
            2 : "SVM",
            3 : "Random Forest",
            4 : "Neural Network"  }

#
# Printing Models
#
for i in range(1,5):
    print("\n   ",i," :",models[i])

#
# Choosing Models
#
n = ChooseN()
print("\nYou've chosen",models[n])


#
# Confirming Action
#
y = confirm()
if y in ['C','c']:
    chooseM(models, n)
elif y in ['Y','y']:
    n = ChooseN()
    print("\nYou've chosen",models[n])
    y = confirm()