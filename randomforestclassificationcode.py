import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#importing the data set
dataset = pd.read_csv('Data.csv')

#matrix of features mane, indipendent o dependent variable gula define kortesi ekhane
#X hocche independent, Y hocche dependent
X = dataset.iloc[:, :-1].values
Y = dataset.iloc[:,6].values

#Missing Data handling
from sklearn.impute import SimpleImputer
# object toiri kortesi
missingvalues = SimpleImputer(missing_values = np.nan, strategy = 'mean', verbose = 0)
# akhon X a fit kortechi....upperbound is exculded...tai 3 dewar por o 1 theke 2 porjonto hobe,,,so CAREFUL
missingvalues = missingvalues.fit(X[:, 0:6])
X[:, 0:6]=missingvalues.transform(X[:, 0:6])


# Encoding Y data
from sklearn.preprocessing import LabelEncoder
Y = LabelEncoder().fit_transform(Y)

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.30, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)


# Fitting Random Forest Classification to the Training set
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 0)
classifier.fit(X_train, Y_train)


# Predicting the Test set results
Y_pred = classifier.predict(X_test)


# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm_random_forest_classification = confusion_matrix(Y_test, Y_pred)


#Measure Accuracy
from sklearn.metrics import accuracy_score 
print ("Accuracy : ", accuracy_score(Y_test, Y_pred)) 


