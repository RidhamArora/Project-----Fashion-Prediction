# Importing the Libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('fashion-mnist_train.csv')
data = dataset.values
X = data[:,1:]
Y = data[:,0]

"""# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)"""

# Fitting Logistic Regression to the Training set
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(multi_class='multinomial',random_state = 0,solver = 'saga')
classifier.fit(X, Y)

# Predicting the Test set results
y_pred = classifier.predict(X)

cnt=0
for i in range(X.shape[0]):
    if(y_pred[i]==Y[i]):
        cnt=cnt+1
print(cnt)
