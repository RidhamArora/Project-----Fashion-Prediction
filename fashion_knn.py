# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('fashion-mnist_train.csv')
data = dataset.values
X = data[:10000,1:]
Y = data[:10000,0]

# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = .2, random_state = 1)

def draw_img(X):
    img = X.reshape(28,28)
    plt.imshow(img,cmap='gray')
    plt.show()

def dist(X,Y):
    return np.sqrt(sum((X-Y)**2))

def KNN(X,Y,query_point,k=5):
    
    vals = []
    m = X.shape[0]
    
    for i in range(m):
        d = dist(query_point,X[i])
        vals.append((d,Y[i]))
    
    vals = sorted(vals)
    
    vals = vals[:k]
    vals = np.array(vals)
    
    new_vals = np.unique(vals[:,1],return_counts=True)
    
    index = new_vals[1].argmax()
    pred = new_vals[0][index]

    return int(pred)
    
def compute_accuracy(X_train,Y_train,X_test,Y_test):
    
    m = X_test.shape[0]
    cnt=0
    for i in range(m):
        if(KNN(X_train,Y_train,X_test[i])==Y_test[i]):
            cnt=cnt+1
        print(cnt)
    return (1.0*cnt)/(1.0*m)

print(compute_accuracy(X_train,Y_train,X_test,Y_test))