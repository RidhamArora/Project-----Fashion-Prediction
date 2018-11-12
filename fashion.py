# Importing the Libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('fashion-mnist_train.csv')
dataset2 = pd.read_csv('fashion-mnist_test.csv')
data = dataset.values
data2 = dataset2.values
X = data[:10000,1:]
Y = data[:10000,0]
X_test = data2[1000:2000,1:]
Y_test = data2[1000:2000,0]
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X = sc.fit_transform(X)
def draw_img(X):
    img = X.reshape(28,28)
    plt.imshow(img,cmap='gray')
    plt.show()

def hypothesis(x,w,b):
    h=np.dot(x,w)+b
    return sigmoid(h)

def sigmoid(z):
    zz = 1.0/(1.0+np.exp(-1.0*z))
    return zz

def error(Y,x,w,b,a,c):
    
    m=x.shape[0]
    err=0.0
    cnts = 0
    for i in range(m):
        if(Y[i]==a or Y[i]==c):
            hx = hypothesis(x[i],w,b)
            if(Y[i]==c):
                err += np.log2(hx)
            else:
                err += np.log2(1-hx)
            cnts = cnts + 1
    return -err/cnts

def get_grads(Y,x,w,b,a,c):
    
    grad_w = np.zeros(w.shape)
    grad_b = 0.0
    
    m = x.shape[0]
    
    cnts=0
    for i in range(m):
        if(Y[i]==a or Y[i]==c):
            hx = hypothesis(x[i],w,b)
            if(Y[i]==c):
                grad_w += (1-hx)*x[i]
                grad_b += (1-hx)
            if(Y[i]==a):
                grad_w += (0-hx)*x[i]
                grad_b += (0-hx)
            cnts = cnts + 1
    grad_w /= cnts
    grad_b /= cnts
    return [grad_w,grad_b]

def batch_gradient(Y,x,w,b,a,c,batch_size=1):
    
    grad_w = np.zeros(w.shape)
    grad_b = 0.0
    m=x.shape[0]
    indices=np.arange(m)
    np.random.shuffle(indices)
    indices=indices[:batch_size]
    cnts=0
    for i in indices:
        if(Y[i]==a or Y[i]==c):
            hx = hypothesis(x[i],w,b)
            if(Y[i]==a):
                grad_w += (1-hx)*x[i]
                grad_b += (1-hx)
            if(Y[i]==c):
                grad_w += (0-hx)*x[i]
                grad_b += (0-hx)
            cnts = cnts + 1
    grad_w /= cnts
    grad_b /= cnts
    #print(cnts)
    return [grad_w,grad_b]

def grad_descent(x,Y,w,b,a,c,learning_rate=0.1):
    
    err = error(Y,x,w,b,a,c)
    [grad_w,grad_b] = get_grads(Y,x,w,b,a,c)
    
    w = w + learning_rate*grad_w
    b = b + learning_rate*grad_b
    
    return err,w,b

def logistic(x,Y,a,c):
    loss = []
    acc = []
    w=np.zeros((x.shape[1],))
    b=np.random.random()
    l,w,b = grad_descent(x,Y,w,b,a,c,learning_rate = 0.001)
    while(l>0.05):
        l,w,b = grad_descent(x,Y,w,b,a,c,learning_rate = 0.1)
        loss.append(l)
    
    return loss,w,b

def classify_it(x,Y):
    Z = np.unique(Y,return_counts=True)
    Z = np.array(Z)
    #print(Z)
    m = Z.shape[1]
    dic={}
    cnt=0
    for i in range(m):
        for j in range(i+1,m):
         final_loss,final_w,final_b = logistic(x,Y,i,j)
         #print(final_loss)
         plt.plot(final_loss)
         plt.show()
         dic[(i,j)] = (final_w,final_b)
         cnt = cnt + 1
         print("done ",cnt)
         
    return dic

    
dic = classify_it(X,Y)


fin = []

def tell_me(X,Y,z):
    global dic
    global fin
    Z = np.unique(Y,return_counts=True)
    Z = np.array(Z)
    cnt = {}
    ans = 0
    ans2 = 0
    m = Z.shape[1]
    for i in range(m):
        cnt[Z[0][i]]=0
    for key in dic:
        #fin.append(hypothesis(z,dic[key][0],dic[key][1]))
        if(hypothesis(z,dic[key][0],dic[key][1])>0.5):
            cnt[key[1]]=cnt[key[1]]+1
        else:
            cnt[key[0]]=cnt[key[0]]+1
    for i in cnt:
        if (cnt[i]>ans):
            ans = cnt[i]
            ans2=i
    return ans2
def comp_acc(X,Y):
    hey = 0
    for i in range(X.shape[0]):
        if(tell_me(X,Y,X[i]) == Y[i]):
            hey = hey + 1
    return (1.0*hey)/(1.0*X.shape[0])
print("Training Accuracy is : ",end=' ')
print(comp_acc(X,Y))
print("Training Accuracy is : ",end=' ')    
print(comp_acc(X_test,Y_test))

