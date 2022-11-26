import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

data = pd.read_csv('/kaggle/input/predict-diabities/diabetes.csv')
data


data = np.array(data)
m, n = data.shape
print(f'{m,n}')
np.random.shuffle(data) # shuffle before splitting into dev and training sets

data_dev = data[0:20].T
Y_dev = data_dev[-1]
X_dev = data_dev[0:n-1]
X_dev = X_dev / 255.

data_train = data[20:m].T
Y_train = data_train[-1]
X_train = data_train[0:n-1]
X_train = X_train / 255.
_,m_train = X_train.shape
print(X_train.shape)


Y_train.shape

def init_params():
    W1 = np.random.rand(10, 8) - 0.5
    b1 = np.random.rand(10, 1) - 0.5
    W2 = np.random.rand(2, 10) - 0.5
    b2 = np.random.rand(2, 1) - 0.5
    weightlist = []
    weightlist.append((W1,b1))
    weightlist.append((W2,b2))
    return weightlist

def ReLU(Z):
    return np.maximum(Z, 0)

def softmax(Z):
    #print(Z.shape)
    exp = np.exp(Z)/np.max(np.exp(Z),axis=0)
    A = exp / sum(exp)
    return A

def forward_prop(W1, b1, W2, b2, X):
    #W1, b1 = weightlist[0]
    #W2, b2 = weightlist[1]
    Z1 = W1.dot(X) + b1
    A1 = ReLU(Z1)
    Z2 = W2.dot(A1) + b2
    A2 = softmax(Z2)
    #print("forward prop " + f'{A2.shape}')
    return Z1, A1, Z2, A2

def ReLU_deriv(Z):
    return Z > 0
'''
def one_hot(Y):
    one_hot_Y = np.zeros((Y.size, Y.max() + 1))
    one_hot_Y[np.arange(Y.size), Y] = 1
    one_hot_Y = one_hot_Y.T
    return one_hot_Y
'''

def one_hot(Y):
    Y=np.array(Y).astype(int)
    one_hot_Y = np.zeros((int(Y.size), int(Y.max()) + 1))
    one_hot_Y[np.arange(int(Y.size)), Y] = 1
    one_hot_Y = one_hot_Y.T
    return one_hot_Y

def backward_prop(Z1, A1, Z2, A2, W1, W2, X, Y):
    one_hot_Y = one_hot(Y)
    dZ2 = A2 - one_hot_Y
    dW2 = 1 / m * dZ2.dot(A1.T)
    db2 = 1 / m * np.sum(dZ2)
    dZ1 = W2.T.dot(dZ2) * ReLU_deriv(Z1)
    dW1 = 1 / m * dZ1.dot(X.T)
    db1 = 1 / m * np.sum(dZ1)
    return dW1, db1, dW2, db2

def update_params(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha):
    W1 = W1 - alpha * dW1
    b1 = b1 - alpha * db1
    W2 = W2 - alpha * dW2
    b2 = b2 - alpha * db2
    return W1, b1, W2, b2


def get_predictions(A2):
    return np.argmax(A2, 0)

def get_accuracy(predictions, Y):
    #print(predictions, Y)
    return np.sum(predictions == Y) / Y.size

def gradient_descent(X, Y, alpha, iterations):
    weightlist = init_params()
    W1, b1 = weightlist[0]
    W2, b2 = weightlist[1]
    for i in range(iterations):
        Z1, A1, Z2, A2 = forward_prop(W1,b1,W2,b2, X)
        dW1, db1, dW2, db2 = backward_prop(Z1, A1, Z2, A2, W1, W2, X, Y)
        W1, b1, W2, b2 = update_params(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha)
        if i % 10 == 0:
            print("Iteration: ", i)
            predictions = get_predictions(A2)
            print(get_accuracy(predictions, Y))
    return W1, b1, W2, b2

#print(X_train.shape)
print(Y_train.shape)
W1, b1, W2, b2 = gradient_descent(X_train, Y_train, 0.10, 2000)
