import numpy as np

def indentity_funtion(x):
    return x

def step_function(x):
    return np.array(x > 0, dtype=np.int)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_grad(x):
    return sigmoid(x) * (1-sigmoid(x))

def relu(x):
    return np.maximum(x, 0)

def relu_grad(x):
    grad = np.zreos(x)
    grad[x>=0] = 1
    return grad
    
def softmax(x):
    if x.ndim == 2:     #若x為2維
        x = x.T     #為了運算將x轉置
        x = x - np.max(x, axis=0)
        y = np.exp(x) / np.sum(np.exp(x), axis=0)
        return y.T
    
    x = x - np.max(x)
    return np.exp(x) / np.sum(np.exp(x))
    
    
def mean_square_error(y, t):    #均方誤差
    if y.ndim == 2:
        return 0.5*(np.sum(y - t)**2) / y.shape[0]
    return   0.5*(np.sum(y - t)**2)

def cross_entropy_error(y, t):
    if  y.ndim == 1:
        y = y.reshape(1, y.size)
        t = t.reshape(1, t.size) 
        
    #若使用one-hot標籤，轉成非one-hot
    if t.size == y.size:
        t = t.argmax(axis=0)
        
    batch_size = y.shape[0]
    return -np.sum(np.log(y[np.arange(batch_size), t] + 1e-7))  / batch_size

def sofmax_loss(x, t):
    y = softmax(x)
    return cross_entropy_error(y, t)



        
    
    

    
