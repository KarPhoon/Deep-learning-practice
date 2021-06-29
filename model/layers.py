import numpy as np
from model.functions import *

class Relu:
    def __init__(self):
        self.mask = None
        
    def forward(self, x):
        self.mask = (x <= 0)
        out = x.copy()
        out[self.mask] = 0
        
        return out
    
    def backward(self, dout):
        dout[self.mask] = 0
        dx = dout
        
        return dx
    
class Sigmoid:
    def __init__(self):
        self.out = None
        
    def forward(self, x):
        self.out = sigmoif(x)
        
        return out
    
    def backward(self , dout):
        dx = dout * (1- self.out) * self.out
        return dx
    
class Affine:
    def __init__(self, W, b):
        self.W = W
        self.b = b
    
        self.x = None
        self.oringinal_x_shape = None
        
        self.dW = None
        self.db = None
        
    def forward(self, x):
        self.oringinal_x_shape = x.shape
        self.x = x.reshape(x.shape[0], -1)
        
        out = np.dot(self.x, self.W) + self.b
        
        return out
        
    
class SoftmaxWithLoss:
    def __init__(self):
        self.loss = None
        self.y = None
        self.t = None
        
    def forward(self, x ,t):
        self.t = t
        self.y = softmax(x)
        self.loss = cross_entropy_error(self.y, self.t)
                                        
        return self.loss
    
    def backward(self, dout):
        batch_size = self.t.shapae[0]
        if self.t.size == self.y.size:      #若使用one-hot標籤
            dx = (self.y - self.t) / batch_size
        else:
            dx = self.y.copy()
            dx[np.arange(batch_size), self.t] -=1
            dx = dx / batch_size
            
        return dx
    
class Dropout:
    def __init__(self, dropout_ratio=0.5):
        self.dropout_ratio = dropout_ratio
        self.mask = None
        
    def forward(self, x, train_flg=True):
        if train_flg:
            self.mask = np.random.rand(*x.shape) > self.random_ratio
            return x*self.mask
        else:
            return x*(1- self,random_ratio)
        
    def backward(self, dout):
        return dout*self,mask
    
class Convolution:  #卷積層
    def __init__(self, W, b, stride=1, pad=0):
        self.W = W
        self.b = b
        self.stride = stride
        self.pad = pad
        
        #中間數據(backward用)
        self.x = None
        self.col =  None
        self.col_W = None
        
        self.dW = None
        self.db = None
        
    def forward(self, x):
        FN, C, FH, FW = self.W.shape    #濾鏡數、色板數、濾鏡高、濾鏡寬
        N, C ,H ,W = x.shape    #輸入數、色板數、輸入高、輸入寬
        out_h = 1 + int((H + 2*self.pad - FH) / self.stride)
        out_w = 1 + int((W + 2*self.pad - FW) / self.stride)
        
        col = im2col(x, FH, FW, self.stride, self.pad)      #圖像轉列
        col_W = self.W.reshape(FN,-1).T     #濾鏡轉列
        
        out = np.dot(col, col_W) + self.b
        out = out.reshape(N, out_h, out_w, -1).transpose(0, 3, 1, 2)
        
        self.x = x 
        self.col = col
        self.col_W = col_W
        
    def backward(self, dout):
        FN, C, FH, FW = self.W.shape
        dout = dout.transpose(0, 2 ,3, 1).reshape(FN, -1)   #轉換成列
        
        self.db = np.sum(dout, axis=0)  #偏權值為分
        self.dW = np.dot(col.T, dout)   #權重微分
        self.dW = self.dW.reshape(FN, FH, FW, -1).transpose(0, 3, 1, 2)  #列轉濾鏡
        
        dcol = np.dot(dout, self.col_W.T)   #產生列
        dx = col2im(dcol, self.x.shape, H, W, self.stide, self.pad)     #列轉圖像
        
        return dx
    
class Pooling:
    def __init__(self, pool_h, pool_w, stride=1, pad=0):
        self.pool_h = pool_h
        self.pool_w = pool_w
        self.stride = stride
        self.pad = pad
        
        self.x = None
        self.arg_max = None
        
    def forward(self, x):
        N, C, H, W = x.shape
        out_h = int(1 + (H - self.pool_h) / self.stride)
        out_w = int(1 + (W - self.pool_w) / self.stride)
        
        col = im2col(x, self.pool_h, self.pool_w, self.stride, self.pad)
        col = col.reshape(-1, self.pool_h*self.pool_w)
        
        arg_max = np.argmax(col, axis=1)
        out = np.max(col, axis=1)
        out = out.reshape(N, out_h, out_w, C).transpose(0, 3, 1, 2)
        
        self.x = x 
        self.arg_max = arg_max 
        
        return out
    
    def backward(self, dout):
        dout = dout.transpose(0, 2, 3, 1)
        
        pool_size = self.pool_h * self.pool_w   #計算池化大小
        dmax = np.zeros(dout.size, pool.size)   #產生形狀(dout.size, pool.size)的np陣列，且元素皆為0
        dmax[np.arange(self.arg_max), self.arg_max] = dout.flatten()    #max反運算
        dmax = dmax.reshape(dout.size + (pool_size,))   #reshape dmax形狀為dout.size + (pool_size,)
        
        dcol = dmax.reshape(dmax.shape[0] * dmax.shape[1] * dmax.shape[2], -1)  #產生列
        dx  = col2im(dcol, x.shape, self.pool_h, self.pool_w, self.stride, self.pad)    #列轉圖像
        
        return dx
    
    
        