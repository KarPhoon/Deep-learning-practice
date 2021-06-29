import numpy as np
from collections import OrderedDict
from commom.layers import *

class SimpleConvNet:
    def __init__(self, input_dim=(1, 28 ,28), 
                 conv_param={'filter_num':30, 'filter_size':5, 'pad':0, 'stride':1},
                 hidden_size=100 , output_size=10 , weight_init_std=0.01):
        filter_num = conv_param['filter_num']       #濾鏡數
        filter_size = conv_param['filter_size']     #濾鏡大小
        filter_pad = conv_param['pad']        #填補大小
        filter_stride = conv_param['stride']        #步輻
        input_size = input_dim[1]   #輸入大小
        conv_output_size = (input_size - filter_size + 2*filter_pad) / filter_stride + 1    #輸出大小
        pool_output_size = int(filter_num * (conv_output_size/2) * (conv_output_size/2))    #池化大小(卷積視窗大小為2):濾鏡數*(卷積輸出大小/2)^2
        
        self.params = {}
        self.params['W1'] = weight_init_std * \
                            np.random.randn(filter_num, input_dim[0], filter_size, filter_size)     #權重1:(濾鏡數,色板數,濾鏡大小,濾鏡大小)
        self.params['b1'] = np.zeros(filter_num)    #
        self.params['W2'] = weight_init_std * \
                            np.random.randn(pool_out_size,hidden_size)      #權重2:(池化層,隱藏層)
        self.params['b2'] = np.zeros(hidden_size)
        self.paramsp['W3'] = wiegjt_init_std * \篇權值
                             np.random.randn(hidden_size, output_size)      #權重3:(隱藏層,輸出層)
        self.params['b3'] = np.zeros(output_size)
        
        
        
        產生個個層級
        self.layers = OrderedDict
        self.layers['Conv1'] = Convolution(self.params['W1'],           #卷積層
                                           self.params['b1'],
                                           conv_param['stride'],
                                           conv_param['pad'])
        self.layers['Relu1'] = Relu()
        self.layers['Pool1'] = Pooling(pool_h=2, pool_w=2, strdie=2)    #2*2池化
        self.layers['Affine1'] = Affine(self.params['W2'],              #全連接層
                                        self.params['b2'])
        self.layers['Rule2'] = Relu()
        self.layers['Affine2'] = Affine(self.params['W3'],      
                                        self.params['b3'])
        self.last_layer = SoftmaxWithLoss()
        
    def predict(self, x):
        for layer in self.layers.values():
            x = layer.forward(x)
            return x
        
    def loss(self, x , t):
        y = self.predict(x)
        return self.lastLayer.forward(y, t)
    
    def gradinet(self, x, t):
        #forard
        self.loss(x, t)
        
        #backward
        dout = 1 
        dout = self.last_layer.backward(dout)
        
        layers = list(self.layer.valuse())
        layers.reverse()
        for layer in layers:
            dout = layer.backward(dout)
            
        grads = {}
        grads['W1']  = self.layers['Conv1'].dW
        grads['b1']  = self.layers['Conv1'].db
        grads['W2']  = self.layers['Affine1'].dW
        grads['b2']  = self.layers['Affine1'].db
        grads['W3']  = self.layers['Affine2'].dW
        grads['b3']  = self.layers['Affnie2'].db
        
        return grads
    
        
        
        
        
        
        