# -*- coding: utf-8 -*-
import sys, os 
sys.path.append(os.pardir)
import numpy as np
from yyy.mnist import load_mnist
from PIL import Image


def img_show(img):
    pil_img = Image.fromarray(np.uint8(img))
    pil_img.show()  # 顯示影像
    
(x_train, t_train), (x_test, t_test) = load_mnist(flatten=True, normalize=False)
    
img = x_train[0]
label = t_train[0]
print(label)

print(img.shape)    #原影像一維陣列形狀(784,)
img=img.reshape(28,28)  #變換回原本影像大小
print(img.shape)    #(28,28)
 
img_show(img)
