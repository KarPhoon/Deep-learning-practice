# -*- coding: utf-8 -*-
"""
Created on Fri Dec 21 16:54:26 2018

@author: 詹凱丰
"""

from fun import *

def mean_squared_error(y, t):
    return 0.5*np.sum((y-t)**2)

def cross_entropy_errpr(y, t):
    delta = 1e-7
    return -np.sum(t * np.log(y+delta))



    