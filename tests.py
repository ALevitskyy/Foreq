# -*- coding: utf-8 -*-
"""
Created on Fri Jun 21 11:28:27 2019

@author: s1968881
"""
from neural_net import make_net
import torch
def neural_net_test():
    net = make_net()
    test_tensor = torch.zeros(2,3,256)
    print(net(test_tensor).shape)
neural_net_test()