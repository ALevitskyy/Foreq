# -*- coding: utf-8 -*-
"""
Created on Fri Jun 21 11:28:27 2019

@author: s1968881
"""
from neural_net import make_net
import torch
from datamanager import DataManager
import pandas as pd
def neural_net_test():
    net = make_net()
    test_tensor = torch.zeros(2,3,256)
    return net(test_tensor)
#tensor = neural_net_test()

def spreadsheet_load():
    dataManager = DataManager()
    dataManager.load_all()
    table = dataManager.raw_data["cur1"]
    return dataManager, table

def next_candle_test():
    dataManager, table = spreadsheet_load()
    timestamp = pd.to_datetime("2014-08-28 22:54:00")
    index = table.index.get_loc(timestamp, method = "nearest")
    candle = dataManager.get_next_candle("cur1", timestamp, "5T")
    print(table.iloc[index+1])
    print(table.iloc[index+5])
    print(table.iloc[index+5]["C"]-table.iloc[index+1]["O"])
    print(candle)
#next_candle_test()
    
def sample_test():
    dataManager, table = spreadsheet_load()
    timestamp = pd.to_datetime("2016-08-28 22:54:00")
    candle = dataManager.get_sample("cur1", timestamp, "5T")
    print(candle)
    print(table[table.index<timestamp])
#sample_test()
    
