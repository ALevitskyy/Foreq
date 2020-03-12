# -*- coding: utf-8 -*-
"""
Created on Fri Jun 21 11:28:27 2019

@author: s1968881
"""
from neural_net import make_net
import torch
from datamanager import DataManager
import pandas as pd
import datetime
from loss import MaximumLikelyhoodLoss


def neural_net_test():
    net = make_net()
    test_tensor = torch.zeros(2, 3, 256)
    return net(test_tensor)


# tensor = neural_net_test()


def spreadsheet_load(currency):
    dataManager = DataManager()
    dataManager.load_all()
    table = dataManager.raw_data[currency]
    return dataManager, table


def next_candle_test():
    currency = "AUDCAD"
    dataManager, table = spreadsheet_load(currency)
    timestamp = pd.to_datetime("2014-08-28 22:54:00")
    index = table.index.get_loc(timestamp, method="nearest")
    candle = dataManager.get_next_candle("cur1", timestamp, "5T")
    print(table.iloc[index + 1])
    print(table.iloc[index + 5])
    print(table.iloc[index + 5]["C"] - table.iloc[index + 1]["O"])
    print(candle)


# next_candle_test()


def sample_test():
    currency = "AUDCAD"
    dataManager, table = spreadsheet_load(currency)
    timestamp = pd.to_datetime("2016-08-28 22:54:00")
    candle = dataManager.get_sample("cur1", timestamp, "5T")
    print(candle)
    print(table[table.index < timestamp])


# sample_test()


def input_test():
    currency = "AUDCAD"
    dataManager, table = spreadsheet_load(currency)
    dataManager.init_norm_params()
    dataManager.init_splits()
    timestamp = pd.to_datetime("2016-08-28 22:54:00")
    test = dataManager.get_input(currency, timestamp)
    return test, table


# test,table = input_test()
# print(test.shape)


def output_test():
    currency = "AUDCAD"
    dataManager, table = spreadsheet_load(currency)
    dataManager.init_norm_params()
    dataManager.init_splits()
    timestamp = pd.to_datetime("2016-08-28 22:54:00")
    test = dataManager.get_output(currency, timestamp)
    return test, table


# test,table = output_test()
# print(test.shape)


def unprocessed_test():
    currency = "AUDCAD"
    dataManager, table = spreadsheet_load(currency)
    timestamp = pd.to_datetime("2016-08-28 22:54:00")
    test = dataManager.get_unprocessed_sample(currency, timestamp)
    return test, table


# test,table = unprocessed_test()


def random_processed_sample_test():
    currency = "AUDCAD"
    dataManager, table = spreadsheet_load(currency)
    dataManager.init_norm_params()
    dataManager.init_splits()
    now = datetime.datetime.now()
    test = dataManager.get_random_processed_sample()
    print(datetime.datetime.now() - now)
    return test, table


test, table = random_processed_sample_test()
# print(np.mean(np.abs(test["input"]),axis=1))


def normparams_test():
    dataManager = DataManager()
    dataManager.load_all()
    dataManager.init_norm_params()
    return dataManager.norm_params


# norm_params = normparams_test()
# print(norm_params)
def normparams_splits_test():
    dataManager = DataManager()
    dataManager.load_all()
    dataManager.init_norm_params()
    dataManager.init_splits()
    return dataManager.norm_params, dataManager.bins


# norm_params, splits = normparams_splits_test()
# splits_CAD = splits["AUDCAD"]["24H"]
# splits_CHF = splits["AUDCHF"]["24H"]
# print(splits_CAD.mid/norm_params["AUDCAD"]["24H"]["std"])
# print(splits_CHF.mid/norm_params["AUDCHF"]["24H"]["std"])
# print(norm_params)


def loss_test():
    net = make_net()
    test_tensor = torch.zeros(2, 11, 256)
    currency = "AUDCAD"
    dataManager, table = spreadsheet_load(currency)
    dataManager.init_norm_params()
    dataManager.init_splits()
    timestamp = pd.to_datetime("2016-08-28 22:54:00")
    targets = dataManager.get_output(currency, timestamp)
    targets = torch.from_numpy(targets).float()
    outputs = net(test_tensor)
    return MaximumLikelyhoodLoss()(outputs, targets), targets, outputs


# print(loss_test())
