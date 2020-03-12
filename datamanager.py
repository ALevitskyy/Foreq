# -*- coding: utf-8 -*-
"""
Created on Fri Jun 21 11:34:28 2019

@author: s1968881
"""

from pathlib import Path
import pandas as pd
from dateutil.relativedelta import relativedelta
import random
import numpy as np
import collections


def load_spreadsheet(csv_path):
    table = pd.read_csv(
        csv_path,
        sep=";",
        header=None,
        names=["O", "H", "L", "C", "V"],
        parse_dates=True,
    )
    return table[["O", "C"]]


def random_timestamp(start, end):
    seconds = (end - start).total_seconds()
    seconds = random.randint(0, int(seconds))
    return start + relativedelta(seconds=seconds)


def train_start_end_func(time_index):
    start = time_index.min() + relativedelta(months=16)
    end = time_index.max() - relativedelta(years=1)
    return [start, end]


def val_start_end_func(time_index):
    start = time_index.max() - relativedelta(years=1)
    end = time_index.max() - relativedelta(months=6)
    return [start, end]


horizons = collections.OrderedDict()
horizons["1T"] = 1
horizons["5T"] = 5
horizons["15T"] = 15
horizons["30T"] = 30
horizons["1H"] = 60
horizons["2H"] = 120
horizons["4H"] = 240
horizons["8H"] = 480
horizons["12H"] = 720
horizons["16H"] = 960
horizons["24H"] = 1440


class DataManager:
    def __init__(
        self,
        bins=32,
        start_end_func=train_start_end_func,
        data_path="./reduced",
        pred_horizons=["15T", "30T", "1H", "2H", "4H", "8H", "12H", "16H", "24H"],
        horizons=horizons,
        channel_size=256,
    ):
        self.currency_gen = Path(data_path).glob("./*")
        self.start_end_func = start_end_func
        self.raw_data = {}
        self.start_end = {}
        self.norm_params = {}
        self.splits = {}
        self.Nbins = bins
        self.bins = {}
        self.pred_horizons = pred_horizons
        self.horizons = horizons
        self.channel_size = channel_size

    def load_currency(self, currency_path):
        file_list = []
        for file in currency_path.glob("./*.csv"):
            file_list.append(load_spreadsheet(file))
        return pd.concat(file_list).sort_index()

    def load_all(self):
        for currency in self.currency_gen:
            if "." not in currency.name:
                self.norm_params[currency.name] = {}
                self.bins[currency.name] = {}
                self.raw_data[currency.name] = self.load_currency(currency)
        self.calculate_start_end()

    def calculate_start_end(self):
        for currency in self.raw_data:
            table = self.raw_data[currency]
            start_end = self.start_end_func(table.index)
            self.start_end[currency] = start_end

    def get_random_currency(self):
        currency_list = list(self.raw_data.keys())
        return random.choice(currency_list)

    def get_random_timestamp(self, currency):
        start, end = self.start_end[currency]
        random_time = random_timestamp(start, end)
        return random_time

    def get_norm_params(self, currency, horizon):
        table = (
            self.raw_data[currency]
            .resample(horizon)
            .aggregate({"C": "last", "O": "first"})
            .dropna()
        )
        candles = table["C"] - table["O"]
        self.norm_params[currency][horizon] = {
            "mean": np.mean(candles),
            "std": np.std(candles),
        }

    def get_splits(self, currency, horizon):
        table = (
            self.raw_data[currency]
            .resample(horizon)
            .aggregate({"C": "last", "O": "first"})
            .dropna()
        )
        candles = table["C"] - table["O"]
        category = pd.qcut(candles, self.Nbins).values.categories
        self.bins[currency][horizon] = category

    def init_norm_params(self):
        for currency in self.raw_data.keys():
            for horizon in self.horizons.keys():
                self.get_norm_params(currency, horizon)

    def init_splits(self):
        for currency in self.raw_data.keys():
            for horizon in self.pred_horizons:
                self.get_splits(currency, horizon)

    def binarize(self, value, currency, horizon):
        dummy = pd.get_dummies(pd.cut([value], self.bins[currency][horizon]))
        return np.array(dummy).squeeze()

    def get_sample(self, currency, timestamp, horizon):
        table = self.raw_data[currency]
        reduced_table = table[table.index < timestamp].iloc[
            -(self.horizons[horizon] * self.channel_size) :
        ]
        candles = (
            reduced_table.resample(horizon)
            .aggregate({"C": "last", "O": "first"})
            .dropna()
            .iloc[-self.channel_size :]
        )
        return candles["C"] - candles["O"]

    def get_next_candle(self, currency, timestamp, horizon):
        rounded_timestamp = timestamp.ceil(horizon)
        table = self.raw_data[currency]
        reduced_table = table[table.index >= rounded_timestamp].iloc[
            0 : self.horizons[horizon]
        ]
        candle = (
            reduced_table.resample(horizon)
            .aggregate({"C": "last", "O": "first"})
            .dropna()
            .iloc[0]
        )
        return candle["C"] - candle["O"]

    def get_input(self, currency, timestamp):
        input_list = []
        for horizon in self.horizons.keys():
            horizon_seria = self.get_sample(currency, timestamp, horizon)
            mean = self.norm_params[currency][horizon]["mean"]
            std = self.norm_params[currency][horizon]["std"]
            horizon_seria = (horizon_seria - mean) / std
            input_list.append(horizon_seria.values.squeeze())
        return np.array(input_list)

    def get_output(self, currency, timestamp):
        output_list = []
        for horizon in self.pred_horizons:
            horizon_candle = self.get_next_candle(currency, timestamp, horizon)
            horizon_candle = self.binarize(horizon_candle, currency, horizon)
            output_list.append(horizon_candle)
        return np.array(output_list)

    def get_processed_sample(self, currency, timestamp):
        input_matrix = self.get_input(currency, timestamp)
        output_vector = self.get_output(currency, timestamp)
        return {"input": input_matrix, "output": output_vector}

    def get_random_processed_sample(self):
        currency = self.get_random_currency()
        timestamp = self.get_random_timestamp(currency)
        print(timestamp)
        return self.get_processed_sample(currency, timestamp)
