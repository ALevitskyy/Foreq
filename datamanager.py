# -*- coding: utf-8 -*-
"""
Created on Fri Jun 21 11:34:28 2019

@author: s1968881
"""

from pathlib import Path
import pandas as pd
from dateutil.relativedelta import relativedelta
import random

def load_spreadsheet(csv_path):
    table = pd.read_csv(csv_path, 
                        sep=";",
                        header = None,
                        names = ["O","H","L","C","V"],
                       parse_dates = True)
    return table[["O", "C"]]

def random_timestamp(start, end):
    seconds = (end-start).total_seconds()
    seconds = random.randint(0,int(seconds))
    return start + relativedelta(seconds = seconds)


class DataManager:
    def __init__(self, 
                 data_path = "./data",
                horizons = {"1T":1,"5T":5,"15T":15,
                            "30T":30,"1H":60,"2H":120,
                           "4H":240, "8H":420, "12H":720,
                           "16H":960, "24H":1440},
                channel_size = 256):
        self.currency_gen = Path(data_path).glob("./*")
        self.raw_data = {}
        self.start_end = {}
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
                self.raw_data[currency.name] = self.load_currency(currency)
                table = self.raw_data[currency.name]
                start = table.index.min()+relativedelta(months = 16)
                end = table.index.max()-relativedelta(years = 1)
                self.start_end[currency.name] = [start, end]
    
    def get_random_currency(self):
        pass
        
    def get_random_timestamp(self, currency):
        start, end = self.start_end[currency]
        random_time = random_timestamp(start, end)
        return random_time
        
    def get_norm_params(self,
                        currency,
                        horizon):
        pass
    
    def get_splits(self,
                  currency,
                  horizon):
        pass
    
    def get_sample(self, currency, timestamp , horizon):
        table = self.raw_data[currency]
        reduced_table = table[table.index<timestamp]\
                        .iloc[-(self.horizons[horizon]*self.channel_size):]
        candles = reduced_table.resample(horizon)\
                           .aggregate({"C":"last","O":"first"})\
                           .dropna().iloc[-self.channel_size:]
        return candles["C"] - candles["O"]
        
    def get_next_candle(self,currency, timestamp, horizon):
        rounded_timestamp = timestamp.ceil(horizon)
        table = self.raw_data[currency]
        reduced_table = table[table.index>=rounded_timestamp]\
                                    .iloc[0:self.horizons[horizon]]
        candle = reduced_table.resample(horizon)\
               .aggregate({"C":"last","O":"first"}).dropna().iloc[0]
        return candle["C"] - candle["O"]
    
    def get_input(self, currency, timestamp):
        pass
    
    def get_output(self, currency, timestamp):
        pass
    
    def get_unprocessed_sample(self, currency, timestamp):
        pass
    
    def get_random_unprocessed_sample(self):
        pass

def norm_params_stability():
    pass

def splits_stability():
    pass
