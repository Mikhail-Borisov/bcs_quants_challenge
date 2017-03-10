import pandas as pd
import numpy as np

from TestSamplesGeneration.DataPreprocessingForNonlinear import DataPreprocessingForNonlinear
from TestSamplesGeneration.Utils import Frequency, Tickers

"""
This resulted to very bad performance compared to full open high low information set, across all assets
"""
class DataPreprocessingForLinear(DataPreprocessingForNonlinear):
    def __init__(self, frequency = Frequency.m1, backward_lags = 5, forward_lag = 5, hour_dummies = True):
        super(DataPreprocessingForLinear, self).__init__(frequency, backward_lags, forward_lag, hour_dummies)
        self.columns_for_lags = [self.VWAP, self.COUNT, self.VOLUME, self.CLOSE + '_relative']

    def get_full_ticker_data(self, ticker_name = Tickers.USD000UTSTOM, sample_size = 0.5, use_first_sample = True):
        raw_data = self.get_raw_ticker_data(ticker_name)
        data = self.get_weekly_and_daytime_parameters(raw_data)
        data[self.VWAP] = ((data[self.OPEN] + data[self.CLOSE] + (data[self.LOW] + data[self.HIGH])/2)/3)/data[self.CLOSE]
        data[self.CLOSE + '_relative'] = data[self.CLOSE].pct_change()
        data = self.get_lagged_values(data)
        y, X = self.clean_from_unneeded_data(data, sample_size, use_first_sample)
        return y, X

if __name__ == '__main__':
    prep = DataPreprocessingForLinear(hour_dummies=False)
    prep.get_full_ticker_data(Tickers.USD000UTSTOM)