import pandas as pd
import numpy as np
import os
from TestSamplesGeneration.Utils import Frequency, Tickers


class DataPreprocessing(object):
    WEEKDAY = 'weekday'
    HOUR_FLOAT = 'hour_float'
    TIME = 'time'
    OPEN = 'open'
    HIGH = 'high'
    LOW = 'low'
    CLOSE = 'close'
    VOLUME = 'volume'
    COUNT = 'count'
    AVG_VOLUME = 'avg_deal_volume'
    TARGET = 'target'

    def __init__(self, frequency = Frequency.m1):
        self.default_data_path = os.path.dirname(__file__) + '/../data/'
        self.frequency = frequency
        # This is to fix the fact that 1 minute lag at 10 am is 18:39 of prev day
        self.default_market_open_hour = 10
        self.default_market_close_hour = 18.65
        self.clean_market_open_close_data = True
        ########################
        self.columns_for_lags = [self.OPEN + '_relative', self.HIGH + '_relative', self.LOW + '_relative',
                                 self.COUNT, self.VOLUME, self.CLOSE + '_relative', self.AVG_VOLUME]
        # lags for time series, backward for X, forward for y
        self.backward_lags = 5
        self.target_forward_freq_shift = 5

    def get_full_ticker_data(self, ticker_name = Tickers.USD000UTSTOM):
        raw_data = self.get_raw_ticker_data(ticker_name)
        data = self.adjust_prices_relative_to_another_price(raw_data)
        data = self.get_weekly_and_daytime_parameters(data)
        data[self.CLOSE + '_relative'] = data[self.CLOSE].pct_change()
        data = self.get_lagged_values(data)
        y, X = self.clean_from_unneeded_data(data)
        return y, X

    def get_raw_ticker_data(self, ticker_name):
        folder_path = self.default_data_path + self.frequency.value + '/' + ticker_name.value + '/'
        final = pd.DataFrame()
        final[self.CLOSE] = np.load(folder_path + self.CLOSE + '.npy')
        final[self.OPEN] = np.load(folder_path + self.OPEN + '.npy')
        final[self.HIGH] = np.load(folder_path + self.HIGH + '.npy')
        final[self.LOW] = np.load(folder_path + self.LOW + '.npy')
        final[self.TIME] = np.load(folder_path + self.TIME + '.npy')
        final[self.COUNT] = np.load(folder_path + self.COUNT + '.npy')
        final[self.VOLUME] = np.load(folder_path + self.VOLUME + '.npy')
        final[self.AVG_VOLUME] = final[self.VOLUME]/final[self.COUNT]
        return final.set_index(self.TIME)

    @staticmethod
    def adjust_prices_relative_to_another_price(data, to_adjust_prices_list = ['open', 'high', 'low'],
                                                relative_vector = 'close'):
        for column in to_adjust_prices_list:
            data[column + '_relative'] = data[column]/data[relative_vector]
        return data

    def get_weekly_and_daytime_parameters(self, data):
        # TODO Add possible russian holidays
        # TODO Add important news timers
        data[self.HOUR_FLOAT] = data.index.hour + data.index.minute/60.0
        data[self.WEEKDAY] = data.index.weekday
        return data

    def get_lagged_values(self, data, clean_for_market_open = True):
        data[self.TARGET] = data[self.CLOSE + '_relative'].shift(-self.target_forward_freq_shift)
        for i in range(1, self.backward_lags + 1):
            for column in self.columns_for_lags:
                data[column + '_lag' + str(i)] = data[column].shift(i)
        if clean_for_market_open and self.HOUR_FLOAT in data.columns:
            data = self.clean_lags_for_market_open(data, self.backward_lags, self.columns_for_lags)
        return data

    def clean_lags_for_market_open(self, data, lags, columns_for_lags):
        hour_to_start = self.default_market_open_hour + self.frequency.get_frequency_part_in_hours() * lags
        hour_to_end = self.default_market_close_hour - self.frequency.get_frequency_part_in_hours() \
                                                       * self.target_forward_freq_shift
        if self.clean_market_open_close_data:
            data = data.loc[(data[self.HOUR_FLOAT]>hour_to_start) & (data[self.HOUR_FLOAT]<hour_to_end)].dropna()
        else:
            for i in range(1, lags + 1):
                for column in columns_for_lags:
                    data.loc[data[self.HOUR_FLOAT]<=hour_to_start, column + '_lag' + str(i)] = np.nan
                    data.loc[data[self.HOUR_FLOAT]>=hour_to_end, column + '_lag' + str(i)] = np.nan
        return data

    def clean_from_unneeded_data(self, data):
        del data[self.OPEN]
        del data[self.CLOSE]
        del data[self.HIGH]
        del data[self.LOW]
        del data[self.VOLUME]
        del data[self.AVG_VOLUME]
        del data[self.COUNT]
        del data[self.CLOSE + '_relative']
        del data[self.OPEN + '_relative']
        del data[self.LOW + '_relative']
        del data[self.HIGH + '_relative']
        target = data[self.TARGET]
        return target, data


if __name__ == '__main__':
    prep = DataPreprocessing()
    prep.get_full_ticker_data(Tickers.ALRS)