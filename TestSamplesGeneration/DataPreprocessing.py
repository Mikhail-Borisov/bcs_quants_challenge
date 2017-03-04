import pandas as pd
import numpy as np

from TestSamplesGeneration.Utils import Frequency


class DataPreprocessing(object):
    def __init__(self, frequency = Frequency.m1):
        self.default_data_path = '../data/'
        self.frequency = frequency

    def get_raw_ticker_data(self, ticker_name):
        folder_path = self.default_data_path + str(self.frequency) + '/' + str(ticker_name)
        final = pd.DataFrame()
        final['close'] = np.load(folder_path + 'close.npy')
        final['open'] = np.load(folder_path + 'open.npy')
        final['high'] = np.load(folder_path + 'high.npy')
        final['low'] = np.load(folder_path + 'low.npy')
        final['time'] = np.load(folder_path + 'time.npy')
        final['count'] = np.load(folder_path + 'count.npy')
        final['volume'] = np.load(folder_path + 'volume.npy')
        final['avg_deal_volume'] = final['volume']/final['count']
        return final

    @staticmethod
    def adjust_prices_relative_to_another_price(data, to_adjust_prices_list = ['open', 'high', 'low'],
                                                relative_vector = 'close'):
        for column in to_adjust_prices_list:
            data[column + '_relative'] = data[column]/data[relative_vector]
        return data

    @staticmethod
    def get_weekly_and_daytime_parameters(datetime_numpy):
        datetime_python = pd.to_datetime(datetime_numpy)
        return datetime_python.hour + datetime_python.minute / 60, datetime_python.weekday()

    @staticmethod
    def get_simple_lags(data, lags = 10, clean_for_market_open = True,
                        columns = ['open', 'high', 'low', 'count', 'volume', 'close', 'avg_deal_volume']):
        for i in range(1, lags + 1):
            for column in columns:
                data[column + '_lag' + str(i)] = data[column].shift(i)
        if clean_for_market_open and 'hour_float' in 
        return data
