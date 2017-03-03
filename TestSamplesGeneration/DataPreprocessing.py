import pandas as pd
import numpy as np

from TestSamplesGeneration.Utils import Frequency


class DataPreprocessing(object):
    def __init__(self, frequency = Frequency.m1):
        self.default_data_path = '../data/'
        self.frequency = str(frequency)

    def raw_ticker_data(self, ticker_name):
        folder_path = self.default_data_path + self.frequency + '/' + str(ticker_name)
        final = pd.DataFrame()
        final['close'] = np.load(folder_path + 'close.npy')
        final['open'] = np.load(folder_path + 'open.npy')
        final['high'] = np.load(folder_path + 'high.npy')
        final['low'] = np.load(folder_path + 'low.npy')
        final['time'] = np.load(folder_path + 'time.npy')
        final['count'] = np.load(folder_path + 'count.npy')
        final['volume'] = np.load(folder_path + 'volume.npy')
        return final
