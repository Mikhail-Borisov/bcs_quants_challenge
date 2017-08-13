import pandas as pd
import numpy as np
import os
from TestSamplesGeneration.Utils import Frequency, Tickers
import talib
from talib import MA_Type


class DataPreprocessingForNonlinear(object):
    VWAP = 'vwap_aprox'
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
    HOUR = 'hour'
    BB_UPPER = 'bb_upper'
    BB_LOWER = 'bb_lower'
    BB_MIDLE = 'bb_midle'
    ADDITIONAL_PRICES = [Tickers.USD000UTSTOM, Tickers.MICEXINDEXCF, Tickers.SIX, Tickers.BRX, Tickers.GZX]

    def __init__(self, frequency = Frequency.m1, backward_lags = 5, forward_lag = 1, hour_dummies = True,
                 threshold = 0.0002):
        self.default_data_path = os.path.dirname(__file__) + '/../data/'
        self.frequency = frequency
        # This is to fix the fact that 1 minute lag at 10 am is 18:39 of prev day
        self.default_market_open_hour = 10
        self.default_market_close_hour = 18.65
        self.clean_market_open_close_data = True
        ########################
        self.columns_for_lags = [self.OPEN + '_relative', self.HIGH + '_relative', self.LOW + '_relative',
                                 self.COUNT, self.VOLUME, self.CLOSE + '_relative', self.VWAP]
        # lags for time series, backward for X, forward for y
        self.backward_lags = backward_lags
        self.target_forward_freq_shift = forward_lag
        # get dummies for linear model
        self.hour_dummies = hour_dummies
        self.actual_additional_tickers = []
        self.threshold = threshold

    def get_full_ticker_data(self, ticker_name = Tickers.USD000UTSTOM, sample_size = 0.5, use_first_sample = True,
                             is_classification=True):
        self.actual_additional_tickers = []
        raw_data = self.get_raw_ticker_data(ticker_name)
        data = self.add_bbands(raw_data)
        data = self.adjust_prices_relative_to_another_price(data)
        data = self.get_weekly_and_daytime_parameters(data)
        data[self.VWAP] = ((data[self.OPEN] + data[self.CLOSE] +
                            (data[self.LOW] + data[self.HIGH]) / 2) / 3) / data[self.CLOSE]
        if not is_classification:
            data[self.CLOSE + '_relative'] = data[self.CLOSE].pct_change()
        else:
            data[self.CLOSE + '_relative'] = data[self.CLOSE].pct_change()
            data[self.CLOSE + '_relative'] = np.vectorize(self.get_three_class_for_target)(data[self.CLOSE + '_relative'])
        data = self.get_lagged_values(data)
        y, X = self.clean_from_unneeded_data(data, sample_size, use_first_sample)
        return y, X

    def get_three_class_for_target(self, price_return):
        if np.isnan(price_return):
            return np.nan
        if price_return <= -self.threshold:
            return -1
        elif price_return >= self.threshold:
            return 1
        else:
            return 0

    def get_raw_ticker_data(self, ticker_name):
        unified_time_index = pd.read_csv(os.path.dirname(__file__) + '/final_index.csv', index_col=1, parse_dates=True,
                                         header=None).index
        folder_path = self.default_data_path + self.frequency.value + '/' + ticker_name.value + '/'
        final = self.get_columns_ticker_data(folder_path, unified_time_index)
        for ticker in self.ADDITIONAL_PRICES:
            if ticker_name != ticker:
                local_data = self.get_reduced_columns_ticker_data(ticker, unified_time_index)
                final.loc[:, ticker.name + self.CLOSE] = local_data[self.CLOSE]
                final.loc[:, ticker.name + self.VOLUME] = np.log(local_data[self.VOLUME] + 1)
                self.actual_additional_tickers.append(ticker.name)
        return final

    def get_reduced_columns_ticker_data(self, ticker, unified_time_index):
        path = self.default_data_path + self.frequency.value + '/' + ticker.value + '/'
        local_data = pd.DataFrame()
        local_data[self.CLOSE] = np.load(path + self.CLOSE + '.npy')
        local_data[self.VOLUME] = np.load(path + self.VOLUME + '.npy')
        local_data[self.TIME] = np.load(path + self.TIME + '.npy')
        local_data.set_index(self.TIME, inplace=True)
        local_data = local_data.loc[unified_time_index].fillna(method='bfill').fillna(method='ffill')
        local_data[self.CLOSE] = local_data[self.CLOSE].pct_change()
        return local_data.fillna(0.0)

    def get_columns_ticker_data(self, folder_path, time_index):
        final = pd.DataFrame()
        final[self.CLOSE] = np.load(folder_path + self.CLOSE + '.npy')
        final[self.OPEN] = np.load(folder_path + self.OPEN + '.npy')
        final[self.HIGH] = np.load(folder_path + self.HIGH + '.npy')
        final[self.LOW] = np.load(folder_path + self.LOW + '.npy')
        final[self.TIME] = np.load(folder_path + self.TIME + '.npy')
        final[self.COUNT] = np.log(np.load(folder_path + self.COUNT + '.npy') + 1)
        final[self.VOLUME] = np.log(np.load(folder_path + self.VOLUME + '.npy') + 1)
        final.set_index(self.TIME, inplace=True)
        final = final.loc[time_index].fillna(method='bfill').fillna(method='ffill')
        return final

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
        if self.hour_dummies:
            data[self.HOUR] = data.index.hour
            data = pd.get_dummies(data, columns=[self.WEEKDAY, self.HOUR])
        else:
            data = pd.get_dummies(data, columns=[self.WEEKDAY])
        return data

    def get_lagged_values(self, data, clean_for_market_open = True):
        data[self.TARGET] = data[self.CLOSE + '_relative'].shift(-self.target_forward_freq_shift)
        for i in range(0, self.backward_lags):
            for column in self.columns_for_lags:
                data[column + '_lag' + str(i+1)] = data[column].shift(i)
            for column in self.actual_additional_tickers:
                data[column + self.CLOSE + '_lag' + str(i+1)] = data[column + self.CLOSE].shift(i)
                data[column + self.VOLUME + '_lag' + str(i+1)] = data[column + self.VOLUME].shift(i)
        data[self.BB_LOWER + '_lag1'] = data[self.BB_LOWER].shift(0)
        data[self.BB_UPPER + '_lag1'] = data[self.BB_UPPER].shift(0)
        data[self.BB_MIDLE + '_lag1'] = data[self.BB_MIDLE].shift(0)
        if clean_for_market_open and self.HOUR_FLOAT in data.columns:
            data = self.clean_lags_for_market_open(data, self.backward_lags, self.columns_for_lags)
        else:
            data.dropna(inplace=True)
        return data

    def clean_lags_for_market_open(self, data, lags, columns_for_lags):
        hour_to_start = self.default_market_open_hour + self.frequency.get_frequency_part_in_hours() * lags
        hour_to_end = self.default_market_close_hour - self.frequency.get_frequency_part_in_hours() \
                                                       * self.target_forward_freq_shift
        data.loc[data[self.HOUR_FLOAT] >= hour_to_end] = np.nan
        if self.clean_market_open_close_data:
            data = data.loc[(data[self.HOUR_FLOAT]>hour_to_start) & (data[self.HOUR_FLOAT]<hour_to_end)].dropna()
        else:
            for i in range(1, lags + 1):
                for column in columns_for_lags:
                    lag_hour_start = self.default_market_open_hour + self.frequency.get_frequency_part_in_hours() * i
                    data.loc[data[self.HOUR_FLOAT]<lag_hour_start, column + '_lag' + str(i)] = np.nan
        return data

    def clean_from_unneeded_data(self, data, sample_size, use_first_sample):
        data = data.loc[data[self.TARGET]>-2].fillna(-999.0)
        daterange = (data.index.max() - data.index.min()) * sample_size
        if use_first_sample:
            data = data.loc[:data.index.min() + daterange]
        else:
            data = data.loc[data.index.min() + daterange:]
        del data[self.OPEN]
        del data[self.CLOSE]
        del data[self.HIGH]
        del data[self.LOW]
        del data[self.VOLUME]
        del data[self.COUNT]
        del data[self.CLOSE + '_relative']
        del data[self.HIGH + '_relative']
        del data[self.LOW + '_relative']
        del data[self.OPEN + '_relative']
        del data[self.VWAP]
        del data[self.BB_MIDLE]
        del data[self.BB_UPPER]
        del data[self.BB_LOWER]
        for ticker in self.actual_additional_tickers:
            del data[ticker + self.CLOSE]
            del data[ticker + self.VOLUME]
        if self.hour_dummies:
            del data[self.HOUR_FLOAT]
        target = data[self.TARGET]
        del data[self.TARGET]
        return target, data

    @staticmethod
    def add_sma(data, column='close', timeperiod=12):
        """ Add simple moving average to your dataframe"""
        # TODO Update description
        if (column in data.columns):
            chosen_column = data[column].values
            new_column = pd.Series(talib.SMA(chosen_column, timeperiod=timeperiod),
                                   name=(column+'_sma_'+str(timeperiod)))
        else:
            print('Column name does not exist in the dataframe')
            return
        return pd.concat([data, new_column], axis=1)

    def add_bbands(self, data, column='close', timeperiod=5, stdup = 2, stddn = 2, matype=MA_Type.T3):
        """" Add upper, middle and lower Bollinger Bands to your dataframe"""
        # TODO Update description
        if (column in data.columns):
            chosen_column = data[column].values
            upper, middle, lower = talib.BBANDS(chosen_column, timeperiod=timeperiod, nbdevup=stdup,
                                                nbdevdn=stddn, matype=matype)
            new_columns = pd.DataFrame(index=data.index)
            new_columns[self.BB_UPPER] = upper/chosen_column
            new_columns[self.BB_LOWER] = lower/chosen_column
            new_columns[self.BB_MIDLE] = middle/chosen_column
        else:
            print('Column name does not exist in the dataframe')
            return
        return pd.concat([data, new_columns], axis=1)

    def add_prev_day_info(self, data):
        pass
    
if __name__ == '__main__':
    prep = DataPreprocessingForNonlinear()
    prep.get_full_ticker_data(Tickers.ALRS)