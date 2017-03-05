from sklearn.linear_model import ElasticNetCV
from sklearn.model_selection import TimeSeriesSplit
import pandas as pd
from TestSamplesGeneration.DataPreprocessing import DataPreprocessing
from TestSamplesGeneration.Utils import Frequency, Tickers


class L1L2Checker(object):
    def __init__(self):
        self.for_cross_val_sample = 0.5
        self.forward_lag = 5
        self.l1_ratios = [0.0, 0.001, 0.005, .01, .03, .05, .07, .1, .5, .9, .95, .99]

    def get_cross_assest_l1l2_report(self, frequency = Frequency.m1, lags = 15):
        print(frequency, lags, self.forward_lag)
        prep_class = DataPreprocessing(frequency= frequency, backward_lags=lags, forward_lag=self.forward_lag)
        reg_params = {}
        for ticker in Tickers:
            y, X = prep_class.get_full_ticker_data(ticker, sample_size=self.for_cross_val_sample, use_first_sample = True)
            model = ElasticNetCV(l1_ratio= self.l1_ratios, cv=TimeSeriesSplit(10), n_jobs=-1, normalize=True)
            model.fit(X, y)
            reg_params[ticker] = [model.alpha_, model.l1_ratio_]
        final = pd.DataFrame(reg_params).T
        final.columns = ['alpha', 'l1_ratio']
        print(final)

if __name__ == '__main__':
    checker = L1L2Checker()
    checker.get_cross_assest_l1l2_report(Frequency.m1, lags=25)