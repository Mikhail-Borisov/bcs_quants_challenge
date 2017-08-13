from datetime import timedelta

from sklearn.linear_model import RidgeCV
from sklearn.model_selection import GridSearchCV
from sklearn import preprocessing
from TestSamplesGeneration.DataPreprocessingForNonlinear import DataPreprocessingForNonlinear
from TestSamplesGeneration.Utils import Tickers


class SGDFastParametersFit(object):
    def __init__(self):
        self.data_class = DataPreprocessingForNonlinear(backward_lags=5, forward_lag=0)
        self.windows_in_days = 20

    def run_sgd_hyperparams_selection(self, ticker = Tickers.SBER):
        y_train, X_train = self.data_class.get_full_ticker_data(ticker)
        starting_time = X_train.index.min()
        sample_length = (X_train.index.max() - X_train.index.min()).days
        best_params = []
        for period in range(1, sample_length/10):
            print(starting_time)
            X = X_train.loc[(X_train.index>=starting_time) &
                            (X_train.index<=starting_time + timedelta(days=self.windows_in_days))]
            y = y_train.loc[(y_train.index>=starting_time) &
                            (y_train.index <= starting_time + timedelta(days=self.windows_in_days))]
            best_param = self.get_best_params_for_sample(X, y, starting_time)
            print(best_param)
            best_params.append(best_param)
            starting_time += timedelta(days=self.windows_in_days)

    def get_best_params_for_sample(self, X, y, starting_time):
        params_set = {}
        est = RidgeCV(fit_intercept=False, normalize=True, cv = 3,
                      alphas=[0.01, .1, .5, .9, 5.0, 10.0, 15.0, 20.0, 30.0, 50.0, 70.0, 100.0])
        est.fit(X, y)
        params_set['alpha'] = est.alpha_
        print(est.alpha_)
        params_set['time'] = starting_time
        return params_set


if __name__ == '__main__':
    fitter = SGDFastParametersFit()
    fitter.run_sgd_hyperparams_selection()