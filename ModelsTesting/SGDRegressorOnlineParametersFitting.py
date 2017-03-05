from datetime import timedelta

from sklearn.linear_model import SGDRegressor
from sklearn.model_selection import GridSearchCV
from sklearn import preprocessing
from TestSamplesGeneration.DataPreprocessing import DataPreprocessing
from TestSamplesGeneration.Utils import Tickers


class SGDFastParametersFit(object):
    def __init__(self):
        self.data_class = DataPreprocessing(backward_lags=4, forward_lag=0)
        self.windows_in_days = 10

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
        param_test1 = {
            'alpha': [0.01, .1, .5, .9, 5.0, 10.0, 15.0, 20.0, 30.0, 50.0, 70.0, 100.0],
        }
        est = SGDRegressor(warm_start=True, n_iter=1000, loss='squared_loss', alpha=0.1,
                     l1_ratio=0.0, learning_rate='invscaling', shuffle=False, eta0=0.001)
        scaler = preprocessing.StandardScaler()
        X_train_scaled = scaler.fit_transform(X)
        gsearch1 = GridSearchCV(estimator=est, param_grid=param_test1,
                                scoring='neg_mean_squared_error', n_jobs=6, cv=3)
        gsearch1.fit(X_train_scaled, y)
        params_set['alpha'] = gsearch1.best_params_['alpha']
        print(gsearch1.best_params_['alpha'])
        params_set['time'] = starting_time
        return params_set


if __name__ == '__main__':
    fitter = SGDFastParametersFit()
    fitter.run_sgd_hyperparams_selection()