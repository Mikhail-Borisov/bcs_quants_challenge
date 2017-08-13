from datetime import timedelta
import numpy as np
from sklearn.linear_model import SGDRegressor
from sklearn.model_selection import GridSearchCV
from sklearn import preprocessing
from TestSamplesGeneration.DataPreprocessingForNonlinear import DataPreprocessingForNonlinear
from TestSamplesGeneration.Utils import Tickers
from sklearn.metrics import mean_squared_error, r2_score
import pandas as pd


class SGDOnlineBatchParametersFit(object):
    def __init__(self):
        self.data_class = DataPreprocessingForNonlinear(backward_lags=4, forward_lag=0)
        self.backward_window_in_days = 50
        self.forward_window_in_days = 1
        self.weights = 'none'
        self.alpha = 0.5

    def run_sgd_testing(self, ticker = Tickers.USD000UTSTOM):
        y_train, X_train = self.data_class.get_full_ticker_data(ticker, sample_size=0.2)
        starting_time = X_train.index.min()
        sample_length = (X_train.index.max() - X_train.index.min()).days
        results = []
        for period in range(1, sample_length):
            sample_end_time = starting_time + timedelta(days=self.backward_window_in_days) - timedelta(hours=1)
            # train model with weights
            X_train_model = X_train.loc[(X_train.index>=starting_time) &
                                        (X_train.index <= sample_end_time)]
            y_train_model = y_train.loc[(y_train.index>=starting_time) &
                                        (y_train.index <= sample_end_time)]
            model, in_sample_r_square, scaler = self.get_fitted_model(X_train_model, y_train_model)
            # 1 day ahead forecast
            test_set_start_time = X_train.loc[(X_train.index>sample_end_time)].index.min()
            X_test_pandas = X_train.loc[(X_train.index >= test_set_start_time) & (
                X_train.index <= test_set_start_time + timedelta(days=self.forward_window_in_days) - timedelta(hours=1))]
            if len(X_test_pandas) == 0:
                continue
            X_test = scaler.transform(X_test_pandas)
            y_test = y_train.loc[(y_train.index>=test_set_start_time) &
                                 (y_train.index <= test_set_start_time
                                  + timedelta(days=self.forward_window_in_days) - timedelta(hours=1))]
            test_r_square = r2_score(y_test, model.predict(X_test))
            half_test_r_square = r2_score(y_test.iloc[:len(y_test)/2], model.predict(X_test[:len(X_test)/2]))
            quater_test_r_square = r2_score(y_test.iloc[:len(y_test)/4], model.predict(X_test[:len(X_test)/4]))
            results.append([starting_time, round(in_sample_r_square,4), round(test_r_square, 4),
                            round(half_test_r_square, 4), round(quater_test_r_square, 4)])
            # print([starting_time, round(in_sample_r_square,4), round(test_r_square, 4),
            #        round(half_test_r_square, 4), round(quater_test_r_square, 4)])
            starting_time += timedelta(days=1)
        final_result = self.get_final_result(results)

    def get_fitted_model(self, X, y):
        est = SGDRegressor(warm_start=True, n_iter=1000, loss='squared_loss', alpha=self.alpha, fit_intercept=False,
                     l1_ratio=0.0, learning_rate='invscaling', shuffle=False, eta0=0.01)
        scaler = preprocessing.StandardScaler()
        X_train_scaled = scaler.fit_transform(X)
        if self.weights == 'linear':
            weights = np.linspace(0.0, 1.0, num=len(X_train_scaled))
        elif self.weights == 'exp':
            weights = np.logspace(1.0, 3.0, num=len(X_train_scaled))
            weights = weights/max(weights)
        else:
            weights = np.linspace(1.0, 1.0, num=len(X_train_scaled))
        est.fit(X_train_scaled, y, sample_weight=weights)
        r_square = r2_score(y, est.predict(X_train_scaled), sample_weight=weights)
        return est, r_square, scaler

    def get_final_result(self, results):
        final = pd.DataFrame(results, columns=['time', 'in_sample', 'full_test', 'test/2', 'test/4'])
        print(self.weights, self.backward_window_in_days, self.forward_window_in_days,
              final['full_test'].mean(), final['test/2'].mean(), final['test/4'].mean())

if __name__ == '__main__':
    fitter = SGDOnlineBatchParametersFit()
    for ticker in Tickers:
        print(ticker)
        fitter.run_sgd_testing(ticker)