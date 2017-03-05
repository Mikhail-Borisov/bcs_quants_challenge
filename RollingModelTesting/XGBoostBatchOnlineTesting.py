from datetime import timedelta
import numpy as np
from sklearn.linear_model import SGDRegressor
import pandas as pd
import xgboost as xgb
from sklearn import preprocessing
from TestSamplesGeneration.DataPreprocessing import DataPreprocessing
from TestSamplesGeneration.Utils import Tickers
from sklearn.metrics import mean_squared_error, r2_score

class XGBoostOnlineBatchParametersFit(object):
    def __init__(self):
        self.data_class = DataPreprocessing(backward_lags=30, forward_lag=0)
        self.backward_window_in_days = 100
        self.forward_window_in_days = 5
        self.weights = 'none'

    def run_xgboost_testing(self, ticker = Tickers.USD000UTSTOM):
        y_train, X_train = self.data_class.get_full_ticker_data(ticker, sample_size=0.3)
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
            model, in_sample_r_square = self.get_fitted_model(X_train_model, y_train_model)
            # 1 day ahead forecast
            test_set_start_time = X_train.loc[(X_train.index>sample_end_time)].index.min()
            X_test_pandas = X_train.loc[(X_train.index>=test_set_start_time) &
                            (X_train.index <= test_set_start_time +
                             timedelta(days=self.forward_window_in_days) - timedelta(hours=1))]
            if X_test_pandas.empty:
                continue
            y_test = y_train.loc[(y_train.index>=test_set_start_time) &
                            (y_train.index <= test_set_start_time +
                             timedelta(days=self.forward_window_in_days) - timedelta(hours=1))]
            half_test_r_square, quater_test_r_square, test_r_square = self.get_test_r_square_result(X_test_pandas,
                                                                                                    model, y_test)
            # print([starting_time, round(in_sample_r_square,4), round(test_r_square, 4),
            #                 round(half_test_r_square, 4), round(quater_test_r_square, 4)])
            results.append([starting_time, round(in_sample_r_square,4), round(test_r_square, 4),
                            round(half_test_r_square, 4), round(quater_test_r_square, 4)])
            starting_time += timedelta(days=1)
        final_result = self.get_final_result(results, ticker)

    def get_test_r_square_result(self, X_test_pandas, model, y_test):
        test_r_square = r2_score(y_test, model.predict(xgb.DMatrix(X_test_pandas)))
        half_test_r_square = r2_score(y_test.iloc[:len(y_test) / 2],
                                      model.predict(xgb.DMatrix(X_test_pandas[:len(X_test_pandas) / 2])))
        quater_test_r_square = r2_score(y_test.iloc[:len(y_test) / 4],
                                        model.predict(xgb.DMatrix(X_test_pandas[:len(X_test_pandas) / 4])))
        return half_test_r_square, quater_test_r_square, test_r_square

    def get_fitted_model(self, X, y):
        param = {'max_depth': 1, 'min_child_weight': 9, 'eta': 0.05,
                 'silent': 1, 'objective': 'reg:linear',
                 'subsample': 1.0, 'colsample_bytree': 0.5, 'colsample_bylevel': 1.0}
        if self.weights == 'linear':
            weights = np.linspace(0.0, 1.0, num=len(X))
        elif self.weights == 'exp':
            weights = np.logspace(1.0, 3.0, num=len(X))
            weights = weights/max(weights)
        else:
            weights = np.linspace(1.0, 1.0, num=len(X))
        dtrain = xgb.DMatrix(X, y, weight=weights)
        model = xgb.train(param, dtrain, num_boost_round=250)
        r_square = r2_score(y, model.predict(dtrain), sample_weight=weights)
        return model, r_square

    def get_final_result(self, results, ticker):
        final = pd.DataFrame(results, columns=['time', 'in_sample', 'full_test', 'test/2', 'test/4'])
        print(self.weights, self.backward_window_in_days, self.forward_window_in_days, ticker,
            final['full_test'].mean(), final['test/2'].mean(), final['test/4'].mean())


if __name__ == '__main__':
    fitter = XGBoostOnlineBatchParametersFit()
    for ticker in Tickers:
        fitter.run_xgboost_testing(ticker)