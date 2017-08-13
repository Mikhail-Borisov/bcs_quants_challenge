from datetime import timedelta
import numpy as np
from sklearn.linear_model import SGDRegressor
import pandas as pd
import xgboost as xgb
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler

from TestSamplesGeneration.DataPreprocessingForNonlinear import DataPreprocessingForNonlinear
from TestSamplesGeneration.Utils import Tickers
from sklearn.metrics import mean_absolute_error, r2_score


class XGBoostOnlineBatchParametersFit(object):
    def __init__(self):
        self.data_class = DataPreprocessingForNonlinear(backward_lags=2, forward_lag=1, hour_dummies=True)
        self.backward_window_in_days = 150
        self.forward_window_in_days = 1
        self.weights = 'none'
        self.importance = pd.DataFrame()
        self.model = None

    def run_xgboost_testing(self, ticker = Tickers.USD000UTSTOM):
        y_train, X_train = self.data_class.get_full_ticker_data(ticker, sample_size=0.5)
        starting_time = X_train.index.min() - timedelta(hours=1)
        sample_length = (X_train.index.max() - X_train.index.min()).days
        results = pd.DataFrame()
        for period in range(1, sample_length):
            sample_end_time = starting_time + timedelta(days=self.backward_window_in_days) - timedelta(hours=1)
            # print(starting_time, sample_end_time)
            # train model with weights
            X_train_model = X_train.loc[(X_train.index>=starting_time) &
                                        (X_train.index <= sample_end_time)]
            y_train_model = y_train.loc[(y_train.index>=starting_time) &
                                        (y_train.index <= sample_end_time)]
            test_set_start_time = X_train.loc[(X_train.index>sample_end_time)].index.min()
            X_test_pandas = X_train.loc[(X_train.index>=test_set_start_time) &
                                        (X_train.index <= test_set_start_time +
                                         timedelta(days=self.forward_window_in_days) - timedelta(hours=1))]
            if X_test_pandas.empty:
                starting_time += timedelta(days=self.forward_window_in_days)
                continue
            model, in_sample_r_square, scaler = self.get_fitted_model(X_train_model, y_train_model)
            # 1 day ahead forecast
            y_test = y_train.loc[(y_train.index>=test_set_start_time) &
                                 (y_train.index <= test_set_start_time +
                                  timedelta(days=self.forward_window_in_days) - timedelta(hours=1))]
            X_test_pandas = scaler.transform(X_test_pandas)
            y_test_final = self.get_test_result(X_test_pandas, model, y_test)
            print(r2_score(y_test_final['y_test'], y_test_final['predicted']))
            results = pd.concat([results, y_test_final], copy=False)
            to_add_move_diff = (test_set_start_time - sample_end_time).days
            starting_time += timedelta(days=self.forward_window_in_days + to_add_move_diff)
        self.get_final_result(results, ticker)

    def get_test_result(self, X_test_pandas, model, y_test):
        final = pd.DataFrame()
        final['y_test'] = y_test
        final['predicted'] = model.predict(xgb.DMatrix(X_test_pandas))
        return final

    def get_fitted_model(self, X, y):
        if self.weights == 'linear':
            weights = np.linspace(0.5, 1.0, num=len(X))
            weights = weights / sum(weights)
        elif self.weights == 'exp':
            weights = np.logspace(1.0, 3.0, num=len(X))
            weights = weights/sum(weights)
        else:
            weights = np.linspace(1.0, 1.0, num=len(X))
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X)
        param = {'booster': 'gblinear', 'lambda': 0.1, 'alpha': 0.1, 'eta': 0.05, 'silent': 1, 'objective': 'reg:linear'}
        dtrain = xgb.DMatrix(X_train_scaled, y, weight=weights)
        if self.model is None:
            model = xgb.train(param, dtrain, num_boost_round=250)
        else:
            param.update({'process_type': 'update',
                          'updater': 'refresh',
                          'refresh_leaf': True})
            model = xgb.train(param, dtrain, num_boost_round=250, xgb_model=self.model)
        self.model = model
        r_square = r2_score(y, model.predict(dtrain), sample_weight=weights)
        return model, r_square, scaler

    def get_final_result(self, results, ticker):
        r2 = r2_score(results['y_test'], results['predicted'])
        rmse = mean_absolute_error(results['y_test'], results['predicted'])
        print(self.weights, self.backward_window_in_days, self.forward_window_in_days, ticker,
            r2, rmse)
        results.to_csv('xgboost_result_UPDATED_' + ticker.value + '_ADDITIONAL_ASSETS.csv')


if __name__ == '__main__':
    fitter = XGBoostOnlineBatchParametersFit()
    fitter.run_xgboost_testing()
    # for ticker in Tickers:
    #     fitter.run_xgboost_testing(ticker)