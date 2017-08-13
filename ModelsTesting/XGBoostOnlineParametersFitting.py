from datetime import timedelta

from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import TimeSeriesSplit
from xgboost.sklearn import XGBRegressor

from TestSamplesGeneration.DataPreprocessingForNonlinear import DataPreprocessingForNonlinear
from TestSamplesGeneration.Utils import Tickers


class XGBoostFastParametersFit(object):
    def __init__(self):
        self.data_class = DataPreprocessingForNonlinear(backward_lags=15, forward_lag=0)
        self.windows_in_days = 10

    def run_xgb_hyperparams_selection(self, ticker = Tickers.USD000UTSTOM):
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
            'max_depth': range(1, 9, 2),
            'min_child_weight': range(1, 10, 2)
        }
        est = XGBRegressor(learning_rate=0.1, n_estimators=100, max_depth=5, min_child_weight=1, gamma=0,
                           subsample=1.0, colsample_bytree=1.0, objective='reg:linear',
                           nthread=-1, colsample_bylevel=1.0, seed=27)
        gsearch1 = GridSearchCV(estimator=est, param_grid=param_test1,
                                scoring='neg_mean_squared_error', n_jobs=1, cv=TimeSeriesSplit(3))
        gsearch1.fit(X, y)
        params_set['max_depth'] = gsearch1.best_params_['max_depth']
        params_set['min_child_weight'] = gsearch1.best_params_['min_child_weight']
        param_test2 = {
            'colsample_bylevel': [0.2, 0.5, 0.7, 1.0],
            'colsample_bytree': [0.2, 0.5, 0.7, 1.0]
        }
        est = XGBRegressor(learning_rate=0.1, n_estimators=100, max_depth=gsearch1.best_params_['max_depth'],
                           min_child_weight=gsearch1.best_params_['min_child_weight'], gamma=0,
                           subsample=1.0, colsample_bytree=1.0, objective='reg:linear',
                           nthread=-1, colsample_bylevel=1.0, seed=27)
        gsearch2 = GridSearchCV(estimator=est, param_grid=param_test2,
                                scoring='neg_mean_squared_error', n_jobs=1, cv=TimeSeriesSplit(3))
        gsearch2.fit(X, y)
        params_set['colsample_bylevel'] = gsearch2.best_params_['colsample_bylevel']
        params_set['colsample_bytree'] = gsearch2.best_params_['colsample_bytree']
        params_set['time'] = starting_time
        return params_set


if __name__ == '__main__':
    fitter = XGBoostFastParametersFit()
    fitter.run_xgb_hyperparams_selection()