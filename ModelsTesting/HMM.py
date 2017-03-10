from datetime import datetime

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_squared_error, r2_score
import sys

from statsmodels.tsa.regime_switching.markov_regression import MarkovRegression

from TestSamplesGeneration.DataPreprocessingForNonlinear import DataPreprocessingForNonlinear
from TestSamplesGeneration.Utils import Tickers

print(datetime.now())
prep_class = DataPreprocessingForNonlinear(backward_lags=1, forward_lag=0)
y_train, X_train = prep_class.get_full_ticker_data(Tickers.USD000UTSTOM, sample_size=0.05, use_first_sample=True)
y_test, X_test = prep_class.get_full_ticker_data(Tickers.USD000UTSTOM, sample_size=0.05, use_first_sample=False)
model = MarkovRegression(y_train, 2, trend='nt', exog = X_train)
result = model.fit()
print(result)
print(datetime.now())