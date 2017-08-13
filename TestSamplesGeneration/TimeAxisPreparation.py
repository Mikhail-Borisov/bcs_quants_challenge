import numpy as np
import pandas as pd

from TestSamplesGeneration.DataPreprocessingForNonlinear import DataPreprocessingForNonlinear
from TestSamplesGeneration.Utils import Frequency, Tickers

prep = DataPreprocessingForNonlinear()
folder_path = prep.default_data_path + Frequency.m1.value + '/'
full_time_index = []
for ticker in Tickers:
    single_data = pd.DataFrame()
    single_data[prep.TIME] = np.load(folder_path + ticker.value + '/' + prep.TIME + '.npy')
    full_time_index.extend(single_data[prep.TIME].tolist())
final = pd.DataFrame()
final['Time'] = full_time_index
final_time_index = sorted(final['Time'].unique())
pd.Series(final_time_index).to_csv('final_index.csv', index=False)