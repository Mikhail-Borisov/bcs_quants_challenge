from glob import glob

import pandas as pd
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
import seaborn as sns
sns.set(style="darkgrid", color_codes=True)
from TestSamplesGeneration.Utils import Tickers
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score


class ResultsCheck(object):
    def __init__(self):
        self.path_name = 'xgboost_result_'

    def load_all_tickers_report(self, result_filename):
        pdf = PdfPages(result_filename)
        for ticker in Tickers:
            ticker_filename = self.path_name + ticker.value + '.csv'
            try:
                fig = self.get_ticker_report(ticker_filename)
                pdf.savefig(fig)
            except:
                continue
        pdf.close()

    def get_ticker_report(self, ticker_filename):
        print(ticker_filename)
        data = self.get_returns(ticker_filename)
        for_thr = self.get_data_for_threshold(data)
        fig, ax = plt.subplots(4, 1, figsize=(10, 10))
        if for_thr.empty:
            return fig
        for_thr['mean'].plot(ax=ax[0], color='r', linestyle='--')
        ax[0].fill_between(for_thr.index, for_thr['lower'], for_thr['upper'], alpha=0.2)
        for_thr['sharpe'].plot(ax=ax[1], color='r', linestyle='--', marker='o')
        ax[1].set_title('annualized sharpe')
        for_thr['n'].plot(ax=ax[2], color='r', linestyle='-', marker='o')
        ax[2].set_title('number of deals')
        for_thr['total'].plot(ax=ax[3], color='r', linestyle='-', marker='o')
        ax[3].set_title('total return')
        fig.suptitle(ticker_filename)
        fig.tight_layout()
        return fig

    def get_data_for_threshold(self, data):
        for_thr = []
        for threshold in np.linspace(0.0, data['abs_predicted'].max(), 300):
            mean_with_costs = data.loc[data['abs_predicted'] >= threshold, 'net_return'].mean()
            return_std = data.loc[data['abs_predicted'] >= threshold, 'net_return'].std()
            total_return = data.loc[data['abs_predicted'] >= threshold, 'net_return'].sum()
            n_deals = data.loc[data['abs_predicted'] >= threshold, 'net_return'].count()
            daily = data.loc[data['abs_predicted'] >= threshold].groupby('day')['net_return'].mean().loc[
                data['day'].unique()].fillna(0.0)
            sharpe = daily.mean() * np.sqrt(240) / daily.std()
            for_thr.append([threshold, mean_with_costs, return_std, sharpe, total_return, n_deals])
        for_thr = pd.DataFrame(for_thr, columns=['threshold', 'mean', 'std', 'sharpe', 'total', 'n'])
        for_thr = for_thr.loc[(for_thr['n'] > 10) & (for_thr['threshold'] > 0.02 / 100)]
        for_thr['upper'] = for_thr['mean'] + for_thr['std']
        for_thr['lower'] = for_thr['mean'] - for_thr['std']
        for_thr.set_index('threshold', inplace=True)
        return for_thr

    def get_returns(self, ticker_filename):
        data = pd.read_csv(ticker_filename, index_col=0, parse_dates=True)
        print(ticker_filename, r2_score(data['y_test'], data['predicted']))
        data['predicted_true'] = np.sign(data['y_test'] * data['predicted'])
        data['return'] = data['y_test'].abs() * data['predicted_true']
        data['abs_predicted'] = np.abs(data['predicted'])
        data['cost'] = 0.03 / 100
        data['net_return'] = data['return'] - data['cost']
        data['day'] = data.index.date
        return data

if __name__ == '__main__':
    report = ResultsCheck()
    report.load_all_tickers_report('simple_multi_asset_125days_window_m1_linearModels.pdf')