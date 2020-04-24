import alpaca_trade_api as tradeapi
import collections
import datetime
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import simulate
import os
import tensorflow.keras as keras
import unittest
import unittest.mock as mock
import utils
import yfinance as yf

Clock = collections.namedtuple('Clock', ['is_open'])
Asset = collections.namedtuple('Asset', ['symbol', 'tradable'])


class TradingSimulateTest(unittest.TestCase):

    def setUp(self):
        self.patch_open = mock.patch('builtins.open', mock.mock_open())
        self.patch_open.start()
        self.patch_isfile = mock.patch.object(os.path, 'isfile', return_value=False)
        self.patch_isfile.start()
        self.fake_model = mock.Mock()
        self.fake_model.predict.side_effect = lambda x: [50] * len(x)
        self.patch_keras = mock.patch.object(keras.models, 'load_model', return_value=self.fake_model)
        self.patch_keras.start()
        self.patch_mkdirs = mock.patch.object(os, 'makedirs')
        self.patch_mkdirs.start()
        self.patch_savefig = mock.patch.object(plt, 'savefig')
        self.patch_savefig.start()
        np.random.seed(0)
        fake_history_data = pd.DataFrame({'Close': np.append(np.random.random(990) * 10 + 100,
                                                             np.random.random(10) * 10 + 90),
                                          'High': np.random.random(1000) * 10 + 110,
                                          'Low': np.random.random(1000) * 10 + 90,
                                          'Volume': [10000] * 1000},
                                         index=[datetime.datetime.today().date() - pd.tseries.offsets.BDay(offset)
                                                for offset in range(999, -1, -1)])
        self.patch_history = mock.patch.object(yf.Ticker, 'history', return_value=fake_history_data)
        self.patch_history.start()
        self.alpaca = mock.create_autospec(tradeapi.REST)
        self.alpaca.list_assets.return_value = [Asset(symbol, True)
                                                for symbol in [utils.REFERENCE_SYMBOL,
                                                               'SYMA', 'SYMB', 'QQQ']]
        self.alpaca.get_clock.return_value = Clock(False)
        self.trading = simulate.TradingSimulate(
            self.alpaca,
            start_date=(datetime.datetime.today().date() - pd.tseries.offsets.BDay(30)).strftime('%F'))

    def tearDown(self):
        self.patch_open.stop()
        self.patch_isfile.stop()
        self.patch_keras.stop()
        self.patch_mkdirs.stop()
        self.patch_history.stop()
        self.patch_savefig.stop()

    def test_run_default(self):
        self.trading.run()
        self.assertGreater(self.trading.gain_transactions + self.trading.loss_transactions, 0)

    def test_run_with_data_file(self):
        data_dict = {feature: np.random.random(30) for feature in utils.ML_FEATURES}
        data_dict['Date'] = ['2020-01-01'] * 10 + ['2020-01-02'] * 10 + ['2020-01-03'] * 10
        data_dict['Gain'] = np.random.random(30) - 0.5
        data_dict['Symbol'] = ['SYMX'] * 30
        fake_data = pd.DataFrame(data_dict)
        with mock.patch.object(pd, 'read_csv', return_value=fake_data):
            trading = simulate.TradingSimulate(
                self.alpaca, data_file='fake_data_file')
            trading.run()
        self.assertGreater(trading.gain_transactions + trading.loss_transactions, 0)


if __name__ == '__main__':
    unittest.main()
