import alpaca_trade_api as tradeapi
import alpaca_trade_api.polygon as polygonapi
import argparse
import collections
import datetime
import itertools
import numpy as np
import pandas as pd
import realtime
import requests
import os
import time
import unittest
import unittest.mock as mock
import utils
import yfinance as yf
from parameterized import parameterized

Clock = collections.namedtuple('Clock', ['is_open', 'next_close'])
Asset = collections.namedtuple('Asset', ['symbol', 'tradable', 'marginable',
                                         'shortable', 'easy_to_borrow'])
Account = collections.namedtuple('Account', ['equity', 'cash'])
LastTrade = collections.namedtuple('LastTrade', ['price'])
Position = collections.namedtuple('Position', ['symbol', 'qty', 'current_price',
                                               'market_value', 'cost_basis'])


class TradingRealTimeTest(unittest.TestCase):

    def setUp(self):
        self.patch_open = mock.patch('builtins.open', mock.mock_open())
        self.patch_open.start()
        self.patch_isfile = mock.patch.object(os.path, 'isfile', return_value=False)
        self.patch_isfile.start()
        self.patch_mkdirs = mock.patch.object(os, 'makedirs')
        self.patch_mkdirs.start()
        self.patch_to_csv = mock.patch.object(pd.DataFrame, 'to_csv')
        self.patch_to_csv.start()
        self.patch_sleep = mock.patch.object(time, 'sleep')
        self.mock_sleep = self.patch_sleep.start()
        fake_history_data = pd.DataFrame({'Close': np.array(([100] * 9 + [90]) * 98 + ([100] * 9 + [70]) * 2),
                                          'High': np.random.random(1000) * 10 + 110,
                                          'Low': np.random.random(1000) * 10 + 90,
                                          'Volume': [1E6] * 1000},
                                         index=[datetime.datetime.today().date() - pd.tseries.offsets.BDay(offset)
                                                for offset in range(999, -1, -1)])
        self.patch_history = mock.patch.object(yf.Ticker, 'history', return_value=fake_history_data)
        self.patch_history.start()
        self.alpaca = mock.create_autospec(tradeapi.REST)
        self.alpaca.get_account.return_value = Account(2000, 2000)
        self.alpaca.list_assets.return_value = [Asset(symbol, True, True, True, True)
                                                for symbol in [utils.REFERENCE_SYMBOL,
                                                               'SYMA', 'SYMB', 'SYMC']]
        fake_next_close = mock.Mock()
        fake_next_close.timestamp.return_value = 1000
        self.alpaca.get_clock.return_value = Clock(True, fake_next_close)
        self.alpaca.list_orders.return_value = []
        self.polygon = mock.create_autospec(polygonapi.REST)
        self.polygon.last_trade.return_value = LastTrade(69)
        self.trading = realtime.TradingRealTime(self.alpaca, self.polygon)

    def tearDown(self):
        self.patch_open.stop()
        self.patch_isfile.stop()
        self.patch_mkdirs.stop()
        self.patch_history.stop()
        self.patch_sleep.stop()
        self.patch_to_csv.stop()

    def test_run_fail(self):
        self.polygon.last_trade.side_effect = requests.exceptions.HTTPError('Test error')
        with mock.patch.object(time, 'time', side_effect=itertools.repeat(800)), \
                mock.patch.object(realtime.TradingRealTime, 'update_all_prices') as mock_update_all_prices, \
                mock.patch.object(realtime.TradingRealTime, 'trade_clock_watcher') as trade_clock_watcher, \
                self.assertRaises(requests.exceptions.HTTPError):
            self.trading.run()
        mock_update_all_prices.assert_called_once()
        trade_clock_watcher.assert_called_once()

if __name__ == '__main__':
    unittest.main()
