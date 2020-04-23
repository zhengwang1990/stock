import alpaca_trade_api as tradeapi
import alpaca_trade_api.polygon as polygonapi
import collections
import datetime
import itertools
import numpy as np
import pandas as pd
import realtime
import requests
import os
import tensorflow.keras as keras
import time
import unittest
import unittest.mock as mock
import utils
import yfinance as yf
from parameterized import parameterized

Clock = collections.namedtuple('Clock', ['is_open', 'next_close'])
Asset = collections.namedtuple('Asset', ['symbol', 'tradable'])
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
        self.fake_model = mock.Mock()
        self.fake_model.predict.side_effect = lambda x: [50] * len(x)
        self.patch_keras = mock.patch.object(keras.models, 'load_model', return_value=self.fake_model)
        self.patch_keras.start()
        self.patch_mkdirs = mock.patch.object(os, 'makedirs')
        self.patch_mkdirs.start()
        self.patch_sleep = mock.patch.object(time, 'sleep')
        self.mock_sleep = self.patch_sleep.start()
        self.patch_web_scraping = mock.patch.object(utils, 'web_scraping', return_value='50')
        self.patch_web_scraping.start()
        fake_history_data = pd.DataFrame({'Close': np.append(np.random.random(998) * 10 + 100, [90, 89]),
                                          'High': np.random.random(1000) * 10 + 110,
                                          'Low': np.random.random(1000) * 10 + 90,
                                          'Volume': [10000] * 1000},
                                         index=[datetime.datetime.today().date() - pd.tseries.offsets.BDay(offset)
                                                for offset in range(999, -1, -1)])
        self.patch_history = mock.patch.object(yf.Ticker, 'history', return_value=fake_history_data)
        self.patch_history.start()
        self.alpaca = mock.create_autospec(tradeapi.REST)
        self.alpaca.get_account.return_value = Account(2000, 2000)
        self.alpaca.list_assets.return_value = [Asset(symbol, True)
                                                for symbol in [utils.REFERENCE_SYMBOL,
                                                               'SYMA', 'SYMB', 'SYMC']]
        fake_next_close = mock.Mock()
        fake_next_close.timestamp.return_value = 1000
        self.alpaca.get_clock.return_value = Clock(True, fake_next_close)
        self.polygon = mock.create_autospec(polygonapi.REST)
        self.polygon.last_trade.return_value = LastTrade(88)
        self.trade = realtime.TradingRealTime(self.alpaca, self.polygon)

    def tearDown(self):
        self.patch_open.stop()
        self.patch_isfile.stop()
        self.patch_keras.stop()
        self.patch_mkdirs.stop()
        self.patch_history.stop()
        self.patch_sleep.stop()
        self.patch_web_scraping.stop()

    def test_trade_clock_watcher(self):
        with mock.patch.object(realtime.TradingRealTime, 'trade') as trade:
            with mock.patch.object(time, 'time', side_effect=itertools.count(900)):
                self.trade.trade_clock_watcher()
                trade.assert_called_once()
                self.assertEqual(self.mock_sleep.call_count, 11)

    def test_get_realtime_price_accumulate_error(self):
        self.polygon.last_trade.side_effect = requests.exceptions.RequestException('Test error')
        self.assertEqual(len(self.trade.errors), 0)
        self.trade.get_realtime_price('SYMA')
        self.assertEqual(len(self.trade.errors), 1)

    def test_update_trading_list(self):
        with mock.patch.object(time, 'time', side_effect=itertools.count(999)):
            self.trade.update_trading_list()
            self.assertEqual(len(self.trade.trading_list), 4)

    @parameterized.expand([('limit',), ('market',)])
    def test_buy(self, order_type):
        self.alpaca.list_orders.return_value = []
        with mock.patch.object(time, 'time', side_effect=itertools.count(999)):
            self.trade.update_trading_list()
        self.trade.buy(order_type)
        self.assertEqual(self.alpaca.submit_order.call_count, 4)

    @parameterized.expand([('limit',), ('market',)])
    def test_sell(self, order_type):
        self.alpaca.list_orders.return_value = []
        self.alpaca.list_positions.return_value = [Position('SYMA', '10', '10.0', '100.0', '99.0'),
                                                   Position('SYMB', '20', '20.0', '400.0', '555.5')]
        self.trade.sell(order_type)
        self.assertEqual(self.alpaca.submit_order.call_count, 2)


if __name__ == '__main__':
    unittest.main()
