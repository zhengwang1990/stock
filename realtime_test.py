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
        self.patch_to_csv = mock.patch.object(pd.DataFrame, 'to_csv')
        self.patch_to_csv.start()
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
        self.alpaca.list_orders.return_value = []
        self.polygon = mock.create_autospec(polygonapi.REST)
        self.polygon.last_trade.return_value = LastTrade(88)
        self.trading = realtime.TradingRealTime(self.alpaca, self.polygon)

    def tearDown(self):
        self.patch_open.stop()
        self.patch_isfile.stop()
        self.patch_keras.stop()
        self.patch_mkdirs.stop()
        self.patch_history.stop()
        self.patch_sleep.stop()
        self.patch_web_scraping.stop()
        self.patch_to_csv.stop()

    def test_trade_clock_watcher(self):
        with mock.patch.object(realtime.TradingRealTime, 'trade') as trade, \
                mock.patch.object(time, 'time', side_effect=itertools.count(900)):
            self.trading.trade_clock_watcher()
            trade.assert_called_once()
            self.assertEqual(self.mock_sleep.call_count, 11)

    def test_get_realtime_price_accumulate_error(self):
        self.polygon.last_trade.side_effect = requests.exceptions.RequestException('Test error')
        self.assertEqual(len(self.trading.errors), 0)
        self.trading.get_realtime_price('SYMA')
        self.assertEqual(len(self.trading.errors), 1)

    def test_update_trading_list(self):
        with mock.patch.object(time, 'time', side_effect=itertools.count(500, 50)):
            self.trading.update_trading_list()
        self.assertEqual(len(self.trading.trading_list), 4)

    def test_update_trading_list_prices(self):
        with mock.patch.object(time, 'time', side_effect=itertools.count(999)), \
                mock.patch.object(utils, 'web_scraping', return_value='35'):
            self.trading.update_trading_list_prices()
        self.assertEqual(self.trading.prices['^VIX'], 35)

    def test_update_stats(self):
        self.polygon.last_trade.return_value = LastTrade(666)
        with mock.patch.object(time, 'time', side_effect=itertools.count(999)):
            self.trading.update_stats(4, 1)
        self.assertEqual(self.trading.prices['SYMA'], 666)
        self.assertEqual(self.trading.prices['SYMB'], 666)
        self.assertEqual(self.trading.prices['SYMC'], 666)

    @parameterized.expand([(20, None, 10), (1000, 999, 1)])
    def test_wait_for_order_to_fill(self, timeout, deadline, list_call_count):
        self.alpaca.list_orders.return_value = ['fake_order']
        with mock.patch.object(time, 'time', side_effect=itertools.count(999)):
            self.trading.wait_for_order_to_fill(timeout, deadline)
        self.assertEqual(self.alpaca.list_orders.call_count, list_call_count)
        self.alpaca.cancel_all_orders.assert_called_once()

    @parameterized.expand([('limit',), ('market',)])
    def test_buy(self, order_type):
        with mock.patch.object(time, 'time', side_effect=itertools.count(999)):
            self.trading.update_trading_list()
        self.trading.buy(order_type)
        self.assertEqual(self.alpaca.submit_order.call_count, 4)

    @parameterized.expand([('limit',), ('market',)])
    def test_sell(self, order_type):
        self.alpaca.list_positions.return_value = [Position('SYMX', '10', '10.0', '100.0', '99.0'),
                                                   Position('SYMY', '20', '20.0', '400.0', '555.5')]
        self.trading.sell(order_type)
        self.assertEqual(self.alpaca.submit_order.call_count, 2)

    def test_trade(self):
        with mock.patch.object(time, 'time', side_effect=itertools.count(999)):
            self.trading.update_trading_list()
        self.alpaca.list_positions.side_effect = [
            # Original positions
            [Position('SYMX', '10', '10.0', '100.0', '99.0'),
             Position('SYMY', '20', '20.0', '400.0', '555.5')],
            # SYMY sold with limit order
            [Position('SYMX', '10', '10.0', '100.0', '99.0')],
            # SYMX sold with market order
            [],
            # SYMA filled with limit order, SYMB partially filled
            [Position('SYMA', '5', '88.0', '440.0', '440.0'),
             Position('SYMB', '1', '88.0', '88.0', '88.0')]]
        self.trading.trade()
        # Sell 2 + 1, Buy 4 + 3
        self.assertEqual(self.alpaca.submit_order.call_count, 10)

    def test_run_success(self):
        with mock.patch.object(time, 'time', side_effect=itertools.count(990)), \
                mock.patch.object(realtime.TradingRealTime, 'update_stats') as mock_update_stats, \
                mock.patch.object(realtime.TradingRealTime, 'update_trading_list_prices') as mock_update_prices, \
                mock.patch.object(realtime.TradingRealTime, 'update_trading_list') as mock_update_trading_list, \
                mock.patch.object(realtime.TradingRealTime, 'trade') as mock_trade:
            self.trading.run()
        self.assertEqual(mock_update_stats.call_count, 3)
        mock_update_prices.assert_called_once()
        mock_update_trading_list.assert_called_once()
        mock_trade.assert_called_once()

    def test_run_fail(self):
        with mock.patch.object(time, 'time', side_effect=itertools.repeat(800)), \
                mock.patch.object(realtime.TradingRealTime, 'update_stats') as mock_update_stats, \
                mock.patch.object(realtime.TradingRealTime, 'update_trading_list') as mock_update_trading_list, \
                mock.patch.object(realtime.TradingRealTime, 'trade_clock_watcher') as trade_clock_watcher, \
                mock.patch.object(utils, 'web_scraping', side_effect=requests.exceptions.HTTPError('Test error')), \
                self.assertRaises(requests.exceptions.HTTPError):
            self.trading.run()
        self.assertEqual(mock_update_stats.call_count, 3)
        mock_update_trading_list.assert_called_once()
        trade_clock_watcher.assert_called_once()

    @parameterized.expand([(True,), (False,)])
    def test_main(self, real_trade):
        saved_environ = dict(os.environ)
        os.environ['ALPACA_API_KEY'] = 'fake_api_key'
        os.environ['ALPACA_API_SECRET'] = 'fake_api_secret'
        os.environ['ALPACA_PAPER_API_KEY'] = 'fake_paper_api_key'
        os.environ['ALPACA_PAPER_API_SECRET'] = 'fake_paper_api_secret'
        mock_trading = mock.create_autospec(realtime.TradingRealTime)
        with mock.patch.object(tradeapi, 'REST', return_value=self.alpaca) as alpaca_init, \
                mock.patch.object(polygonapi, 'REST', return_value=self.polygon) as polygon_init, \
                mock.patch.object(realtime, 'TradingRealTime', return_value=mock_trading) as trading_init, \
                mock.patch.object(argparse.ArgumentParser, 'parse_args',
                                  return_value=argparse.Namespace(real_trade=real_trade,
                                                                  api_key=None,
                                                                  api_secret=None,
                                                                  force=False)):
            realtime.main()
        if real_trade:
            alpaca_init.assert_called_once_with('fake_api_key', 'fake_api_secret',
                                                utils.ALPACA_API_BASE_URL, 'v2')
            polygon_init.assert_called_once_with('fake_api_key')
        else:
            alpaca_init.assert_called_once_with('fake_paper_api_key', 'fake_paper_api_secret',
                                                utils.ALPACA_PAPER_API_BASE_URL, 'v2')
            polygon_init.assert_called_once_with('fake_paper_api_key')
        trading_init.assert_called_once()
        mock_trading.run.assert_called_once()
        os.environ.clear()
        os.environ.update(saved_environ)


if __name__ == '__main__':
    unittest.main()
