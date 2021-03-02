import alpaca_trade_api as tradeapi
import argparse
import collections
import datetime
import itertools
import pandas as pd
import realtime
import requests
import os
import time
import unittest
import unittest.mock as mock
import utils
import yfinance as yf
from iexfinance.stocks import Stock
from parameterized import parameterized

Clock = collections.namedtuple('Clock', ['is_open', 'next_close'])
Asset = collections.namedtuple('Asset', ['symbol', 'tradable', 'marginable',
                                         'shortable', 'easy_to_borrow'])
Account = collections.namedtuple('Account', ['equity', 'cash'])
Position = collections.namedtuple('Position', ['symbol', 'qty', 'current_price',
                                               'market_value', 'cost_basis',
                                               'avg_entry_price'])


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
        self.alpaca = mock.create_autospec(tradeapi.REST)
        self.alpaca.get_account.return_value = Account(2000, 2000)
        self.alpaca.list_assets.return_value = [Asset(symbol, True, True, True, True)
                                                for symbol in [utils.REFERENCE_SYMBOL,
                                                               'SYMA', 'QQQ', 'SPY', 'TQQQ']]
        fake_next_close = mock.Mock()
        fake_next_close.timestamp.return_value = 1000
        self.alpaca.get_clock.return_value = Clock(True, fake_next_close)
        self.alpaca.list_orders.return_value = []
        self.patch_get_price = mock.patch.object(Stock, 'get_price', return_value='69')
        self.patch_get_price.start()
        fake_closes = ([100] * 9 + [90]) * 98 + ([100] * 8 + [70] * 2) * 2
        fake_timestamps = [datetime.datetime.today().date() - pd.tseries.offsets.DateOffset(offset)
                           for offset in range(999, -1, -1)]
        fake_history = pd.DataFrame(
            [[fake_timestamps[i], 100, fake_closes[i], 1E6] for i in range(1000)],
            columns=['Date', 'Open', 'Close', 'Volume'])
        fake_history.set_index('Date', inplace=True)
        self.patch_yf = mock.patch.object(yf.Ticker, 'history', return_value=fake_history)
        self.patch_yf.start()
        self.trading = realtime.TradingRealTime(self.alpaca)

    def tearDown(self):
        self.patch_open.stop()
        self.patch_isfile.stop()
        self.patch_mkdirs.stop()
        self.patch_sleep.stop()
        self.patch_to_csv.stop()
        self.patch_get_price.stop()
        self.patch_yf.stop()

    def test_trade_clock_watcher(self):
        with mock.patch.object(realtime.TradingRealTime, 'trade') as trade, \
                mock.patch.object(time, 'time', side_effect=itertools.count(900)):
            self.trading.trade_clock_watcher()
            trade.assert_called_once()
            self.assertEqual(self.mock_sleep.call_count, 11)

    def test_get_realtime_price_accumulate_error(self):
        with mock.patch.object(Stock, 'get_price',
                               side_effect=requests.exceptions.RequestException('Test error')):
            self.assertEqual(len(self.trading.errors), 0)
            self.trading.get_realtime_price('SYMA')
            self.assertEqual(len(self.trading.errors), 1)

    def test_update_trading_list(self):
        with mock.patch.object(time, 'time', side_effect=itertools.count(500, 50)):
            self.trading.update_trading_list()
        self.assertEqual(len(self.trading.trading_list), 5)

    def test_update_trading_list_prices(self):
        with mock.patch.object(time, 'time', side_effect=itertools.count(999)):
            self.trading.update_trading_list_prices()
        self.assertEqual(self.trading.prices['SYMA'], 69)

    def test_update_all_prices(self):
        with mock.patch.object(Stock, 'get_price', return_value='666'), \
             mock.patch.object(time, 'time', side_effect=itertools.count(999)):
                self.trading.update_all_prices()
        self.assertEqual(self.trading.prices['SYMA'], 666)
        self.assertEqual(self.trading.prices['SPY'], 666)
        self.assertEqual(self.trading.prices['TQQQ'], 666)

    @parameterized.expand([(20, None, 16), (1000, 999, 7)])
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
        self.assertEqual(self.alpaca.submit_order.call_count, 5)

    @parameterized.expand([('limit',), ('market',)])
    def test_sell(self, order_type):
        self.alpaca.list_positions.return_value = [
            Position('SYMX', '10', '10.0', '100.0', '99.0', '9.9'),
            Position('SYMY', '20', '20.0', '400.0', '555.5', '27.7')]
        self.trading.sell(order_type)
        self.assertEqual(self.alpaca.submit_order.call_count, 2)

    def test_trade(self):
        with mock.patch.object(time, 'time', side_effect=itertools.count(999)):
            self.trading.update_trading_list()
        self.alpaca.list_positions.side_effect = [
            # Original positions. Next: sell 4 SYMA, and all of SYMX, SYMY.
            [Position('SYMA', '10', '10.0', '100.0', '99.0', '9.9'),  # Sell 4 shares
             Position('TQQQ', '1', '10.0', '100.0', '99.0', '27.7'),  # No sell transactions
             Position('SYMX', '10', '10.0', '100.0', '99.0', '9.9'),
             Position('SYMY', '20', '20.0', '400.0', '555.5', '27.7')],
            # SYMY sold with limit order. Next: Sell all of SYMX.
            [Position('SYMA', '6', '10.0', '100.0', '99.0', '9.9'),  # No sell transactions
             Position('TQQQ', '1', '10.0', '100.0', '99.0', '9.9'),  # No sell transactions
             Position('SYMX', '10', '10.0', '100.0', '99.0', '9.9')],
            # SYMX sold with market order. SYMA has most of it. Next: Buy AAPL, SYMA, SPY, QQQ, TQQQ.
            [Position('SYMA', '6', '10.0', '100.0', '99.0', '9.9'),
             Position('TQQQ', '1', '10.0', '100.0', '99.0', '9.9')],
            # SYMA filled with limit order, TQQQ partially filled. Next: Buy AAPL, QQQ, TQQQ.
            [Position('SYMA', '7', '88.0', '440.0', '440.0', '60'),
             Position('SPY', '7', '88.0', '88.0', '88.0', '14.5'),
             Position('TQQQ', '3', '10.0', '100.0', '99.0', '33')]]
        self.trading.trade()
        # Sell 3 + 1, Buy 5 + 3
        self.assertEqual(self.alpaca.submit_order.call_count, 12)

    def test_run_success(self):
        with mock.patch.object(time, 'time', side_effect=itertools.count(990)), \
             mock.patch.object(realtime.TradingRealTime, 'update_all_prices') as mock_update_all_prices, \
                mock.patch.object(realtime.TradingRealTime, 'update_trading_list_prices') as mock_update_prices, \
                mock.patch.object(realtime.TradingRealTime, 'update_trading_list') as mock_update_trading_list, \
                mock.patch.object(realtime.TradingRealTime, 'trade') as mock_trade:
            self.trading.run()
        mock_update_all_prices.assert_called_once()
        mock_update_prices.assert_called_once()
        mock_update_trading_list.assert_called_once()
        mock_trade.assert_called_once()

    def test_run_fail(self):
        with mock.patch.object(Stock, 'get_price',
                               side_effect=requests.exceptions.HTTPError('Test error')), \
             mock.patch.object(time, 'time', side_effect=itertools.repeat(800)), \
             mock.patch.object(realtime.TradingRealTime, 'update_all_prices') as mock_update_all_prices, \
                mock.patch.object(realtime.TradingRealTime, 'trade_clock_watcher') as trade_clock_watcher, \
                self.assertRaises(requests.exceptions.HTTPError):
            self.trading.run()
        mock_update_all_prices.assert_called_once()
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
        else:
            alpaca_init.assert_called_once_with('fake_paper_api_key', 'fake_paper_api_secret',
                                                utils.ALPACA_PAPER_API_BASE_URL, 'v2')
        trading_init.assert_called_once()
        mock_trading.run.assert_called_once()
        os.environ.clear()
        os.environ.update(saved_environ)


if __name__ == '__main__':
    unittest.main()
