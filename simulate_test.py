import alpaca_trade_api as tradeapi
import argparse
import collections
import datetime
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import simulate
import os
import unittest
import unittest.mock as mock
import utils
import yfinance as yf

Clock = collections.namedtuple('Clock', ['is_open'])
Asset = collections.namedtuple('Asset', ['symbol', 'tradable', 'marginable',
                                         'shortable', 'easy_to_borrow'])


class TradingSimulateTest(unittest.TestCase):

    def setUp(self):
        self.patch_open = mock.patch('builtins.open', mock.mock_open())
        self.patch_open.start()
        self.patch_isfile = mock.patch.object(os.path, 'isfile', return_value=False)
        self.patch_isfile.start()
        self.patch_mkdirs = mock.patch.object(os, 'makedirs')
        self.patch_mkdirs.start()
        self.patch_savefig = mock.patch.object(plt, 'savefig')
        self.mock_savefig = self.patch_savefig.start()
        self.patch_tight_layout = mock.patch.object(plt, 'tight_layout')
        self.patch_tight_layout.start()
        np.random.seed(0)
        self.alpaca = mock.create_autospec(tradeapi.REST)
        self.alpaca.list_assets.return_value = [Asset(symbol, True, True, True, True)
                                                for symbol in [utils.REFERENCE_SYMBOL,
                                                               'SYMA', 'QQQ', 'SPY', 'TQQQ']]
        self.alpaca.get_clock.return_value = Clock(False)
        fake_closes = np.append(np.random.random(990) * 10 + 100, np.random.random(10) * 10 + 90)
        fake_timestamps = [datetime.datetime.today().date() - pd.tseries.offsets.DateOffset(offset)
                           for offset in range(999, -1, -1)]
        fake_history = pd.DataFrame(
            [[fake_timestamps[i], 100, fake_closes[i], 1E6] for i in range(1000)],
            columns=['Date', 'Open', 'Close', 'Volume'])
        fake_history.set_index('Date', inplace=True)
        self.patch_yf = mock.patch.object(yf.Ticker, 'history', return_value=fake_history)
        self.patch_yf.start()
        self.trading = simulate.TradingSimulate(
            self.alpaca,
            start_date=(datetime.datetime.today().date() - pd.tseries.offsets.BDay(30)).strftime('%F'))

    def tearDown(self):
        self.patch_open.stop()
        self.patch_isfile.stop()
        self.patch_mkdirs.stop()
        self.patch_savefig.stop()
        self.patch_tight_layout.stop()
        self.patch_yf.stop()

    def test_run(self):
        self.trading.run()
        self.assertGreaterEqual(self.mock_savefig.call_count, 3)  # quarter, year, total plots

    def test_main(self):
        mock_trading = mock.create_autospec(simulate.TradingSimulate)
        with mock.patch.object(tradeapi, 'REST', return_value=self.alpaca) as alpaca_init, \
                mock.patch.object(simulate, 'TradingSimulate', return_value=mock_trading) as trading_init, \
                mock.patch.object(
                    argparse.ArgumentParser, 'parse_args',
                    return_value=argparse.Namespace(start_date=None, end_date=None,
                                                    api_key='fake_api_key',
                                                    api_secret='fake_api_secret')):
            simulate.main()
        alpaca_init.assert_called_once_with('fake_api_key', 'fake_api_secret',
                                            utils.ALPACA_PAPER_API_BASE_URL, 'v2')
        trading_init.assert_called_once()
        mock_trading.run.assert_called_once()


if __name__ == '__main__':
    unittest.main()
