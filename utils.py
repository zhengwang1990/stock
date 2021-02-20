import datetime
import logging
import os
import re
import sys
from concurrent import futures
import numpy as np
import pandas as pd
import retrying
from tqdm import tqdm

from exclusions import EXCLUSIONS

REFERENCE_SYMBOL = 'AAPL'
DAYS_IN_A_WEEK = 5
DAYS_IN_A_MONTH = 20
DAYS_IN_A_QUARTER = 60
DAYS_IN_A_YEAR = 250
CACHE_DIR = 'cache'
OUTPUTS_DIR = 'outputs'
MAX_STOCK_PICK = 8
MAX_PROPORTION = 0.25
VOLUME_FILTER_THRESHOLD = 1E6
ESSENTIAL_SYMBOLS = ['TQQQ', 'QQQ', 'SPY']

ALPACA_API_BASE_URL = 'https://api.alpaca.markets'
ALPACA_PAPER_API_BASE_URL = 'https://paper-api.alpaca.markets'


class NetworkError(Exception):
    """Network error occurred."""


class NotFoundError(Exception):
    """Content not found."""


class TradingBase(object):
    """Basic trade utils."""

    def __init__(self, alpaca, start_date=None, end_date=None):
        self.alpaca = alpaca
        self.root_dir = os.path.dirname(os.path.realpath(__file__))
        self.hists, self.closes, self.opens, self.volumes = {}, {}, {}, {}
        self.symbols = []
        self.errors = []
        self.history_start_date = None
        self.history_end_date = None
        self.prices = None

        if end_date:
            e = min(pd.to_datetime(end_date) + pd.tseries.offsets.BDay(1),
                    datetime.datetime.today() + pd.tseries.offsets.BDay(1))
            self.history_end_date = e.strftime('%Y-%m-%d')
        else:
            self.history_end_date = get_business_day(1)
        if start_date:
            s = pd.to_datetime(start_date) - pd.tseries.offsets.BDay(300)
            self.history_start_date = s.strftime('%Y-%m-%d')
        else:
            s = datetime.datetime.today() - pd.tseries.offsets.DateOffset(years=2)
            self.history_start_date = s.strftime('%Y-%m-%d')

        self.cache_path = os.path.join(self.root_dir, CACHE_DIR,
                                       get_business_day(0),
                                       self.history_start_date,
                                       self.history_end_date)
        os.makedirs(self.cache_path, exist_ok=True)
        self.is_market_open = self.alpaca.get_clock().is_open
        calendar = alpaca.get_calendar(datetime.date.today(), datetime.date.today())
        self.is_trading_day = (len(calendar) > 0 and
                               calendar[0].date.strftime('%F') == datetime.date.today().strftime('%F'))
        self.history_length = self.get_history_length()
        self.history_dates = self.get_history_dates()
        self.load_all_symbols()
        self.load_histories()
        self.read_series_from_histories()

    def load_all_symbols(self):
        """Loads all tradable symbols on Alpaca."""
        assets = self.alpaca.list_assets()
        self.symbols = [asset.symbol for asset in assets
                        if re.match('^[A-Z]*$', asset.symbol)
                        and asset.tradable and asset.marginable
                        and asset.shortable and asset.easy_to_borrow]
        self.symbols = list(set(self.symbols).difference(EXCLUSIONS))
        for symbol in ESSENTIAL_SYMBOLS:
          if symbol not in self.symbols:
            self.symbols.append(symbol)

    def load_histories(self):
        """Loads history of all stock symbols."""
        logging.info('Loading stock histories...')
        threads = []
        with futures.ThreadPoolExecutor(max_workers=4) as pool:
            for symbol in self.symbols:
                t = pool.submit(self.load_history, symbol)
                threads.append(t)
            iterator = tqdm(threads, ncols=80) if sys.stdout.isatty() else threads
            for t in iterator:
                try:
                    t.result()
                except Exception as e:
                    logging.error('Error occurred in load_histories: %s', e)
                    self.errors.append(sys.exc_info())

    def read_series_from_histories(self):
        """Reads out close price and volume."""
        for symbol, hist in self.hists.items():
            closes = np.array(hist.get('Close'))
            if np.any(closes <= 0):
                logging.warning('Found non-positive close prices in %s. Skip.', symbol)
                continue
            self.closes[symbol] = closes
            self.opens[symbol] = np.array(hist.get('Open'))
            self.volumes[symbol] = np.array(hist.get('Volume'))
        logging.info('Attempt to load %d symbols, and %d symbols actually loaded',
                     len(self.symbols), len(self.closes))

    @retrying.retry(stop_max_attempt_number=3, wait_fixed=1000)
    def load_history(self, symbol):
        """Loads history for a single symbol."""
        cache_name = os.path.join(self.cache_path, 'history_%s.csv' % (symbol,))
        if os.path.isfile(cache_name):
            hist = pd.read_csv(cache_name, index_col=0, parse_dates=True)
        else:
            aggs = self.alpaca.get_aggs(symbol, 1, 'day',
                                        self.history_start_date, self.history_end_date)
            hist = pd.DataFrame(
                [[pd.to_datetime(agg.timestamp.date()), agg.open, agg.close, agg.volume] for agg in aggs],
                columns=['Date', 'Open', 'Close', 'Volume'])
            hist.set_index('Date', inplace=True)
            drop_key = datetime.datetime.today().date()
            if (self.is_market_open or not self.is_trading_day) and drop_key in hist.index:
                hist.drop(drop_key, inplace=True)
            if len(hist):
                hist.to_csv(cache_name)
            else:
                raise NotFoundError('History of %s not found' % (symbol,))
        if symbol == REFERENCE_SYMBOL or len(hist) == self.history_length:
            self.hists[symbol] = hist
        elif symbol in ESSENTIAL_SYMBOLS:
            os.remove(cache_name)
            raise Exception('Error loading %s: expect length %d, but got %d.' % (
                symbol, self.history_length, len(hist)))

    def get_history_length(self):
        """Get the number of trading days in the period of interest."""
        self.load_history(REFERENCE_SYMBOL)
        return len(self.hists[REFERENCE_SYMBOL])

    def get_history_dates(self):
        """Gets the list trading dates in the period of interest."""
        self.load_history(REFERENCE_SYMBOL)
        return self.hists[REFERENCE_SYMBOL].index

    def get_stock_universe(self, cutoff=-1):
        stock_universe = []
        for symbol in self.closes.keys():
            avg_trading_volume = np.average(np.multiply(
                self.closes[symbol][cutoff - DAYS_IN_A_MONTH:cutoff],
                self.volumes[symbol][cutoff - DAYS_IN_A_MONTH:cutoff]))
            # Enough trading volume
            if avg_trading_volume < VOLUME_FILTER_THRESHOLD:
                continue
            stock_universe.append(symbol)
        return stock_universe

    def get_buy_symbols(self, cutoff=-1):
        """Gets symbols which trigger buy signals."""
        buy_symbols = []
        symbols_dip = self.dip_reversion(cutoff)
        buy_symbols.extend([(symbol, weight, 'long') for symbol, weight in symbols_dip])
        return buy_symbols

    def dip_reversion(self, cutoff=-1):
        stock_universe = self.get_stock_universe(cutoff)
        symbols_dip = []
        n = 7
        for symbol in stock_universe:
            closes_year = self.closes[symbol][cutoff - DAYS_IN_A_YEAR:cutoff]
            n_day_returns = [np.log(closes_year[i] / closes_year[i - n])
                             for i in range(n, len(closes_year))]
            price = self.closes[symbol][cutoff]
            if not price:
                continue
            if (price > 1.2 * closes_year[-DAYS_IN_A_MONTH] and
                    price / closes_year[-DAYS_IN_A_MONTH] >
                    closes_year[-DAYS_IN_A_MONTH] / closes_year[-DAYS_IN_A_YEAR]):
                continue
            mean = np.mean(n_day_returns)
            std = np.std(n_day_returns)
            threshold = mean - 3 * std
            n_day_return = np.log(price / closes_year[-n])
            if n_day_return >= threshold:
                continue
            next_day_return = [
                np.log(closes_year[i + 1] / closes_year[i])
                for i in range(n, len(closes_year) - 1)
                if n_day_returns[i - n] < threshold]
            avg_next_day_return = np.mean(next_day_return) if next_day_return else 0
            if avg_next_day_return > 0 and 0.05 > np.log(price / closes_year[-1]) > 0.5 * n_day_return:
                symbols_dip.append((symbol, avg_next_day_return))
        return symbols_dip

    def get_trading_list(self, buy_symbols=None, cutoff=-1):
        """Gets a list of symbols with trading information."""
        if buy_symbols is None:
            buy_symbols = self.get_buy_symbols(cutoff)
        buy_symbols.sort(key=lambda s: s[1], reverse=True)
        trading_list = []
        global_proportion = self.get_market_proportion(cutoff)
        for i in range(len(buy_symbols)):
            symbol, weight, side = buy_symbols[i]
            proportion = min(1 / min(len(buy_symbols), MAX_STOCK_PICK),
                             MAX_PROPORTION) if i < MAX_STOCK_PICK else 0
            trading_list.append((symbol, proportion, side))
        if len(buy_symbols) < 1 / MAX_PROPORTION:
            trading_list.append(
                ('TQQQ', (1 - len(buy_symbols) * MAX_PROPORTION) * global_proportion, 'long'))
        return trading_list

    def get_market_proportion(self, cutoff=-1):
        monthly_gain = self.closes['QQQ'][cutoff] / self.closes['QQQ'][cutoff - DAYS_IN_A_MONTH] - 1
        quarterly_gain = self.closes['QQQ'][cutoff] / self.closes['QQQ'][cutoff - DAYS_IN_A_QUARTER] - 1
        return np.clip(1 - 8 * quarterly_gain ** 2 - 20 * monthly_gain ** 2, 0.2, 1)


def get_business_day(offset):
    day = datetime.datetime.today() + pd.tseries.offsets.BDay(offset)
    return '%4d-%02d-%02d' % (day.year, day.month, day.day)


def get_header(title):
    header_left = '== [ %s ] ' % (title,)
    return header_left + '=' * (80 - len(header_left))


def to_percent(f, sign=False):
    formatter = '%+.2f%%' if sign else '%.2f%%'
    return formatter % (f * 100,)


def has_peak(a, multiplier=1):
    std = np.std(a)
    mean = np.mean(a)
    for elem in a:
        if elem > mean + multiplier * std:
            return True
    return False


def logging_config(logging_file=None):
    """Configuration for logging."""
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter(
        '[%(levelname)s] [%(asctime)s] [%(filename)s:%(lineno)d] [%(threadName)s]\n%(message)s')
    stream_handler = logging.StreamHandler()
    if sys.stdout.isatty():
        stream_handler.setLevel(logging.INFO)
    else:
        stream_handler.setLevel(logging.WARNING)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)
    if logging_file:
        file_handler = logging.FileHandler(logging_file)
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
