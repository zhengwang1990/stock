import functools
import os
import re
import requests
import retrying
import ta
import tensorflow.keras as keras
import numpy as np
import pandas as pd
import sys
import yfinance as yf
from concurrent import futures
from exclusions import EXCLUSIONS
from tqdm import tqdm

DATE_RANGE = 5
DATE_RANGE_CHANGE_CEIL = -0.5
REFERENCE_SYMBOL = 'AAPL'
DAYS_IN_A_YEAR = 250
CACHE_DIR = 'cache'
DATA_DIR = 'data'
OUTPUTS_DIR = 'outputs'
MODELS_DIR = 'models'
DEFAULT_HISTORY_LOAD = '5y'
MAX_STOCK_PICK = 8
VOLUME_FILTER_THRESHOLD = 100000
MAX_THREADS = 5
ML_FEATURES = [
    'Today_Change',
    'Yesterday_Change',
    'Day_Before_Yesterday_Change',
    'Twenty_Day_Change',
    'Day_Range_Change',
    'Year_High_Change',
    'Year_Low_Change',
    'Change_Average',
    'Change_Variance',
    'RSI',
    'MACD_Rate',
    'TSI',
    'WR',
    'VIX']
ALPACA_API_BASE_URL = 'https://api.alpaca.markets'
ALPACA_PAPER_API_BASE_URL = 'https://paper-api.alpaca.markets'
DEFAULT_MODEL = 'model_p727217.hdf5'


class TradingBase(object):
    """Basic trade utils."""

    def __init__(self, alpaca, period=None, model=None, load_history=True):
        model = model or DEFAULT_MODEL
        self.alpaca = alpaca
        self.root_dir = os.path.dirname(os.path.realpath(__file__))
        self.model = keras.models.load_model(os.path.join(self.root_dir, MODELS_DIR, model))
        self.hists = {}
        self.period = period or DEFAULT_HISTORY_LOAD
        self.cache_path = os.path.join(self.root_dir, CACHE_DIR, get_business_day(1))
        os.makedirs(os.path.join(self.cache_path, self.period), exist_ok=True)
        self.is_market_open = self.alpaca.get_clock().is_open
        self.history_length = self.get_history_length(self.period)
        self.history_dates = self.get_history_dates(self.period)
        if not load_history:
            return
        self.load_all_symbols()
        self.load_histories(self.period)
        self.read_series_from_histories(self.period)

    def load_all_symbols(self):
        assets = self.alpaca.list_assets()
        self.symbols = ['^VIX'] + [
            asset.symbol for asset in assets
            if re.match('^[A-Z]*$', asset.symbol)
            and asset.symbol not in EXCLUSIONS]

    @retrying.retry(stop_max_attempt_number=10, wait_fixed=1000 * 60 * 10)
    def load_histories(self, period):
        """Loads history of all stock symbols."""
        print('Loading stock histories...')
        threads = []
        # Allow at most 10 errors
        error_tol = 10

        with futures.ThreadPoolExecutor(max_workers=MAX_THREADS) as pool:
            for symbol in self.symbols:
                t = pool.submit(self.load_history, symbol, period)
                threads.append(t)
            iterator = tqdm(threads, ncols=80) if sys.stdout.isatty() else threads
            for t in iterator:
                try:
                    t.result()
                except Exception as e:
                    print(e)
                    error_tol -= 1
                    if error_tol <= 0:
                        raise e

    def read_series_from_histories(self, period):
        """Reads out close price and volume."""
        self.closes = {}
        self.volumes = {}

        for symbol, hist in self.hists.items():
            close = hist.get('Close')
            volume = hist.get('Volume')
            self.closes[symbol] = np.array(close)
            self.volumes[symbol] = np.array(volume)
        print('Attempt to load %d symbols' % (len(self.symbols)))
        print('%d loaded symbols after drop symbols traded less than %s' % (
            len(self.closes), period))

    @retrying.retry(stop_max_attempt_number=2, wait_fixed=500)
    def load_history(self, symbol, period):
        """Loads history for a single symbol."""
        cache_name = os.path.join(self.cache_path, self.period, 'history_%s.csv' % (symbol,))
        if os.path.isfile(cache_name):
            hist = pd.read_csv(cache_name, index_col=0, parse_dates=True)
        else:
            try:
                tk = yf.Ticker(symbol)
                hist = tk.history(period=period, interval='1d')
                if len(hist):
                    hist.to_csv(cache_name)
            except Exception as e:
                print('Can not get history of %s: %s' % (symbol, e))
                raise e
        drop_key = pd.datetime.today().date()
        if self.is_market_open and drop_key in hist.index:
            hist.drop(drop_key, inplace=True)
        if symbol == REFERENCE_SYMBOL or len(hist) == self.history_length:
            self.hists[symbol] = hist
        elif symbol in ('QQQ', 'SPY', '^VIX'):
            os.remove(cache_name)
            raise Exception('Error loading %s: expect length %d, but got %d.' % (
                symbol, self.history_length, len(hist)))

    def get_history_length(self, period):
        """Get the number of trading days in a given period."""
        self.load_history(REFERENCE_SYMBOL, period=period)
        return len(self.hists[REFERENCE_SYMBOL])

    def get_history_dates(self, period):
        """Gets the list trading dates in a given period."""
        self.load_history(REFERENCE_SYMBOL, period=period)
        return self.hists[REFERENCE_SYMBOL].index

    def get_buy_symbols(self, prices=None, cutoff=None):
        """Gets symbols which trigger buy signals.

        A list of tuples will be returned with symbol, weight and all ML features.
        """
        if not (prices or cutoff) or (prices and cutoff):
            raise Exception('Exactly one of prices or cutoff must be provided')
        buy_info = []
        iterator = (tqdm(self.closes.items(), ncols=80, leave=False)
                    if cutoff and sys.stdout.isatty() else self.closes.items())
        for symbol, close in iterator:
            # Non-tradable symbols
            if symbol == '^VIX':
                continue
            if cutoff:
                close_year = close[cutoff - DAYS_IN_A_YEAR:cutoff]
                volumes_year = self.volumes[symbol][cutoff - DAYS_IN_A_YEAR:cutoff]
            else:
                close_year = close[-DAYS_IN_A_YEAR:]
                volumes_year = self.volumes[symbol][-DAYS_IN_A_YEAR:]
            avg_trading_volume = np.average(np.multiply(
                close_year[-20:], volumes_year[-20:]))
            # Enough trading volume
            if avg_trading_volume < VOLUME_FILTER_THRESHOLD:
                continue
            if prices:
                price = prices.get(symbol, 1E10)
            else:
                price = close[cutoff]
            threshold = self.get_threshold(symbol, cutoff)
            day_range_max = np.max(close_year[-DATE_RANGE:])
            day_range_change = price / day_range_max - 1
            today_change = price / close_year[-1] - 1
            # Today change is tamed
            if np.abs(today_change) > 0.5 * np.abs(day_range_change):
                continue
            # Enough drop but not too crazy
            if threshold > day_range_change > DATE_RANGE_CHANGE_CEIL:
                buy_info.append(symbol)

        buy_symbols = []
        for symbol in buy_info:
            ml_feature = self.get_ml_feature(symbol, prices=prices, cutoff=cutoff)
            x = np.array([[ml_feature[key] for key in ML_FEATURES]])
            weight = self.model.predict(x)[0]
            buy_symbols.append((symbol, weight, ml_feature))
        return buy_symbols

    def get_trading_list(self, buy_symbols=None, **kwargs):
        if buy_symbols is None:
            buy_symbols = self.get_buy_symbols(**kwargs)
        buy_symbols.sort(key=lambda s: s[1], reverse=True)
        n_symbols = min(MAX_STOCK_PICK, len(buy_symbols))
        trading_list = []
        for i in range(len(buy_symbols)):
            symbol = buy_symbols[i][0]
            weight = buy_symbols[i][1]
            proportion = 1 / n_symbols if i < n_symbols else 0
            trading_list.append((symbol, proportion, weight))
        return trading_list

    def get_ml_feature(self, symbol, prices=None, cutoff=None):
        if prices:
            price = prices.get(symbol, 1E10)
            vix = prices['^VIX']
        else:
            price = self.closes[symbol][cutoff]
            vix = self.closes['^VIX'][cutoff]

        if cutoff:
            close = self.closes[symbol][cutoff - DAYS_IN_A_YEAR:cutoff]
            high = np.array(self.hists[symbol].get('High')[cutoff - DAYS_IN_A_YEAR:cutoff])
            low = np.array(self.hists[symbol].get('Low')[cutoff - DAYS_IN_A_YEAR:cutoff])
        else:
            close = self.closes[symbol][-DAYS_IN_A_YEAR:]
            high = np.array(self.hists[symbol].get('High')[-DAYS_IN_A_YEAR:])
            low = np.array(self.hists[symbol].get('Low')[-DAYS_IN_A_YEAR:])
        # Basic stats
        day_range_change = price / np.max(close[-DATE_RANGE:]) - 1
        today_change = price / close[-1] - 1
        yesterday_change = close[-1] / close[-2] - 1
        day_before_yesterday_change = close[-2] / close[-3] - 1
        twenty_day_change = price / close[-20] - 1
        year_high_change = price / np.max(close) - 1
        year_low_change = price / np.min(close) - 1
        all_changes = [close[t + 1] / close[t] - 1
                       for t in range(len(close) - 1)
                       if close[t + 1] > 0 and close[t] > 0]
        # Technical indicators
        close = np.append(close, price)
        high = np.append(high, price)
        low = np.append(low, price)
        pd_close = pd.Series(close)
        pd_high = pd.Series(high)
        pd_low = pd.Series(low)
        rsi = ta.momentum.rsi(pd_close).values[-1]
        macd_rate = ta.trend.macd_diff(pd_close).values[-1] / price
        wr = ta.momentum.wr(pd_high, pd_low, pd_close).values[-1]
        tsi = ta.momentum.tsi(pd_close).values[-1]
        feature = {'Today_Change': today_change,
                   'Yesterday_Change': yesterday_change,
                   'Day_Before_Yesterday_Change': day_before_yesterday_change,
                   'Twenty_Day_Change': twenty_day_change,
                   'Day_Range_Change': day_range_change,
                   'Year_High_Change': year_high_change,
                   'Year_Low_Change': year_low_change,
                   'Change_Average': np.mean(all_changes),
                   'Change_Variance': np.var(all_changes),
                   'RSI': rsi,
                   'MACD_Rate': macd_rate,
                   'WR': wr,
                   'TSI': tsi,
                   'VIX': vix}
        return feature

    @functools.lru_cache(maxsize=10000)
    def get_threshold(self, symbol, cutoff=None):
        """Gets threshold for a symbol.
        """
        if cutoff:
            close_year = self.closes[symbol][cutoff - DAYS_IN_A_YEAR:cutoff]
        else:
            close_year = self.closes[symbol][-DAYS_IN_A_YEAR:]
        down_percent = [close_year[i] / np.max(close_year[i - DATE_RANGE:i]) - 1
                        for i in range(DATE_RANGE, len(close_year))
                        if close_year[i] < np.max(close_year[i - DATE_RANGE:i])]
        threshold = np.mean(down_percent) - 2.5 * np.std(down_percent)
        return threshold


def get_business_day(offset):
    day = pd.datetime.today() - pd.tseries.offsets.BDay(offset) if offset else pd.datetime.today()
    return '%4d-%02d-%02d' % (day.year, day.month, day.day)


def get_header(title):
    header_left = '== [ %s ] ' % (title,)
    return header_left + '=' * (80 - len(header_left))


@retrying.retry(stop_max_attempt_number=3, wait_exponential_multiplier=500)
def web_scraping(url, prefixes):
    r = requests.get(url, timeout=3)
    c = str(r.content)
    for prefix in prefixes:
        prefix_pos = c.find(prefix)
        if prefix_pos >= 0:
            pos = prefix_pos + len(prefix)
            s = ''
            while c[pos] > '9' or c[pos] < '0':
                pos += 1
            while '9' >= c[pos] >= '0' or c[pos] in ['.', ',']:
                if c[pos] != ',':
                    s += c[pos]
                pos += 1
            if pos - prefix_pos < 100:
                return s
    else:
        raise Exception('[status %d] %s not found in %s' % (r.status_code, prefixes, url))


def bi_print(message, output_file):
    """Prints to both stdout and a file."""
    print(message)
    if output_file:
        output_file.write(message)
        output_file.write('\n')
        output_file.flush()
