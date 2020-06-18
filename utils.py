import functools
import logging
import numpy as np
import os
import pandas as pd
import re
import requests
import retrying
import sys
import ta.momentum as momentum
import ta.trend as trend
import tensorflow.keras as keras
import yfinance as yf
from concurrent import futures
from exclusions import EXCLUSIONS
from tqdm import tqdm
from scipy import stats

REFERENCE_SYMBOL = 'AAPL'
DAYS_IN_A_YEAR = 250
DAYS_IN_A_WEEK = 5
DAYS_IN_A_MONTH = 20
DAYS_IN_A_QUARTER = 60
CACHE_DIR = 'cache'
DATA_DIR = 'data'
OUTPUTS_DIR = 'outputs'
MODELS_DIR = 'models'
DEFAULT_HISTORY_LOAD = '2y'
MAX_STOCK_PICK = 8
MAX_PROPORTION = 0.25
VOLUME_FILTER_THRESHOLD = 1000000
ML_FEATURES = [
    #'Day_1_Return',
    #'Day_2_Return',
    #'Day_3_Return',
    'Weekly_Return',
    'Monthly_Return',
    'Quarterly_Return',
    #'From_Weekly_High',
    #'From_Weekly_Low',
    'RSI',
    'MACD_Rate',
    'TSI',
    'Acceleration',
    'Momentum',
    'Weekly_Skewness',
    'Monthly_Skewness',
    'Weekly_Volatility',
    'Monthly_Volatility',
    'Z_Score',
    'Monthly_Avg_Dollar_Volume',
    'VIX']
ALPACA_API_BASE_URL = 'https://api.alpaca.markets'
ALPACA_PAPER_API_BASE_URL = 'https://paper-api.alpaca.markets'
DEFAULT_MODEL = 'model_p727217.hdf5'


class NetworkError(Exception):
    """Network error occurred."""


class NotFoundError(Exception):
    """Content not found."""


class TradingBase(object):
    """Basic trade utils."""

    def __init__(self, alpaca, period=None, start_date=None, end_date=None,
                 model=None, load_history=True):
        model = model or DEFAULT_MODEL
        self.alpaca = alpaca
        self.root_dir = os.path.dirname(os.path.realpath(__file__))
        self.model = keras.models.load_model(os.path.join(self.root_dir, MODELS_DIR, model))
        self.hists, self.closes, self.volumes = {}, {}, {}
        self.symbols = []
        self.sectors = {}
        self.period = period
        self.start_date = start_date
        self.end_date = end_date
        if not period:
            if not start_date and not end_date:
                self.period = DEFAULT_HISTORY_LOAD
            if end_date:
                e = min(pd.to_datetime(end_date) + pd.tseries.offsets.BDay(1), pd.datetime.today())
                self.end_date = e.strftime('%Y-%m-%d')
            else:
                self.end_date = pd.datetime.today().strftime('%Y-%m-%d')
            if start_date:
                s = pd.to_datetime(start_date) - pd.tseries.offsets.BDay(300)
                self.start_date = s.strftime('%Y-%m-%d')
        cache_root = os.path.join(self.root_dir, CACHE_DIR, get_business_day(0))
        if self.period:
            self.cache_path = os.path.join(cache_root, self.period)
        else:
            self.cache_path = os.path.join(cache_root, self.start_date, self.end_date)
        os.makedirs(self.cache_path, exist_ok=True)
        self.is_market_open = self.alpaca.get_clock().is_open
        self.history_length = self.get_history_length()
        self.history_dates = self.get_history_dates()
        if not load_history:
            return
        self.load_all_symbols()
        self.load_histories()
        self.read_series_from_histories()

    def load_all_symbols(self):
        """Loads all tradable symbols on Alpaca."""
        assets = self.alpaca.list_assets()
        self.symbols = (['^VIX'] +
                        [asset.symbol for asset in assets
                         if re.match('^[A-Z]*$', asset.symbol) and asset.symbol not in EXCLUSIONS
                         and asset.tradable and asset.marginable and asset.shortable and asset.easy_to_borrow])

    @retrying.retry(stop_max_attempt_number=10, wait_fixed=1000 * 60 * 10)
    def load_histories(self):
        """Loads history of all stock symbols."""
        logging.info('Loading stock histories...')
        threads = []
        # Allow at most 10 errors
        error_tol = 10

        with futures.ThreadPoolExecutor(max_workers=5) as pool:
            for symbol in self.symbols:
                t = pool.submit(self.load_history, symbol)
                threads.append(t)
            iterator = tqdm(threads, ncols=80) if sys.stdout.isatty() else threads
            for t in iterator:
                try:
                    t.result()
                except Exception as e:
                    logging.error('Error occurred in load_histories: %s', e)
                    error_tol -= 1
                    if self.period and error_tol <= 0:
                        raise e

    def read_series_from_histories(self):
        """Reads out close price and volume."""
        for symbol, hist in self.hists.items():
            close = hist.get('Close')
            volume = hist.get('Volume')
            self.closes[symbol] = np.array(close)
            self.volumes[symbol] = np.array(volume)
        logging.info('Attempt to load %d symbols, and %d symbols actually loaded',
                     len(self.symbols), len(self.closes))

    @retrying.retry(stop_max_attempt_number=3, wait_fixed=1000)
    def load_history(self, symbol):
        """Loads history for a single symbol."""
        cache_name = os.path.join(self.cache_path, 'history_%s.csv' % (symbol,))
        if os.path.isfile(cache_name):
            hist = pd.read_csv(cache_name, index_col=0, parse_dates=True)
        else:
            tk = yf.Ticker(symbol)
            if self.period:
                hist = tk.history(period=self.period, interval='1d')
            else:
                hist = tk.history(start=self.start_date, end=self.end_date, interval='1d')
            if len(hist):
                hist.to_csv(cache_name)
            elif self.period:
                raise NotFoundError('History of %s not found' % (symbol,))
            else:
                return
        hist.dropna(inplace=True)
        drop_key = pd.datetime.today().date()
        if self.is_market_open and drop_key in hist.index:
            hist.drop(drop_key, inplace=True)
        if symbol == REFERENCE_SYMBOL or len(hist) == self.history_length:
            self.hists[symbol] = hist
        elif symbol in ('QQQ', 'SPY', '^VIX'):
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

    def get_buy_symbols(self, prices=None, cutoff=None, skip_prediction=False):
        """Gets symbols which trigger buy signals.

        A list of tuples will be returned with symbol, weight and all ML features.
        """
        if not (prices or cutoff) or (prices and cutoff):
            raise Exception('Exactly one of prices or cutoff must be provided')
        quarterly_volatility = {}
        iterator = (tqdm(self.closes.items(), ncols=80, leave=False)
                    if cutoff and sys.stdout.isatty() else self.closes.items())
        buy_info = []
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
                close_year[-DAYS_IN_A_MONTH:], volumes_year[-DAYS_IN_A_MONTH:]))
            # Enough trading volume
            if avg_trading_volume < VOLUME_FILTER_THRESHOLD:
                continue
            # Unable to get realtime price
            if prices and symbol not in prices:
                continue
            if prices:
                price = prices[symbol]
            else:
                price = close[cutoff]
            # 3-day up or down
            if not (price < close_year[-1] < close_year[-2] < close_year[-3] or
                    price > close_year[-1] > close_year[-2] > close_year[-3] > close_year[-4]):
                continue
            # Enough volatility
            returns = [np.log(close_year[i] / close_year[i - 5])
                       for i in range(5, len(close_year))]
            mean = np.mean(returns)
            std = np.std(returns)
            five_day_return = np.log(price / close_year[-5])
            if np.abs(five_day_return - mean) < std:
                continue
            buy_info.append(symbol)

        buy_symbols, ml_features, X = [], [], []
        for symbol in buy_info:
            ml_feature = self.get_ml_feature(symbol, prices=prices, cutoff=cutoff)
            x = [ml_feature[key] for key in ML_FEATURES]
            ml_features.append(ml_feature)
            X.append(x)
        if buy_info:
            X = np.array(X)
            if skip_prediction:
                weights = [1] * len(X)
            else:
                weights = self.model.predict(X)
            buy_symbols = list(zip(buy_info, weights, ml_features))
        return buy_symbols

    def get_trading_list(self, buy_symbols=None, **kwargs):
        """Gets a list of symbols with trading information."""
        if buy_symbols is None:
            buy_symbols = self.get_buy_symbols(**kwargs)
        trading_info = []
        for symbol, classification, _ in buy_symbols:
            trading_info.append((symbol, classification, 'long'))
        trading_info.sort(key=lambda s: s[1], reverse=True)
        n_symbols = min(MAX_STOCK_PICK, len(trading_info))
        trading_list = []
        for i in range(len(trading_info)):
            symbol, weight, side = trading_info[i]
            proportion = min(1 / n_symbols, MAX_PROPORTION) if i < n_symbols else 0
            trading_list.append((symbol, proportion, weight, side))
        return trading_list

    def get_ml_feature(self, symbol, prices=None, cutoff=None):
        feature = {}
        if prices:
            price = prices.get(symbol, 1E10)
            vix = prices['^VIX']
        else:
            price = self.closes[symbol][cutoff]
            vix = self.closes['^VIX'][cutoff]

        if cutoff:
            close = self.closes[symbol][cutoff - DAYS_IN_A_YEAR:cutoff]
            volume = self.volumes[symbol][cutoff - DAYS_IN_A_YEAR:cutoff]
        else:
            close = self.closes[symbol][-DAYS_IN_A_YEAR:]
            volume = self.volumes[symbol][-DAYS_IN_A_YEAR:]
        close = np.append(close, price)

        # Log returns
        feature['Day_1_Return'] = np.log(close[-1] / close[-2])
        feature['Day_2_Return'] = np.log(close[-2] / close[-3])
        feature['Day_3_Return'] = np.log(close[-3] / close[-4])
        feature['Weekly_Return'] = np.log(price / close[-DAYS_IN_A_WEEK])
        feature['Monthly_Return'] = np.log(price / close[-DAYS_IN_A_MONTH])
        feature['Quarterly_Return'] = np.log(price / close[-DAYS_IN_A_QUARTER])
        feature['From_Weekly_High'] = np.log(price / np.max(close[-DAYS_IN_A_WEEK:]))
        feature['From_Weekly_Low'] = np.log(price / np.min(close[-DAYS_IN_A_WEEK:]))

        # Technical indicators
        pd_close = pd.Series(close)
        feature['RSI'] = momentum.rsi(pd_close).values[-1]
        feature['MACD_Rate'] = trend.macd_diff(pd_close).values[-1] / price
        feature['TSI'] = momentum.tsi(pd_close).values[-1]

        # Markets
        feature['VIX'] = vix

        # Other numerical factors
        # Fit five data points to a second order polynomial
        feature['Acceleration'] = (2 * close[-5] - 1 * close[-4] - 2 * close[-3] -
                                   1 * close[-2] + 2 * close[-1]) / 14
        feature['Momentum'] = (-74 * close[-5] + 23 * close[-4] + 60 * close[-3] +
                               37 * close[-2] - 46 * close[-1]) / 70
        quarterly_returns = [np.log(close[i] / close[i - 1])
                             for i in range(-DAYS_IN_A_QUARTER, -1)]
        monthly_returns = quarterly_returns[-DAYS_IN_A_MONTH:]
        weekly_returns = quarterly_returns[-DAYS_IN_A_WEEK:]
        feature['Monthly_Skewness'] = stats.skew(monthly_returns)
        feature['Monthly_Volatility'] = np.std(monthly_returns)
        feature['Weekly_Skewness'] = stats.skew(weekly_returns)
        feature['Weekly_Volatility'] = np.std(weekly_returns)
        feature['Z_Score'] = (feature['Day_1_Return'] - np.mean(quarterly_returns)) / np.std(quarterly_returns)
        feature['Monthly_Avg_Dollar_Volume'] = np.average(np.multiply(
            close[-DAYS_IN_A_MONTH - 1:-1], volume[-DAYS_IN_A_MONTH:])) / 1E6

        return feature

    @functools.lru_cache(maxsize=10000)
    def get_threshold(self, symbol, cutoff=None):
        """Gets threshold for a symbol."""
        if cutoff:
            close_year = self.closes[symbol][cutoff - DAYS_IN_A_YEAR:cutoff]
        else:
            close_year = self.closes[symbol][-DAYS_IN_A_YEAR:]
        down_percent = [close_year[i] / np.max(close_year[i - 5:i]) - 1
                        for i in range(5, len(close_year))
                        if close_year[i] < np.max(close_year[i - 5:i])]
        if not down_percent:
            return 0
        threshold = np.mean(down_percent) - 2.5 * np.std(down_percent)
        return threshold

    def get_volatility(self, symbol, look_back, cutoff=None):
        """Gets threshold for a symbol."""
        if cutoff:
            close = self.closes[symbol][cutoff - look_back:cutoff]
        else:
            close = self.closes[symbol][-look_back:]
        returns = [np.log(close[i] / close[i - 1])
                   for i in range(1, len(close))]
        return np.std(returns) if returns else 0


def get_business_day(offset):
    day = pd.datetime.today() - pd.tseries.offsets.BDay(offset)
    return '%4d-%02d-%02d' % (day.year, day.month, day.day)


def get_header(title):
    header_left = '== [ %s ] ' % (title,)
    return header_left + '=' * (80 - len(header_left))


@retrying.retry(stop_max_attempt_number=3, wait_fixed=1000,
                retry_on_exception=lambda e: isinstance(e, NetworkError))
def web_scraping(url, prefixes):
    """Scrapes a webpage for stock price."""
    try:
        r = requests.get(url, timeout=5)
    except requests.exceptions.RequestException as e:
        raise NetworkError('[%s] %s' % (url, e))
    if r.status_code != 200:
        raise NetworkError('[%s] status %d' % (url, r.status_code))
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
        raise NotFoundError('[%s] %s not found' % (url, prefixes))


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
