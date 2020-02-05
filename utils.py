import datetime
import os
import re
import requests
import retrying
import ta
import numpy as np
import pandas as pd
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
REGRESSION_COEFFICIENT = {
    'Today_Change': 0.000000e+00,
    'Yesterday_Change': -1.990495e+00,
    'Day_Before_Yesterday_Change': -6.625245e-01,
    'Twenty_Day_Change': -1.457217e-01,
    'Day_Range_Change': -0.000000e+00,
    'Year_High_Change': 7.263699e-01,
    'Year_Low_Change': 1.085623e-02,
    'Change_Average': 0.000000e+00,
    'Change_Variance': -0.000000e+00,
    'RSI': -1.215447e-02,
    'MACD_Rate': -1.421378e+01,
    'TSI': 0.000000e+00,
    'WR': 2.405147e-02,
}
REGRESSION_INTERCEPT = 2.72789864441329
ML_FEATURES = list(REGRESSION_COEFFICIENT.keys())
ALPACA_API_BASE_URL = "https://api.alpaca.markets"


class TradingBase(object):
    """Basic trade utils."""

    def __init__(self, alpaca, period=DEFAULT_HISTORY_LOAD):
        self.alpaca = alpaca
        self.pool = futures.ThreadPoolExecutor(max_workers=MAX_THREADS)
        self.root_dir = os.path.dirname(os.path.realpath(__file__))
        self.cache_path = os.path.join(self.root_dir, CACHE_DIR,
                                       get_business_day(1), period)
        os.makedirs(self.cache_path, exist_ok=True)
        self.hists = {}
        self.history_length = self.get_history_length(period)
        self.history_dates = self.get_history_dates(period)
        self._load_all_symbols()
        self._load_histories(period)
        self._read_series_from_histories(period)

    def _load_all_symbols(self):
        self.symbols = []
        assets = self.alpaca.list_assets()
        self.symbols = [asset.symbol for asset in assets
                        if re.match('^[A-Z]*$', asset.symbol)
                        and asset.symbol not in EXCLUSIONS]

    @retrying.retry(stop_max_attempt_number=10, wait_fixed=1000 * 60 * 10)
    def _load_histories(self, period):
        """Loads history of all stock symbols."""
        print('Loading stock histories...')
        threads = []
        # Allow at most 10 errors
        error_tol = 10
        for symbol in self.symbols:
            t = self.pool.submit(self._load_history, symbol, period)
            threads.append(t)
        for t in tqdm(threads, ncols=80):
            try:
                t.result()
            except Exception as e:
                print(e)
                error_tol -= 1
                if error_tol <= 0:
                    raise e

    def _read_series_from_histories(self, period):
        """Reads out close price and volume."""
        self.closes = {}
        self.volumes = {}
        clock = self.alpaca.get_clock()
        for symbol, hist in self.hists.items():
            close = hist.get('Close')
            volume = hist.get('Volume')
            if clock.is_open:
                drop_key = datetime.datetime.today().date()
                close = close.drop(drop_key)
                volume = volume.drop(drop_key)
            self.closes[symbol] = np.array(close)
            self.volumes[symbol] = np.array(volume)
        print('Attempt to load %d symbols' % (len(self.symbols)))
        print('%d loaded symbols after drop symbols traded less than %s' % (
            len(self.closes), period))

    @retrying.retry(stop_max_attempt_number=2, wait_fixed=500)
    def _load_history(self, symbol, period):
        """Loads history for a single symbol."""
        cache_name = os.path.join(self.cache_path, 'history_%s.csv' % (symbol,))
        if os.path.isfile(cache_name):
            hist = pd.read_csv(cache_name, index_col=0, parse_dates=True)
        else:
            try:
                tk = yf.Ticker(symbol)
                hist = tk.history(period=period, interval='1d')
                if len(hist):
                    hist.to_csv(cache_name, header=True)
            except Exception as e:
                print('Can not get history of %s: %s' % (symbol, e))
                raise e
        if symbol == REFERENCE_SYMBOL or len(hist) == self.history_length:
            self.hists[symbol] = hist

    def get_history_length(self, period):
        """Get the number of trading days in a given period."""
        self._load_history(REFERENCE_SYMBOL, period=period)
        return len(self.hists[REFERENCE_SYMBOL])

    def get_history_dates(self, period):
        """Gets the list trading dates in a given period."""
        self._load_history(REFERENCE_SYMBOL, period=period)
        return self.hists[REFERENCE_SYMBOL].index

    def get_buy_symbols(self, prices=None, cutoff=None):
        """Gets symbols which trigger buy signals.

        A list of tuples will be returned with symbol, weight and all ML features.
        """
        if not (prices or cutoff) or (prices and cutoff):
            raise Exception('Exactly one of prices or cutoff must be provided')
        buy_info = []
        for symbol, close in tqdm(self.closes.items(), ncols=80):
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
            threshold = get_threshold(close_year)
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
            weight = REGRESSION_INTERCEPT
            for key in ML_FEATURES:
                weight += REGRESSION_COEFFICIENT.get(key, 0) * ml_feature.get(key, 0)
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
        else:
            price = self.closes[symbol][cutoff]

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
                   'TSI': tsi}
        return feature


def get_threshold(series):
    """Gets threshold for a series.
    """
    down_percent = [series[i] / np.max(series[i - DATE_RANGE:i]) - 1
                    for i in range(DATE_RANGE, len(series))
                    if series[i] < np.max(series[i - DATE_RANGE:i])]
    threshold = np.mean(down_percent) - 2.5 * np.std(down_percent)
    return threshold


def get_picked_points(series):
    """Gets threshold for best return of a series.

    This function uses full information of the series without truncation.
    """
    down_t = np.array([i + 1 for i in range(DATE_RANGE - 1, len(series) - 1)
                       if series[i] >= series[i + 1] > 0])
    down_percent = [(np.max(series[i - DATE_RANGE:i]) - series[i]) /
                    np.max(series[i - DATE_RANGE:i]) for i in down_t]
    pick_ti = sorted(range(len(down_percent)), key=lambda i: down_percent[i],
                     reverse=True)
    tol = 50
    max_avg_gain = np.finfo(float).min
    n_pick = 0
    for n_point in range(1, len(pick_ti)):
        potential_t = down_t[pick_ti[:n_point]]
        gains = [(series[t + 1] - series[t]) / series[t]
                 for t in potential_t if t + 1 < len(series)]
        if len(gains) > 0:
            avg_gain = np.average(gains)
            if avg_gain > max_avg_gain:
                n_pick = n_point
                max_avg_gain = avg_gain
            else:
                tol -= 1
                if not tol:
                    break
    if not n_pick:
        return [], np.finfo(float).min, np.finfo(float).max
    threshold_i = pick_ti[n_pick - 1]
    threshold = down_percent[threshold_i]
    pick_t = down_t[pick_ti[:n_pick]]
    return pick_t, max_avg_gain, threshold


def get_business_day(offset):
    day = pd.datetime.today() - pd.tseries.offsets.BDay(offset) if offset else pd.datetime.today()
    return '%4d-%02d-%02d' % (day.year, day.month, day.day)


def get_header(title):
    header_left = '== [ %s ] ' % (title,)
    return header_left + '=' * (80 - len(header_left))


@retrying.retry(stop_max_attempt_number=3, wait_fixed=500)
def web_scraping(url, prefixes):
    r = requests.get(url, timeout=3)
    c = str(r.content)
    pos = -1
    for prefix in prefixes:
        pos = c.find(prefix)
        if pos >= 0:
            break
    if pos >= 0:
        s = ''
        while c[pos] > '9' or c[pos] < '0':
            pos += 1
        while '9' >= c[pos] >= '0' or c[pos] == '.':
            s += c[pos]
            pos += 1
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
