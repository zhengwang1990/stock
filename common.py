import copy
import datetime
import json
import os
import re
import requests
import sys
import numpy as np
import pandas as pd
import pytz
import yfinance as yf
from concurrent import futures
from tqdm import tqdm

DATE_RANGE = 5
REFERENCE_SYMBOL = 'AAPL'
LOOK_BACK_DAY = 250
CACHE_DIR = 'cache'
MAX_HISTORY_LOAD = '5y'
MAX_STOCK_PICK = 3
GARBAGE_FILTER_THRESHOLD = 0.5
VOLUME_FILTER_THRESHOLD = 10000
MAX_THREADS = 5
# These stocks are de-listed
EXCLUSIONS = ('IBO', 'ZTEST', 'ZNWAA', 'CBO', 'CBX', 'CTEST')


def get_time_now():
    tz = pytz.timezone('America/New_York')
    dt_now = datetime.datetime.now(tz)
    time_now = dt_now.hour + dt_now.minute / 60
    return time_now


def get_series(ticker, time='1y'):
    """Gets close prices of a stock symbol as 1D numpy array."""
    dir_path = os.path.dirname(os.path.realpath(__file__))
    os.makedirs(os.path.join(dir_path, CACHE_DIR, get_business_day(1)), exist_ok=True)
    cache_name = os.path.join(dir_path, CACHE_DIR, get_business_day(1), 'series-%s.csv' % (ticker,))
    if os.path.isfile(cache_name):
        df = pd.read_csv(cache_name, index_col=0, parse_dates=True)
        series = df.get('Close')
    else:
        tk = yf.Ticker(ticker)
        hist = tk.history(period=time, interval='1d')
        series = hist.get('Close')
        if 9.5 < get_time_now() < 16:
          drop_key = datetime.datetime.today().date()
          if drop_key in series.index:
            series = series.drop(drop_key)
        series.to_csv(cache_name, header=True)
    return ticker, series


def get_picked_points(series):
    down_t = np.array([i + 1 for i in range(DATE_RANGE - 1, len(series) - 1) if series[i] >= series[i + 1]])
    down_percent = [(np.max(series[i - DATE_RANGE:i]) - series[i]) / np.max(series[i - DATE_RANGE:i]) for i in down_t]
    pick_ti = sorted(range(len(down_percent)), key=lambda i: down_percent[i], reverse=True)
    tol = 50
    max_avg_gain = np.finfo(float).min
    n_pick = 0
    for n_point in range(1, len(pick_ti)):
        potential_t = down_t[pick_ti[:n_point]]
        gains = [(series[t + 1] - series[t]) / series[t] for t in potential_t if t + 1 < len(series)]
        if len(gains) > 0:
            avg_gain = np.average(gains)
            if avg_gain > max_avg_gain:
                n_pick = n_point
                max_avg_gain = avg_gain
            else:
                tol -= 1
            if not tol:
                break
    threshold_i = pick_ti[n_pick - 1]
    threshold = down_percent[threshold_i]
    pick_t = down_t[pick_ti[:n_pick]]
    return pick_t, max_avg_gain, threshold


def get_buy_signal(series, price):
    if price >= series[-1]:
        return 0, False
    _, avg_return, threshold = get_picked_points(series)
    down_percent = (np.max(series[-DATE_RANGE:]) - price) / np.max(series[-DATE_RANGE:])
    return avg_return, down_percent > threshold


def get_all_symbols():
    res = []
    dir_path = os.path.dirname(os.path.realpath(__file__))
    for f in os.listdir(os.path.join(dir_path, 'data')):
        if f.endswith('csv'):
            df = pd.read_csv(os.path.join('data', f))
            res.extend([row.Symbol for row in df.itertuples()
                        if re.match('^[A-Z]*$', row.Symbol) and
                        row.Symbol not in EXCLUSIONS])
    return res


def get_series_length(time):
    series = get_series(REFERENCE_SYMBOL, time=time)[1]
    return len(series)


def get_series_dates(time):
    tk = yf.Ticker(REFERENCE_SYMBOL)
    hist = tk.history(period=time, interval='1d')
    dates = [p.Index for p in hist.itertuples()]
    return dates


def get_all_series(time):
    """Returns stock price history of all symbols."""
    tickers = get_all_symbols()
    series_length = get_series_length(time)
    all_series = {}
    pool = futures.ThreadPoolExecutor(max_workers=MAX_THREADS)
    print('Loading stock histories...')
    threads = []
    for ticker in tickers:
        t = pool.submit(get_series, ticker, time)
        threads.append(t)
    for t in tqdm(threads, ncols=80, file=sys.stdout):
        ticker, series = t.result()
        if len(series) != series_length:
            continue
        all_series[ticker] = np.array(series)
    pool.shutdown()
    return all_series


def filter_garbage_series(all_series):
    res = {}
    for ticker, series in all_series.items():
        if np.max(series) * GARBAGE_FILTER_THRESHOLD <= series[-1]:
            res[ticker] = series
    return res


def filter_low_volume_series(all_series):
    dir_path = os.path.dirname(os.path.realpath(__file__))
    volume_cache_file = os.path.join(dir_path, 'data', 'volumes.json')
    with open(volume_cache_file) as f:
        volumes = json.loads(f.read())
    res = {}
    overwrite = False
    for ticker, series in all_series.items():
        volume = volumes.get(ticker, VOLUME_FILTER_THRESHOLD)
        if volume >= VOLUME_FILTER_THRESHOLD:
            res[ticker] = series
    if overwrite:
        with open(volume_cache_file, 'w') as f:
            f.write(json.dumps(volumes))
    return res


def get_business_day(offset):
    day = pd.datetime.today() - pd.tseries.offsets.BDay(offset)
    return '%4d-%02d-%02d' % (day.year, day.month, day.day)


def get_header(title):
    header_left = '== [ %s ] ' % (title,)
    return header_left + '=' * (80 - len(header_left))


def get_buy_symbols(all_series, cutoff):
    buy_symbols = []
    for ticker, series in tqdm(all_series.items(), ncols=80, leave=False, file=sys.stdout):
        avg_return, is_buy = get_buy_signal(series[cutoff - LOOK_BACK_DAY:cutoff], series[cutoff])
        if is_buy:
            buy_symbols.append((avg_return, ticker))
    return buy_symbols


def get_trading_list(buy_symbols):
    buy_symbols.sort(reverse=True)
    n_symbols = 0
    while n_symbols < min(MAX_STOCK_PICK, len(buy_symbols)) and buy_symbols[n_symbols][0] >= 0.01:
        n_symbols += 1
    ac = 0
    for i in range(n_symbols):
        ac += buy_symbols[i][0]
    trading_list = []
    for i in range(n_symbols):
        proportion = 0.75 / n_symbols + 0.25 * buy_symbols[i][0] / ac
        ticker = buy_symbols[i][1]
        trading_list.append((ticker, proportion))
    return trading_list


def web_scraping(url, prefixes):
    r = requests.get(url, timeout=10)
    if not r.ok:
        # retry once
        r = requests.get(url, timeout=10)
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
        raise Exception('%s not found in %s' % (prefixes, url))


def bi_print(message, output_file):
    """Prints to both stdout and a file."""
    print(message)
    if output_file:
        output_file.write(message)
        output_file.write('\n')
        output_file.flush()
