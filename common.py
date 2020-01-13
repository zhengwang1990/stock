import json
import os
import re
import sys
import numpy as np
import pandas as pd
import yfinance as yf
from tqdm import tqdm

DATE_RANGE = 5
REFERENCE_SYMBOL = 'AAPL'
LOOK_BACK_DAY = 250
CACHE_DIR = 'cache'
MAX_HISTORY_LOAD = '5y'
MAX_STOCK_PICK = 3


def get_series(ticker, time='1y'):
    """Gets close prices of a stock symbol as 1D numpy array."""
    tk = yf.Ticker(ticker)
    hist = tk.history(period=time, interval='1d')
    return hist.get('Close')


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
    for f in os.listdir('data'):
        df = pd.read_csv(os.path.join('data', f))
        res.extend([row.Symbol for row in df.itertuples() if re.match('^[A-Z]*$', row.Symbol)])
    return res


def get_series_length(time):
    series = get_series(REFERENCE_SYMBOL, time=time)
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
    dir_path = os.path.dirname(os.path.realpath(__file__))
    all_series = {}
    print('Loading stock histories...')
    for ticker in tqdm(tickers, ncols=80, bar_format='{percentage:3.0f}%|{bar}{r_bar}', file=sys.stdout):
        cache_name = os.path.join(dir_path, CACHE_DIR, get_prev_business_day(), 'cache-%s.json' % (ticker,))
        if os.path.isfile(cache_name):
            with open(cache_name) as f:
                series_json = f.read()
            series = np.array(json.loads(series_json))
        else:
            series = get_series(ticker, time=time)
            series_json = json.dumps(series.tolist())
            with open(cache_name, 'w') as f:
                f.write(series_json)
        if len(series) != series_length:
            continue
        all_series[ticker] = series
    return all_series


def filter_all_series(all_series):
    to_del = []
    for ticker, series in all_series.items():
        if np.max(series) * 0.7 > series[-1]:
            to_del.append(ticker)
    for ticker in to_del:
        del all_series[ticker]
    print('Num of picked stocks: %d' % (len(all_series)))
    return all_series


def get_prev_business_day():
    day = pd.datetime.today() - pd.tseries.offsets.BDay(1)
    return '%4d-%02d-%02d' % (day.year, day.month, day.day)


def get_header(title):
    header_left = '== [ %s ] ' % (title,)
    return header_left + '=' * (80 - len(header_left))


def get_buy_symbols(all_series, cutoff):
    buy_symbols = []
    for ticker, series in tqdm(all_series.items(), ncols=80, bar_format='{percentage:3.0f}%|{bar}{r_bar}',
                               leave=False, file=sys.stdout):
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
