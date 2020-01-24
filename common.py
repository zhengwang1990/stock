import copy
import datetime
import json
import os
import re
import requests
import retrying
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
DATA_DIR = 'data'
OUTPUTS_DIR = 'outputs'
MODELS_DIR = 'models'
MAX_HISTORY_LOAD = '5y'
MAX_STOCK_PICK = 3
GARBAGE_FILTER_THRESHOLD = 0.5
VOLUME_FILTER_THRESHOLD = 100000
MAX_THREADS = 5
# These stocks are de-listed
EXCLUSIONS = ('ACTTW', 'ALACW', 'BNTCW', 'CBO', 'CBX', 'CTEST', 'FTACW', 'IBO', 'TACOW', 'ZNWAA', 'ZTEST')
ML_FEATURES = ['Average_Return', 'Threshold',
               'Average_Return_Day_Rank', 'Average_Return_Top_Three',
               'Down_Percent_Day_Rank', 'Down_Percent_Top_Three',
               'Today_Change', 'Yesterday_Change',
               'Day_Range_Change', 'Threshold_Diff', 'Threshold_Quotient',
               'Price', 'Change_Average', 'Change_Variance',
               'Price_Year_Max', 'Price_Year_Min', 'RSI',
               'Price_Average_12', 'Price_Average_26', 'MACD']


def get_time_now():
    tz = pytz.timezone('America/New_York')
    dt_now = datetime.datetime.now(tz)
    time_now = dt_now.hour + dt_now.minute / 60
    return time_now


def get_series(ticker, period='1y'):
    """Gets close prices of a stock symbol as 1D numpy array."""
    dir_path = os.path.dirname(os.path.realpath(__file__))
    os.makedirs(os.path.join(dir_path, CACHE_DIR, get_business_day(1)), exist_ok=True)
    cache_name = os.path.join(dir_path, CACHE_DIR, get_business_day(1), 'series-%s.csv' % (ticker,))
    if os.path.isfile(cache_name):
        df = pd.read_csv(cache_name, index_col=0, parse_dates=True)
        series = df.get('Close')
    else:
        tk = yf.Ticker(ticker)
        hist = tk.history(period=period, interval='1d')
        series = hist.get('Close')
        hist.to_csv(cache_name, header=True)
    if 9.5 < get_time_now() < 16:
        drop_key = datetime.datetime.today().date()
        if drop_key in series.index:
            series = series.drop(drop_key)
    return ticker, series


def get_picked_points(series):
    """Gets threshold for best return of a series.

    This function uses full information of the sereis without truncation.
    """
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


def get_all_symbols():
    res = []
    dir_path = os.path.dirname(os.path.realpath(__file__))
    for f in os.listdir(os.path.join(dir_path, DATA_DIR)):
        if f.endswith('csv'):
            df = pd.read_csv(os.path.join(dir_path, DATA_DIR, f))
            res.extend([row.Symbol for row in df.itertuples()
                        if re.match('^[A-Z]*$', row.Symbol) and
                        row.Symbol not in EXCLUSIONS])
    return res


def get_series_length(period):
    series = get_series(REFERENCE_SYMBOL, period=period)[1]
    return len(series)


def get_series_dates(period):
    series = get_series(REFERENCE_SYMBOL, period=period)[1]
    return series.index


@retrying.retry(stop_max_attempt_number=10, wait_fixed=1000*60*10)
def get_all_series(period):
    """Returns stock price history of all symbols.

    Retyies every 10 min."""
    tickers = get_all_symbols()
    series_length = get_series_length(period)
    all_series = {}
    pool = futures.ThreadPoolExecutor(max_workers=MAX_THREADS)
    print('Loading stock histories...')
    threads = []
    for ticker in tickers:
        t = pool.submit(get_series, ticker, period)
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
    print('%d / %d stock symbols remaining after garbage filter' % (len(res), len(all_series)))
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
    print('%d / %d stock symbols remaining after volume filter' % (len(res), len(all_series)))
    return res


def get_business_day(offset):
    day = pd.datetime.today() - pd.tseries.offsets.BDay(offset) if offset else pd.datetime.today()
    return '%4d-%02d-%02d' % (day.year, day.month, day.day)


def get_header(title):
    header_left = '== [ %s ] ' % (title,)
    return header_left + '=' * (80 - len(header_left))


def get_buy_symbols(all_series, prices, cutoff=None, model=None):
    buy_infos = []
    for ticker, series in tqdm(all_series.items(), ncols=80, leave=False, file=sys.stdout):
        if not cutoff:
            series_year = series[-LOOK_BACK_DAY:]
        else:
            series_year = series[cutoff - LOOK_BACK_DAY:cutoff]
        price = prices.get(ticker, 1E10)
        if price > series_year[-1]:
            continue
        _, avg_return, threshold = get_picked_points(series_year)
        day_range_max = np.max(series_year[-DATE_RANGE:])
        down_percent = (day_range_max - price) / day_range_max
        if down_percent > threshold > 0:
            buy_infos.append((ticker, avg_return, down_percent))

    avg_return_ranking = {tuple[0]: rank + 1
                          for rank, tuple
                          in enumerate(sorted(buy_infos, key=lambda s: s[1], reverse=True))}
    down_percent_ranking = {tuple[0]: rank + 1
                            for rank, tuple
                            in enumerate(sorted(buy_infos, key=lambda s: s[2], reverse=True))}

    buy_symbols = []
    for tuple in buy_infos:
        ticker = tuple[0]
        if not cutoff:
            series_year = all_series[ticker][-LOOK_BACK_DAY:]
        else:
            series_year = all_series[ticker][cutoff - LOOK_BACK_DAY:cutoff]
        price = prices[ticker]
        rankings = {'Average_Return': avg_return_ranking[ticker],
                    'Down_Percent': down_percent_ranking[ticker]}
        ml_feature = get_ml_feature(series_year, price, rankings)
        if model:
            x = [ml_feature[key] for key in ML_FEATURES]
            # 90% boundary
            weight = model.predict(np.array([x]))[0] + 0.025
        else:
            weight = tuple[1]
        buy_symbols.append((ticker, weight, ml_feature))
    return buy_symbols


def get_ml_feature(series, price, rankings):
    _, avg_return, threshold = get_picked_points(series)
    avg_return *= 100
    threshold *= 100
    day_range_max = np.max(series[-DATE_RANGE:])
    day_range_change = (day_range_max - price) / day_range_max * 100
    today_change = (price - series[-1]) / series[-1] * 100
    yesterday_change = (series[-1] - series[-2]) / series[-2] * 100
    all_changes = ((series[1:] - series[:-1]) / series[:-1]) * 100
    rsi = get_rsi(series)
    avg_return_day_rank = rankings['Average_Return']
    down_percent_day_rank = rankings['Down_Percent']
    price_average_12 = np.average(series[-12:])
    price_average_26 = np.average(series[-26:])
    feature = {'Average_Return': avg_return,
               'Threshold': threshold,
               'Yesterday_Change': yesterday_change,
               'Today_Change': today_change,
               'Day_Range_Change': day_range_change,
               'Threshold_Diff': day_range_change - threshold,
               'Threshold_Quotient': day_range_change / threshold,
               'Price': price,
               'Change_Average': np.mean(all_changes),
               'Change_Variance': np.var(all_changes),
               'Price_Year_Max': np.max(series),
               'Price_Year_Min': np.min(series),
               'RSI': rsi,
               'Average_Return_Day_Rank': avg_return_day_rank,
               'Average_Return_Top_Three': int(avg_return_day_rank <= 3),
               'Down_Percent_Day_Rank': down_percent_day_rank,
               'Down_Percent_Top_Three': int(down_percent_day_rank <= 3),
               'Price_Average_12': price_average_12,
               'Price_Average_26': price_average_26,
               'MACD': price_average_12 - price_average_26}
    return feature


def get_trading_list(buy_symbols):
    buy_symbols.sort(key=lambda s: s[1], reverse=True)
    n_symbols = 0
    while n_symbols < min(MAX_STOCK_PICK, len(buy_symbols)) and buy_symbols[n_symbols][1] > 0:
        n_symbols += 1
    ac = 0
    for i in range(n_symbols):
        ac += buy_symbols[i][1]
    trading_list = []
    common_share = 0.75
    for i in range(len(buy_symbols)):
        ticker = buy_symbols[i][0]
        weight = buy_symbols[i][1]
        if i < n_symbols:
            proportion = common_share / n_symbols + (1 - common_share) * weight / ac
        else:
            proportion = 0
        trading_list.append((ticker, proportion, weight))
    return trading_list


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


def get_rsi(series, n=14):
    prices = series[-n:]
    delta = np.diff(prices)
    up = np.zeros_like(delta)
    down = np.zeros_like(delta)
    up[delta > 0] = delta[delta > 0]
    down[delta < 0] = -delta[delta < 0]
    avg_up = np.mean(up)
    avg_down = np.mean(down)
    if avg_down == 0:
        rsi = 50.0 if avg_up == 0 else 100
    else:
        rs = avg_up / avg_down
        rsi = 100 - 100 / (1 + rs)
    return rsi
