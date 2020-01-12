import os
import re
import numpy as np
import pandas as pd
import yfinance as yf

DATE_RANGE = 5
REFERENCE_SYMBOL = 'AAPL'
LOOK_BACK_DAY = 250


def get_series(ticker, time='1y'):
    """Gets close prices of a stock symbol as 1D numpy array."""
    tk = yf.Ticker(ticker)
    hist = tk.history(period=time, interval='1d')
    series = [p.Close for p in hist.itertuples()]
    return np.array(series)


def get_picked_points(series):
    down_t = np.array([i + 1 for i in range(DATE_RANGE - 1, len(series) - 1) if series[i] >= series[i + 1]])
    down_percent = [(np.max(series[i - DATE_RANGE:i]) - series[i]) / np.max(series[i - DATE_RANGE:i]) for i in down_t]
    pick_ti = sorted(range(len(down_percent)), key=lambda i: down_percent[i], reverse=True)
    tol = 50
    max_avg_gain = 0
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
    dates = ['%s-%s-%s' % (p.Index.year, p.Index.month, p.Index.day) for p in hist.itertuples()]
    return dates
