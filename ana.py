import yfinance as yf
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import pandas as pd
import os
import re
from tqdm import tqdm
import json
from tabulate import tabulate

DATE_RANGE = 5


def getSeries(ticker, time='1y'):
  tk = yf.Ticker(ticker)
  hist = tk.history(period=time, interval='1d')
  series = [p.Close for p in hist.itertuples()]
  return np.array(series)

def getPickedPoints(series):
  down_t = []
  for i in range(DATE_RANGE-1, len(series)-1):
    if series[i] >= series[i+1]:
      down_t.append(i+1)
  down_t = np.array(down_t)
  down_percent = [(np.max(series[i-DATE_RANGE:i]) - series[i]) / np.max(series[i-DATE_RANGE:i]) for i in down_t]
  zscores = stats.zscore(down_percent)
  pick_ti = sorted(range(len(down_percent)), key=lambda i:zscores[i], reverse=True)
  avg_gains = []
  for n_point in range(1, len(pick_ti)):
    potential_t = down_t[pick_ti[:n_point]]
    gains = [(series[t+1] - series[t]) / series[t] for t in potential_t if t + 1 < len(series)]
    if len(gains) > 0:
      avg_gains.append((np.average(gains), n_point))
  avg_gains.sort(reverse=True)
  avg_return = avg_gains[0][0]
  n_pick = avg_gains[0][1]
  threshold_i = pick_ti[n_pick-1]
  threshold = down_percent[threshold_i]
  pick_t = down_t[pick_ti[:n_pick]]
  return pick_t, avg_return, threshold

def plotBuyPoints(ticker):
  series = getSeries(ticker)
  pick_t, avg_return, _ = getPickedPoints(series)
  print('Average return of %s: %.2f%%'%(ticker, avg_return*100))
  t = range(len(series))
  plt.plot(t, series)
  plt.plot(pick_t, series[pick_t], 'o')
  plt.show()

def getBuySignal(series, price):
  if price >= series[-1]:
    return 0, False
  _, avg_return, threshold = getPickedPoints(series)
  down_percent = (np.max(series[-DATE_RANGE:]) - price) / np.max(series[-DATE_RANGE:])
  return avg_return, down_percent > threshold

def getStaticBuySymbols(tickers, f=None):
  sl = getSeriesLength('1y')
  for ticker in tickers:
    series = getSeries(ticker, time='1y')
    if len(series) != sl:
      continue
    avg_return, is_buy = getBuySignal(series[:-1], series[-1])
    print('%s: avg return %.2f%% buy %s'%(ticker, avg_return*100, is_buy))
    if is_buy:
      f.write('%s, %.2f%%\n'%(ticker, avg_return*100))
      f.flush()

def getAllSymbols():
  res = []
  for f in os.listdir('csv'):
    df = pd.read_csv(os.path.join('csv', f))
    res.extend([row.Symbol for row in df.itertuples() if re.match('^[A-Z]*$', row.Symbol)])
  return res

def simulate():
  tickers = getAllSymbols()
  sl = getSeriesLength('5y')
  dates = getSeriesDates('5y')
  serieses = {}
  print('Loading stock histories...')
  for ticker in tqdm(tickers, bar_format='{percentage:3.0f}%|{bar:100}{r_bar}'):
    cache_name = 'json/%s.json' % (ticker,)
    if os.path.isfile(cache_name):
      with open(cache_name) as f:
        series_json = f.read()
      series = np.array(json.loads(series_json))
    else:
      series = getSeries(ticker, time='5y')
      series_json = json.dumps(series.tolist())
      with open(cache_name, 'w') as f:
        f.write(series_json)
    if len(series) != sl:
      continue
    serieses[ticker] = series

  to_del = []
  for ticker, series in serieses.items():
    if np.max(series) * 0.7 > series[-1]:
      to_del.append(ticker)
  for ticker in to_del:
    del serieses[ticker]
  print('Picked stocks: %d'%(len(serieses)))

  total_return = 1.0
  start_point = 0
  while '2018' not in dates[start_point]:
    start_point += 1
  for cutoff in range(start_point-1, sl-1):
    print('='*100)
    print('DATE:', dates[cutoff+1])
    buy_symbols = []
    for ticker, series in tqdm(serieses.items(), bar_format='{percentage:3.0f}%|{bar:100}{r_bar}', leave=False):
      avg_return, is_buy = getBuySignal(series[:cutoff], series[cutoff])
      if is_buy:
        buy_symbols.append((avg_return, ticker))
    buy_symbols.sort(reverse=True)
    max_symbol = min(len(buy_symbols), 10)
    while max_symbol > 0 and buy_symbols[max_symbol-1][0] < 0.01:
      max_symbol -= 1
    ac = 0
    for i in range(max_symbol):
      ac += buy_symbols[i][0]
    day_gain = 0
    trading_table = []
    for i in range(max_symbol):
      portion = 0.75 / max_symbol + 0.25 * buy_symbols[i][0] / ac
      ticker = buy_symbols[i][1]
      series = serieses[ticker]
      gain = (series[cutoff + 1] - series[cutoff]) / series[cutoff]
      trading_table.append([ticker, '%.2f%%'%(portion*100,), '%.2f%%'%(gain*100,)])
      day_gain += gain * portion
    print(tabulate(trading_table, headers=['Symbol', 'Portion', 'Gain'], tablefmt="grid"))
    print('DAILY GAIN: %.2f%%'%(day_gain*100,))
    total_return *= (1 + day_gain)
    print('TOTAL GAIN: %.2f%%'%((total_return-1)*100,))

def getSeriesLength(time):
  series = getSeries('AAPL', time=time)
  return len(series)

def getSeriesDates(time):
  tk = yf.Ticker('AAPL')
  hist = tk.history(period=time, interval='1d')
  dates = []
  for p in hist.itertuples():
    dates.append('%s-%s-%s' % (p.Index.year, p.Index.month, p.Index.day))
  return dates

def main():
  simulate()
  #with open('output.csv', 'w') as f:
  #  symbols = getAllSymbols()
  #  getStaticBuySymbols(symbols, f)

if __name__ == '__main__':
  main()
