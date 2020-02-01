import datetime
import os
import re
import requests
import retrying
import sys
import numpy as np
import pandas as pd
import yfinance as yf
from concurrent import futures
from tqdm import tqdm

DATE_RANGE = 5
REFERENCE_SYMBOL = 'AAPL'
DAYS_IN_A_YEAR = 250
CACHE_DIR = 'cache'
DATA_DIR = 'data'
OUTPUTS_DIR = 'outputs'
MODELS_DIR = 'models'
DEFAULT_HISTORY_LOAD = '5y'
MAX_STOCK_PICK = 3
VOLUME_FILTER_THRESHOLD = 1000000
MAX_THREADS = 5
# These stocks are de-listed
EXCLUSIONS = ('AABA', 'ACETQ', 'AETI', 'AGC', 'AKP', 'ALDR', 'ALN', 'ALQA',
              'AMBR', 'AMGP', 'AMID', 'AMMA', 'ANDV', 'ANDX', 'ANWWQ', 'AOI',
              'APC', 'APF', 'APHB', 'APU', 'AQ', 'ARLZQ', 'ARRS', 'ARRY',
              'ASCMQ', 'AST', 'ASV', 'ATTU', 'AVHI', 'BEL', 'BHBK', 'BID',
              'BKS', 'BLH', 'BLMT', 'BMS', 'BNCL', 'BNGOU', 'BOJA', 'BOSXF',
              'BPL', 'BRACU', 'BRSS', 'BRSWQ', 'BT', 'BVX', 'BXEFF', 'CADC',
              'CASM', 'CAW', 'CBAK', 'CBK', 'CBLK', 'CCA', 'CCNI', 'CHAC',
              'CHKE', 'CHSP', 'CIVI', 'CJ', 'CLNS', 'CMSS', 'CMSSU', 'CMTA',
              'CRAY', 'CTRL', 'CTRV', 'CTWS', 'CUR', 'CVRS', 'CYTX', 'CZFC',
              'DATA', 'DCUD', 'DDE', 'DDOC', 'DELT', 'DFBH', 'DFRG', 'DHCP',
              'DHVW', 'DNB', 'DOVA', 'DRYS', 'DSW', 'DTRM', 'DTV', 'DWDP',
              'EAGL', 'EAGLU', 'ECYT', 'EDGE', 'EDR', 'EFII', 'EHIC', 'EIV',
              'ELLI', 'EMCI', 'EMES', 'EQGP', 'ESESD', 'ESL', 'EVJ', 'EVLV',
              'EVO', 'EVP', 'FDC', 'FELP', 'FHY', 'FLF', 'FNSR', 'FNTEU',
              'FRAC', 'FRSH', 'FSNN', 'GG', 'GGP', 'GHDX', 'GLAC', 'GLACU',
              'GNBC', 'GOV', 'GPIC', 'GSHT', 'GTYHU', 'HAIR', 'HBK', 'HEB',
              'HF', 'HFBC', 'HIFR', 'HIVE', 'HKRSQ', 'HLTH', 'HMTA', 'HPJ',
              'HYGS', 'IDTI', 'IMDZ', 'IMI', 'IMMY', 'IPOA', 'ISCA', 'ISRL',
              'ITG', 'ITUS', 'JONE', 'JSYNU', 'KAAC', 'KCAP', 'KED', 'KEYW',
              'KONE', 'KOOL', 'KPFS', 'KYE', 'LABL', 'LEXEA', 'LION', 'LLL',
              'LOGO', 'LOXO', 'LTXB', 'LXFT', 'MACQ', 'MAMS', 'MB', 'MBFI',
              'MBNAA', 'MBNAB', 'MBTF', 'MDSO', 'MFCB', 'MMDM', 'MOC', 'MPAC',
              'MRT', 'MSF', 'MSL', 'MTEC', 'MTECU', 'MTGE', 'MXWL', 'MYND',
              'MZF', 'NANO', 'NAO', 'NAVG', 'NCI', 'NCOM', 'NDRO', 'NETS',
              'NITE', 'NNC', 'NRCG', 'NRE', 'NSU', 'NTC', 'NTRI', 'NVMM',
              'NXEO', 'NXEOU', 'NXTM', 'NYLD', 'NYNY', 'OAK', 'OHGI', 'OHRP',
              'OMED', 'ONEW', 'OPHT', 'ORBK', 'ORM', 'ORPN', 'OSIR', 'OSPRU',
              'P', 'PCMI', 'PERY', 'PETX', 'PGLC', 'PHIIQ', 'PHIKQ', 'PLLL',
              'PNTR', 'PRAN', 'PTIE', 'PTXTQ', 'PYDS', 'QCP', 'QSII', 'QTNA',
              'RDC', 'REN', 'RHT', 'RLM', 'RNN', 'ROX', 'RTEC', 'RVEN', 'RXII',
              'SCAC', 'SCACU', 'SFLY', 'SFS', 'SGYPQ', 'SHLM', 'SHOS', 'SHPG',
              'SIFI', 'SIR', 'SKIS', 'SLD', 'SMSH', 'SPA', 'SSFN', 'STLR',
              'STNL', 'STNLU', 'SVU', 'SXCP', 'TAHO', 'TFCF', 'TFCFA', 'TIER',
              'TISA', 'TLP', 'TMCX', 'TMCXU', 'TOWR', 'TPIV', 'TRCO', 'TRK',
              'TRNC', 'TSS', 'TST', 'TYPE', 'UBNK', 'UBNT', 'UCBA', 'ULTI',
              'UQM', 'USG', 'UWN', 'VEACU', 'VICL', 'VMAX', 'VSAR', 'VSM',
              'WAGE', 'WGP', 'WP', 'XGTI', 'XRM', 'XSPL', 'ZDEO', 'ZF', 'ZJBR',
              'CVON', 'FFKT', 'KLXI', 'HNTUF', 'EGLTQ', 'NEWM', 'HCLP', 'HDP',
              'GSTCQ', 'DFBHU', 'ILG', 'MATR', 'RENX', 'VLP', 'TSRO')
ML_FEATURES = ['Average_Return', 'Threshold',
               'Today_Change', 'Yesterday_Change', 'Twenty_Day_Change',
               'Day_Range_Change',
               'Change_Average', 'Change_Variance',
               'RSI',
               'MACD_Rate']
APCA_API_BASE_URL = "https://api.alpaca.markets"


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

  @retrying.retry(stop_max_attempt_number=10, wait_fixed=1000*60*10)
  def _load_histories(self, period):
    """Loads history of all stock symbols."""
    print('Loading stock histories...')
    threads = []
    # Allow at most 10 errors
    error_tol = 10
    for symbol in self.symbols:
      t = self.pool.submit(self._load_history, symbol, period)
      threads.append(t)
    for t in tqdm(threads, ncols=80, file=sys.stdout):
      try:
        t.result()
      except Exception as e:
        print(e)
        tol -= 1
        if tol <= 0:
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
      close = np.array(close)
      volume = np.array(volume)
      avg_trading_volume = np.average(np.multiply(
          volume[-90:], close[-90:]))
      if avg_trading_volume >= VOLUME_FILTER_THRESHOLD:
        self.closes[symbol] = close
        self.volumes[symbol] = volume
    print('Attempt to load %d symbols' % (len(self.symbols)))
    print('%d loaded symbols after drop symbols traded less than %s' % (
        len(self.hists), period))
    print('%d symbols loaded after drop symbols with volume less than $%s' % (
        len(self.closes), VOLUME_FILTER_THRESHOLD))

  def _load_history(self, symbol, period):
    """Loads history for a single symbol."""
    cache_name = os.path.join(self.cache_path, '%s.csv' % (symbol,))
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

  def get_buy_symbols(self, prices=None, cutoff=None, model=None):
    """Gets symbols which trigger buy signals.

    A list of tuples will be returned with symbol, weight and all ML features.
    """
    if not (prices or cutoff):
      raise Exception('Either prices or cutoff must be provided')
    buy_infos = []
    for symbol, series in tqdm(self.closes.items(), ncols=80, leave=False,
                               file=sys.stdout):
      if not cutoff:
        series_year = series[-DAYS_IN_A_YEAR:]
      else:
        series_year = series[cutoff - DAYS_IN_A_YEAR:cutoff]
      if prices:
        price = prices.get(symbol, 1E10)
      else:
        price = series[cutoff]
      if price > series_year[-1]:
        continue
      _, avg_return, threshold = get_picked_points(series_year)
      day_range_max = np.max(series_year[-DATE_RANGE:])
      down_percent = (day_range_max - price) / day_range_max
      if down_percent > threshold > 0:
        buy_infos.append((symbol, avg_return, threshold))

    buy_symbols = []
    for symbol, avg_return, threshold in buy_infos:
      if not cutoff:
        series_year = self.closes[symbol][-DAYS_IN_A_YEAR:]
      else:
        series_year = self.closes[symbol][cutoff - DAYS_IN_A_YEAR:cutoff]
      if prices:
        if symbol not in prices:
          continue
        price = prices[symbol]
      else:
        price = self.closes[symbol][cutoff]
      ml_feature = get_ml_feature(series_year, price, avg_return, threshold)
      if model:
        x = [ml_feature[key] for key in ML_FEATURES]
        weight = model.predict(np.array([x]))[0]
      else:
        weight = avg_return
      buy_symbols.append((symbol, weight, ml_feature))
    return buy_symbols

  def get_trading_list(self, buy_symbols=None, **kwargs):
    if buy_symbols is None:
      buy_symbols = self.get_buy_symbols(**kwargs)
    buy_symbols.sort(key=lambda s: s[1], reverse=True)
    n_symbols = 0
    while (n_symbols < min(MAX_STOCK_PICK, len(buy_symbols)) and
           buy_symbols[n_symbols][1] > 0):
      n_symbols += 1
    total_weight = 0
    for i in range(n_symbols):
      total_weight += buy_symbols[i][1]
    trading_list = []
    common_share = 0.75
    for i in range(len(buy_symbols)):
      symbol = buy_symbols[i][0]
      weight = buy_symbols[i][1]
      if i < n_symbols:
        proportion = (common_share / n_symbols +
                      (1 - common_share) * weight / total_weight)
      else:
        proportion = 0
      trading_list.append((symbol, proportion, weight))
    return trading_list


def get_picked_points(series):
  """Gets threshold for best return of a series.

  This function uses full information of the sereis without truncation.
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


def get_ml_feature(series, price, avg_return, threshold):
  day_range_max = np.max(series[-DATE_RANGE:])
  day_range_change = (day_range_max - price) / day_range_max
  today_change = (price - series[-1]) / series[-1]
  yesterday_change = (series[-1] - series[-2]) / series[-2]
  twenty_day_change = (price - series[-20]) / series[-20]
  all_changes = [(series[t + 1] - series[t]) / series[t]
                 for t in range(len(series) - 1)
                 if series[t + 1] > 0 and series[t] > 0]
  feature = {'Average_Return': avg_return,
             'Threshold': threshold,
             'Today_Change': today_change,
             'Yesterday_Change': yesterday_change,
             'Twenty_Day_Change': twenty_day_change,
             'Day_Range_Change': day_range_change,
             'Change_Average': np.mean(all_changes),
             'Change_Variance': np.var(all_changes),
             'RSI': get_rsi(series),
             'MACD_Rate': get_macd_rate(series)}
  return feature


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


def get_macd_rate(series):
  price_average_12 = np.average(series[-12:])
  price_average_26 = np.average(series[-26:])
  return (price_average_12 - price_average_26) * 2 / (
      price_average_12 + price_average_26)
