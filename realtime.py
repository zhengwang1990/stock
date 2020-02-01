import alpaca_trade_api as tradeapi
import argparse
import datetime
import json
import ml
import numpy as np
import threading
import time
import os
import utils
import tqdm
from tabulate import tabulate


class TradingRealTime(utils.TradingBase):

  def __init__(self, alpaca, model_name='', fund=None, real_trade=False):
    super(TradingRealTime, self).__init__(alpaca)
    self.active = True
    self.fund = fund
    self.model = ml.load_model(model_name) if model_name else None
    self.lock = threading.RLock()
    self.thresholds = {}
    self.down_percents = {}
    self.prices = {}
    self.ordered_symbols = []
    self.price_cache_file = os.path.join(
        self.root_dir, utils.CACHE_DIR, utils.get_business_day(0) + '-prices.json')

    read_cache = os.path.isfile(self.price_cache_file)
    if read_cache:
      with open(self.price_cache_file) as f:
        self.prices = json.loads(f.read())
    else:
      self.update_prices(self.all_series.keys(), use_tqdm=True)

    for ticker, series in self.closes.items():
      _, _, threshold = utils.get_picked_points(series[-utils.DAYS_IN_A_YEAR:])
      self.thresholds[ticker] = threshold

    self.update_ordered_symbols()

    update_frequencies = [(10, 60), (100, 600),
                          (len(self.ordered_symbols), 2400)]
    self.last_updates = ({update_frequencies[-1][1]: datetime.datetime.now()}
                         if not read_cache else {})

    for args in update_frequencies:
      t = threading.Thread(target=self.update_stats, args=args)
      t.daemon = True
      t.start()

  def update_stats(self, length, sleep_secs):
    while True:
      with self.lock:
        symbols = [symbol for symbol in self.ordered_symbols[:length]]
      self.update_prices(symbols)
      self.update_ordered_symbols()
      self.last_updates[sleep_secs] = datetime.datetime.now()
      time.sleep(sleep_secs)

  def get_real_time_price(self, ticker):
    price = _get_real_time_price_from_yahoo(ticker)
    self.prices[ticker] = price

  def update_prices(self, tickers, use_tqdm=False):
    threads = []
    for ticker in tickers:
      if not self.active:
        return
      t = self.pool.submit(self.get_real_time_price, ticker)
      threads.append(t)
    iterator = tqdm(threads, ncols=80) if use_tqdm else threads
    for t in iterator:
      if not self.active:
        return
      t.result()
    with self.lock:
      with open(self.price_cache_file, 'w') as f:
        f.write(json.dumps(self.prices))

  def update_ordered_symbols(self):
    tmp_ordered_symbols = []
    order_weights = {}
    for ticker, series in self.closes.items():
      if ticker not in self.prices:
        continue
      price = self.prices[ticker]
      down_percent = (np.max(series[-utils.DATE_RANGE:]) -
                      price) / np.max(series[-utils.DATE_RANGE:])
      threshold = self.thresholds[ticker]
      self.down_percents[ticker] = down_percent
      tmp_ordered_symbols.append(ticker)
      if down_percent >= threshold:
        order_weights[ticker] = min(np.abs(down_percent - threshold),
                                    np.abs((price - series[-1]) / series[-1]))
      else:
        order_weights[ticker] = np.abs(down_percent - threshold)
    tmp_ordered_symbols.sort(key=lambda ticker: order_weights[ticker])
    with self.lock:
      self.ordered_symbols = tmp_ordered_symbols

  def run(self):
    output_path = os.path.join(self.root_dir, utils.OUTPUTS_DIR,
                               utils.get_business_day(0) + '.txt')
    self.output_file = open(output_path, 'a')
    next_market_close = self.alpaca.get_clock().next_close.timestamp()
    while time.time() < next_market_close:
      trading_list = self.get_trading_list(prices=self.prices,
                                           model=self.model)
      # Update symbols in trading list to make sure they are up-to-date
      self.update_prices([ticker for ticker, _, _ in trading_list])
      self.update_ordered_symbols()
      utils.bi_print(utils.get_header(datetime.datetime.now().strftime('%H:%M:%S')),
                     self.output_file)
      self.print_trading_list(trading_list)
      utils.bi_print('Last updates: %s' % (
          [second_to_string(update_freq) + ': ' + update_time.strftime('%H:%M:%S')
           for update_freq, update_time in
           sorted(self.last_updates.items(), key=lambda t: t[0])],),
                     self.output_file)
      if time.time() > next_market_close - 60 * 5:
        time.sleep(30)
      else:
        time.sleep(300)
    self.active = False
    time.sleep(1)

  def print_trading_list(self, trading_list):
    trading_table = []
    cost = 0
    max_non_buy_print = 3
    for ticker, proportion, weight in trading_list:
      max_non_buy_print -= 1
      if proportion == 0 and max_non_buy_print < 0:
        continue
      trading_row = [ticker, '%.2f%%' % (proportion * 100,), weight]
      price = self.prices[ticker]
      change = (price - self.closes[ticker][-1]) / self.closes[ticker][-1]
      trading_row.extend(['%.2f%%' % (change * 100,),
                          '%.2f%%' % (-self.down_percents[ticker] * 100,),
                          '%.2f%%' % (-self.thresholds[ticker] * 100,), price])
      if self.fund:
        value = self.fund * proportion
        n_shares = int(value / price)
        share_cost = n_shares * price
        cost += share_cost
        trading_row.extend([share_cost, n_shares])
      trading_table.append(trading_row)
    headers = ['Symbol', 'Proportion', 'Weight', 'Today Change',
               '%d Day Change' % (utils.DATE_RANGE,), 'Threshold', 'Price']
    if self.fund:
      headers.extend(['Cost', 'Quantity'])
    if trading_table:
      utils.bi_print(tabulate(trading_table, headers=headers, tablefmt='grid'),
                     self.output_file)
      if self.fund:
        utils.bi_print('Fund: %.2f' % (fund,), self.output_file)
        utils.bi_print('Actual Cost: %.2f' % (cost,), self.output_file)


def _get_real_time_price_from_yahoo(ticker):
  url = 'https://finance.yahoo.com/quote/{}'.format(ticker)
  prefixes = ['"currentPrice"', '"regularMarketPrice"']
  try:
    price = float(utils.web_scraping(url, prefixes))
  except Exception as e:
    print(e)
    price = None
  return price


def second_to_string(secs):
  if secs < 60:
    return str(secs) + 's'
  elif secs < 3600:
    return str(secs // 60) + 'm'
  else:
    return str(secs // 3600) + 'h'


def main():
  parser = argparse.ArgumentParser(description='Stock trading realtime.')
  parser.add_argument('--fund', default=None, help='Total fund to trade.')
  parser.add_argument('--model', default='model_p612804.hdf5', help='Machine learning model for prediction.')
  parser.add_argument('--api_key', default=None, help='Alpaca API key.')
  parser.add_argument('--api_secret', default=None, help='Alpaca API secret.')
  parser.add_argument("--real_trade", help='Trade with real money.',
                      action="store_true")
  args = parser.parse_args()

  alpaca = tradeapi.REST(args.api_key or os.environ['ALPACA_API_KEY'],
                         args.api_secret or os.environ['ALPACA_API_SECRET'],
                         utils.APCA_API_BASE_URL, 'v2')
  fund = float(args.fund) if args.fund else None
  trading = TradingRealTime(alpaca, model_name=args.model, fund=args.fund,
                            real_trade=args.real_trade)
  trading.run()


if __name__ == '__main__':
    main()
