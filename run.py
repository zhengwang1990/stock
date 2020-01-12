import argparse
import datetime
import requests
import threading
import time
import pytz
from common import *
from tabulate import tabulate


class Trading(object):

    def __init__(self, fund):
        self.fund = fund
        self.lock = threading.RLock()
        self.tz = pytz.timezone('America/New_York')
        self.close_time = datetime.datetime.combine(datetime.datetime.now(self.tz).today(),
                                                    datetime.time(16, 0),
                                                    tzinfo=self.tz)
        self.all_series = filter_all_series(get_all_series(MAX_HISTORY_LOAD))
        self.thresholds = {}
        self.avg_returns = {}
        self.down_percents = {}
        self.prices = {}
        self.ordered_symbols = []
        for ticker, series in tqdm(self.all_series.items(), ncols=80, bar_format='{percentage:3.0f}%|{bar}{r_bar}',
                                   leave=False, file=sys.stdout):
            try:
                price = get_real_time_price(ticker)
            except Exception as e:
                print('\n%s: not able to get realtime price: %s' % (ticker, e))
                continue
            _, avg_return, threshold = get_picked_points(series[-LOOK_BACK_DAY:])
            self.thresholds[ticker] = threshold
            self.avg_returns[ticker] = avg_return
            self.prices[ticker] = price
            down_percent = (np.max(series[-DATE_RANGE:]) - price) / np.max(series[-DATE_RANGE:])
            self.down_percents[ticker] = down_percent
            self.ordered_symbols.append((np.abs(down_percent - threshold), ticker))
        self.ordered_symbols.sort()

        for args in [(10, 30), (100, 500), (None, 3600)]:
            t = threading.Thread(target=self.update_buy_symbols, args=args)
            t.daemon = True
            t.start()

    def update_buy_symbols(self, first_n, sleep_secs):
        while True:
            length = first_n or len(self.ordered_symbols)
            for i in range(length):
                _, ticker = self.ordered_symbols[i]
                self.update_ticker(ticker)
            self.ordered_symbols.sort()
            time.sleep(sleep_secs)

    def update_ticker(self, ticker):
        price = get_real_time_price(ticker)
        with self.lock:
            self.prices[ticker] = price
            series = self.all_series[ticker]
            down_percent = (np.max(series[-DATE_RANGE:]) - price) / np.max(series[-DATE_RANGE:])
            self.down_percents[ticker] = down_percent
            threshold = self.thresholds[ticker]
            self.ordered_symbols[ticker] = (np.abs(down_percent - threshold), ticker)

    def run(self):
        while datetime.datetime.now(self.tz) < self.close_time:
            buy_symbols = []
            for ticker, series in self.all_series.items():
                is_buy = self.price[ticker] < series[-1] and self.down_percents[ticker] > self.thresholds[ticker]
                if is_buy:
                    self.update_ticker(ticker)
                    is_buy_confirm = (self.price[ticker] < series[-1] and
                                      self.down_percents[ticker] > self.thresholds[ticker])
                    if is_buy_confirm:
                        buy_symbols.append((self.avg_returns[ticker], ticker))
            trading_list = get_trading_list(buy_symbols)
            print_trading_list(trading_list, self.prices, self.fund)
            time.sleep(30)


def get_real_time_price(ticker):
    #tk = yf.Ticker(ticker)
    #return tk.info['ask'] * (np.random.random() + 0.5)
    max_retry = 60
    for _ in range(max_retry):
        response = requests.get('https://finnhub.io/api/v1/quote?symbol=AAPL&token=bodp0tfrh5r95o03irlg')
        if response.ok:
            break
        else:
            time.sleep(1)
    else:
        raise Exception('Timeout while requesting quote')
    return response.json()['c'] * (np.random.random() + 0.5)


def get_static_trading_table(fund=None):
    """"Gets stock symbols to buy from previous close."""
    all_series = get_all_series(MAX_HISTORY_LOAD)
    all_series = filter_all_series(all_series)
    buy_symbols = get_buy_symbols(all_series, -1)
    trading_list = get_trading_list(buy_symbols)
    price_list = {ticker: all_series[ticker][-1] for ticker, _ in trading_list}
    print_trading_list(trading_list, price_list, fund)


def print_trading_list(trading_list, price_list=[], fund=None):
    trading_table = []
    cost = 0
    for ticker, proportion in trading_list:
        trading_row = [ticker, '%.2f%%' % (proportion * 100,)]
        if fund:
            price = price_list[ticker]
            value = fund * proportion
            n_shares = np.round(value / price)
            share_cost = n_shares * price
            cost += share_cost
            trading_row.extend([price, share_cost, n_shares])
        trading_table.append(trading_row)
    headers = ['Symbol', 'Proportion']
    if fund:
        headers.extend(['Price', 'Cost', 'Quantity'])
    if trading_table:
        print(tabulate(trading_table, headers=headers, tablefmt='grid'))
        if fund:
            print('Fund: %.2f' % (fund,))
            print('Actual Cost: %.2f' % (cost,))


def get_live_trading_table(fund=None):
    """"Gets stock symbols to buy from previous close."""
    trading = Trading(fund)
    trading.run()


def main():
    parser = argparse.ArgumentParser(description='Stock trading strategy.')
    parser.add_argument('--fund', default=10000, help='Total fund to trade.')
    parser.add_argument('--mode', default='live', choices=['live', 'static'], help='Mode to run.')
    args = parser.parse_args()
    if args.mode == 'live':
        get_live_trading_table(args.fund)
    elif args.mode == 'static':
        get_static_trading_table(args.fund)
    else:
        raise Exception('Unknown mode')


if __name__ == '__main__':
    main()
