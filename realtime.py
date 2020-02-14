import alpaca_trade_api as tradeapi
import argparse
import datetime
import json
import numpy as np
import threading
import time
import os
import utils
from concurrent import futures
from tqdm import tqdm
from tabulate import tabulate


class TradingRealTime(utils.TradingBase):

    def __init__(self, alpaca):
        super(TradingRealTime, self).__init__(alpaca)
        self.active = True
        self.update_account()
        self.lock = threading.RLock()
        self.thresholds = {}
        self.prices = {}
        self.ordered_symbols = []
        self.price_cache_file = os.path.join(
            self.cache_path, utils.get_business_day(0) + '-prices.json')

        read_cache = os.path.isfile(self.price_cache_file)
        if read_cache:
            with open(self.price_cache_file) as f:
                self.prices = json.loads(f.read())
        else:
            print('Loading current stock prices...')
            self.update_prices(self.closes.keys(), use_tqdm=True)

        for symbol, close in self.closes.items():
            threshold = utils.get_threshold(close[-utils.DAYS_IN_A_YEAR:])
            self.thresholds[symbol] = threshold

        self.update_ordered_symbols()

        update_frequencies = [(10, 60), (100, 600),
                              (len(self.ordered_symbols), 2400)]
        self.last_updates = ({update_frequencies[-1][1]: datetime.datetime.now()}
                             if not read_cache else {})

        for args in update_frequencies:
            t = threading.Thread(target=self.update_stats, args=args)
            t.daemon = True
            t.start()
        output_path = os.path.join(self.root_dir, utils.OUTPUTS_DIR,
                                   utils.get_business_day(0) + '.txt')
        self.output_file = open(output_path, 'a')

    def update_stats(self, length, sleep_secs):
        while True:
            with self.lock:
                symbols = [symbol for symbol in self.ordered_symbols[:length]]
            self.update_prices(symbols)
            self.update_ordered_symbols()
            self.last_updates[sleep_secs] = datetime.datetime.now()
            time.sleep(sleep_secs)

    def get_real_time_price(self, symbol):
        price = _get_real_time_price_from_yahoo(symbol)
        if price is not None:
          self.prices[symbol] = price

    def update_prices(self, symbols, use_tqdm=False):
        threads = []
        with futures.ThreadPoolExecutor(max_workers=utils.MAX_THREADS) as pool:
            for symbol in symbols:
                if not self.active:
                    return
                t = pool.submit(self.get_real_time_price, symbol)
                threads.append(t)
            iterator = tqdm(threads, ncols=80, leave=False) if use_tqdm else threads
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
        for symbol, close in self.closes.items():
            if symbol not in self.prices:
                continue
            price = self.prices[symbol]
            today_change = price / close[-1] - 1
            day_range_change = price / np.max(close[-utils.DATE_RANGE:]) - 1
            threshold = self.thresholds[symbol]
            tmp_ordered_symbols.append(symbol)
            if day_range_change < threshold:
                order_weights[symbol] = min(
                    np.abs(day_range_change - threshold),
                    np.abs(np.abs(today_change) - 0.5 * np.abs(day_range_change)))
            else:
                order_weights[symbol] = np.abs(day_range_change - threshold)
        tmp_ordered_symbols.sort(key=lambda symbol: order_weights[symbol])
        with self.lock:
            self.ordered_symbols = tmp_ordered_symbols

    def update_account(self):
        account = self.alpaca.get_account()
        self.equity = float(account.equity)
        self.cash = float(account.cash)

    def run(self):
        next_market_close = self.alpaca.get_clock().next_close.timestamp()
        trading_list = []
        while time.time() < next_market_close:
            utils.bi_print(utils.get_header(datetime.datetime.now().strftime('%T')),
                           self.output_file)
            # Update symbols in trading list to make sure they are up-to-date
            self.update_prices(['^VIX'] + [symbol for symbol, _, _ in trading_list], use_tqdm=True)
            trading_list = self.get_trading_list(prices=self.prices)
            self.update_account()
            self.print_trading_list(trading_list)
            utils.bi_print('Last updates: %s' % (
                [second_to_string(update_freq) + ': ' + update_time.strftime('%T')
                 for update_freq, update_time in
                 sorted(self.last_updates.items(), key=lambda t: t[0])],),
                           self.output_file)
            if time.time() > next_market_close - 90:
                self.trade(trading_list)
                break
            elif time.time() > next_market_close - 60 * 2:
                time.sleep(1)
            elif time.time() > next_market_close - 60 * 5:
                time.sleep(10)
            elif time.time() > next_market_close - 60 * 20:
                time.sleep(100)
            else:
                time.sleep(300)
        self.active = False
        time.sleep(10)

    def trade(self, trading_list):
        # Sell all current positions
        utils.bi_print(utils.get_header('Place Sell Orders At ' +
                                        datetime.datetime.now().strftime('%T')),
                       self.output_file)
        positions = self.alpaca.list_positions()
        positions_table = []
        for position in positions:
            try:
                self.alpaca.submit_order(position.symbol, int(position.qty), 'sell', 'market', 'day')
                positions_table.append([position.symbol, position.current_price, position.qty,
                                        float(position.market_value) - float(position.cost_basis)])
            except tradeapi.rest.APIError as e:
                print('Failed to sell %s: %s' % (position.symbol, e))
        if positions_table:
            utils.bi_print(tabulate(positions_table,
                                    headers=['Symbol', 'Price', 'Quantity', 'Estimated Gain Value'],
                                    tablefmt='grid'),
                           self.output_file)
        self._wait_for_order_to_fill()

        for _ in range(10):
            self.update_account()
            if self.equity == self.cash:
                break
            time.sleep(1)
        else:
            utils.bi_print('-' * 80, self.output_file)
            utils.bi_print(
                'Warning: timeout while waiting for cash to settle. Equity: %s; Cash: %s.' % (
                    self.equity, self.cash),
                self.output_file)
            utils.bi_print('-' * 80, self.output_file)

        # Order all current positions
        utils.bi_print(utils.get_header('Place Buy Orders At ' +
                                        datetime.datetime.now().strftime('%T')),
                       self.output_file)
        orders_table = []
        estimate_cost = 0
        for symbol, proportion, _ in trading_list:
            if proportion == 0:
                continue
            qty = int(self.cash * proportion / self.prices[symbol])
            if qty > 0:
                orders_table.append([symbol, self.prices[symbol], qty, self.prices[symbol] * qty])
                estimate_cost += self.prices[symbol] * qty
                try:
                    self.alpaca.submit_order(symbol, qty, 'buy', 'market', 'day')
                except tradeapi.rest.APIError as e:
                    print('Failed to buy %s: %s' % (symbol, e))
        if orders_table:
            utils.bi_print(tabulate(orders_table,
                                    headers=['Symbol', 'Price', 'Quantity', 'Estimated Cost'],
                                    tablefmt='grid'),
                           self.output_file)
        utils.bi_print('Current Cash: %.2f. Estimated Total Cost: %.2f.' % (
            self.cash, estimate_cost),
                       self.output_file)
        self._wait_for_order_to_fill()

    def _wait_for_order_to_fill(self):
        orders = self.alpaca.list_orders(status='open')
        while orders:
            utils.bi_print('Wait for order to fill. %d open orders remaining...' % (
                len(orders),), self.output_file)
            time.sleep(2)
            orders = self.alpaca.list_orders(status='open')

    def print_trading_list(self, trading_list):
        trading_table = []
        cost = 0
        for symbol, proportion, weight in trading_list:
            if proportion == 0:
                continue
            trading_row = [symbol, '%.2f%%' % (proportion * 100,), weight]
            price = self.prices[symbol]
            change = (price - self.closes[symbol][-1]) / self.closes[symbol][-1]
            day_range_change = price / np.max(self.closes[symbol][-utils.DATE_RANGE:]) - 1
            value = self.equity * proportion
            qty = int(value / price)
            share_cost = qty * price
            cost += share_cost
            trading_row.extend(['%.2f%%' % (change * 100,),
                                '%.2f%%' % (day_range_change * 100,),
                                '%.2f%%' % (self.thresholds[symbol] * 100,), price,
                                share_cost, qty])
            trading_table.append(trading_row)
        headers = ['Symbol', 'Proportion', 'Weight', 'Today Change',
                   '%d Day Change' % (utils.DATE_RANGE,), 'Threshold', 'Price',
                   'Cost', 'Quantity']
        if trading_table:
            utils.bi_print(tabulate(trading_table, headers=headers, tablefmt='grid'),
                           self.output_file)
            utils.bi_print('Equity: %.2f' % (self.equity,), self.output_file)
            utils.bi_print('Estimated Cost: %.2f' % (cost,), self.output_file)
        else:
            utils.bi_print('NO stock satisfying trading criteria.', self.output_file)


def _get_real_time_price_from_yahoo(symbol):
    url = 'https://finance.yahoo.com/quote/{}'.format(symbol)
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
    parser.add_argument('--api_key', default=None, help='Alpaca API key.')
    parser.add_argument('--api_secret', default=None, help='Alpaca API secret.')
    parser.add_argument("--real_trade", help='Trade with real money.',
                        action="store_true")
    args = parser.parse_args()

    if args.api_key and args.api_secret or args.real_trade:
        print('-' * 80)
        print('Using Alpaca API for live trading')
        print('-' * 80)
        alpaca = tradeapi.REST(args.api_key or os.environ['ALPACA_API_KEY'],
                               args.api_secret or os.environ['ALPACA_API_SECRET'],
                               utils.ALPACA_API_BASE_URL, 'v2')
    else:
        print('-' * 80)
        print('Using Alpaca API for paper market')
        print('-' * 80)
        alpaca = tradeapi.REST(os.environ['ALPACA_PAPER_API_KEY'],
                               os.environ['ALPACA_PAPER_API_SECRET'],
                               utils.ALPACA_PAPER_API_BASE_URL, 'v2')
    trading = TradingRealTime(alpaca)
    trading.run()


if __name__ == '__main__':
    main()
