import alpaca_trade_api as tradeapi
import alpaca_trade_api.polygon as polygonapi
import argparse
import datetime
import json
import logging
import numpy as np
import os
import sys
import threading
import time
import requests
import retrying
import utils
from concurrent import futures
from tabulate import tabulate
from tqdm import tqdm

ERROR_TOLERANCE = 10


class TradingRealTime(utils.TradingBase):
    """Tracks daily stock price changes and make transactions on Alpaca."""

    def __init__(self, alpaca, polygon):
        self.root_dir = os.path.dirname(os.path.realpath(__file__))
        output_dir = os.path.join(self.root_dir, utils.OUTPUTS_DIR, 'realtime',
                                  utils.get_business_day(0))
        os.makedirs(output_dir, exist_ok=True)
        utils.logging_config(os.path.join(output_dir, 'log.txt'))
        super(TradingRealTime, self).__init__(alpaca)
        self.active = True
        self.equity, self.cash = 0, 0
        self.polygon = polygon
        self.update_account()
        self.lock = threading.RLock()
        self.thresholds = {}
        self.prices = {}
        self.ordered_symbols = []
        self.errors = []

        self.price_cache_file = os.path.join(output_dir, 'prices.json')
        self.drop_low_volume_symbols()

        read_cache = os.path.isfile(self.price_cache_file)
        if read_cache:
            logging.info('Reading cached stock prices...')
            with open(self.price_cache_file) as f:
                self.prices = json.loads(f.read())
        else:
            logging.info('Loading current stock prices...')
            self.update_prices(self.closes.keys(), use_tqdm=True)

        for symbol in self.closes.keys():
            threshold = self.get_threshold(symbol)
            self.thresholds[symbol] = threshold

        self.update_ordered_symbols()

        self.update_frequencies = [(10, 120), (100, 600),
                                   (len(self.ordered_symbols), 2400)]
        self.last_updates = ({self.update_frequencies[-1][0]: datetime.datetime.now()}
                             if not read_cache else {})
        self.trading_list = []
        self.next_market_close = self.alpaca.get_clock().next_close.timestamp()

    def drop_low_volume_symbols(self):
        """Drops to-be-tracked symbols with low volumes."""
        dropped_keys = []
        for symbol in self.closes.keys():
            avg_trading_volume = np.average(np.multiply(
                self.closes[symbol][-20:], self.volumes[symbol][-20:]))
            if avg_trading_volume < utils.VOLUME_FILTER_THRESHOLD and symbol != '^VIX':
                dropped_keys.append(symbol)
        for symbol in dropped_keys:
            self.closes.pop(symbol)
            self.volumes.pop(symbol)
        logging.info('%d loaded symbols after drop symbols with cash volume lower than $%.1E',
                     len(self.closes), utils.VOLUME_FILTER_THRESHOLD)

    def trade_clock_watcher(self):
        """Makes transactions near market close."""
        while time.time() < self.next_market_close - 90:
            time.sleep(1)
        self.active = False
        # Wait for all printing done
        time.sleep(1)
        self.trade()

    def update_stats(self, length, sleep_secs):
        """Keeps updating a subset of symbols in self.ordered_symbols."""
        while time.time() < self.next_market_close:
            with self.lock:
                symbols = [symbol for symbol in self.ordered_symbols[:length]]
            self.update_prices(symbols)
            self.update_ordered_symbols()
            self.last_updates[length] = datetime.datetime.now()
            if not self.active:
                return
            if time.time() > self.next_market_close - 60 * 5:
                time.sleep(sleep_secs / 20)
            else:
                time.sleep(sleep_secs)

    def update_trading_list_prices(self):
        """Keeps updating stock prices of symbols in the trading list."""
        while time.time() < self.next_market_close:
            self.update_prices(['^VIX'] + [symbol for symbol, _, _ in self.trading_list])
            if not self.active:
                return
            if time.time() > self.next_market_close - 60 * 5:
                time.sleep(1)
            else:
                time.sleep(60)

    def get_realtime_price(self, symbol):
        """Obtains realtime price for a symbol."""

        @retrying.retry(stop_max_attempt_number=5, wait_exponential_multiplier=1000)
        def _get_realtime_price_impl(sym):
            if sym == '^VIX':
                p = float(utils.web_scraping('https://finance.yahoo.com/quote/^VIX',
                                             ['"currentPrice"', '"regularMarketPrice"']))
            else:
                p = self.polygon.last_trade(sym).price
            return p

        try:
            price = _get_realtime_price_impl(symbol)
        except requests.exceptions.RequestException as e:
            logging.error('Exception raised in get_realtime_price for %s: %s', symbol, e)
            self.errors.append(sys.exc_info())
        else:
            self.prices[symbol] = price

    def update_prices(self, symbols, use_tqdm=False):
        """Updates realtime prices for a list of symbols."""
        threads = []
        with futures.ThreadPoolExecutor(max_workers=3) as pool:
            for symbol in symbols:
                if not self.active:
                    return
                t = pool.submit(self.get_realtime_price, symbol)
                threads.append(t)
            iterator = (tqdm(threads, ncols=80, leave=False)
                        if use_tqdm and sys.stdout.isatty() else threads)
            for t in iterator:
                if not self.active:
                    return
                t.result()
        with self.lock:
            with open(self.price_cache_file, 'w') as f:
                f.write(json.dumps(self.prices))

    def update_ordered_symbols(self):
        """Re-orders self.ordered_symbols based on how likely symbols will be selected."""
        tmp_ordered_symbols = []
        order_weights = {}
        for symbol, close in self.closes.items():
            if symbol not in self.prices:
                self.get_realtime_price(symbol)
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
        tmp_ordered_symbols.sort(key=lambda s: order_weights[s])
        with self.lock:
            self.ordered_symbols = tmp_ordered_symbols

    @retrying.retry(stop_max_attempt_number=10, wait_exponential_multiplier=1000)
    def update_account(self):
        account = self.alpaca.get_account()
        self.equity = float(account.equity)
        self.cash = float(account.cash)

    def run(self):
        for args in self.update_frequencies:
            t = threading.Thread(target=self.update_stats, args=args)
            t.daemon = True
            t.start()

        main_threads = []
        for target in [self.update_trading_list_prices,
                       self.trade_clock_watcher,
                       self.update_trading_list]:
            t = threading.Thread(target=target)
            t.daemon = True
            t.start()
            main_threads.append(t)

        while time.time() < self.next_market_close:
            if len(self.errors) > ERROR_TOLERANCE:
                self.active = False
                for i in range(len(self.errors)):
                    _, exc_obj, exc_trace = self.errors[i]
                    logging.error('Error # %d: %s', i + 1, exc_obj)
                    if i == len(self.errors) - 1:
                        raise exc_obj.with_traceback(exc_trace)
            time.sleep(1)

        for t in main_threads:
            t.join()

    def update_trading_list(self):
        """Keeps updating trading list with ML models."""
        print_all = False
        while time.time() < self.next_market_close:
            # Update trading list
            trading_list = self.get_trading_list(prices=self.prices)
            if not self.active:
                return

            self.trading_list = trading_list
            self.update_account()

            # Print
            self.print_trading_list(print_all)
            if not self.active:
                return

            # Wait for next update
            if time.time() > self.next_market_close - 60 * 2:
                time.sleep(1)
            elif time.time() > self.next_market_close - 60 * 5:
                print_all = True
                time.sleep(10)
            elif time.time() > self.next_market_close - 60 * 20:
                time.sleep(100)
            else:
                time.sleep(300)

    def trade(self):
        """Performs sell and buy transactions."""
        # Sell all current positions with limit orders
        self.sell('limit', deadline=self.next_market_close - 60)
        # Sell remaining positions with market orders
        self.sell('market')

        for _ in range(10):
            self.update_account()
            if self.equity == self.cash:
                break
            time.sleep(1)
        else:
            logging.warning(
                'Timeout while waiting for cash to settle. Equity: %s; Cash: %s.', self.equity, self.cash)

        # Buy with limit orders
        self.buy('limit', deadline=self.next_market_close - 30)
        # Buy with market orders
        self.buy('market')

    @retrying.retry(stop_max_attempt_number=10, wait_exponential_multiplier=1000)
    def sell(self, order_type, deadline=None):
        """Sells all current positions."""
        positions = self.alpaca.list_positions()
        positions_table = []
        for position in positions:
            try:
                if order_type == 'limit':
                    self.alpaca.submit_order(position.symbol, int(position.qty), 'sell', 'limit', 'day',
                                             limit_price=float(position.current_price))
                elif order_type == 'market':
                    self.alpaca.submit_order(position.symbol, int(position.qty), 'sell', 'market', 'day')
                else:
                    raise NotImplementedError('Order type %s not recognized' % (order_type,))
                positions_table.append([position.symbol, position.current_price, position.qty,
                                        float(position.market_value) - float(position.cost_basis)])
            except tradeapi.rest.APIError as e:
                logging.error('Failed to sell %s: %s', position.symbol, e)
        outputs = [utils.get_header('Place ' + order_type.capitalize() + ' Sell Order')]
        if positions_table:
            outputs.append(tabulate(positions_table,
                                    headers=['Symbol', 'Price', 'Quantity', 'Estimated Gain Value'],
                                    tablefmt='grid'))
        logging.info('\n'.join(outputs))
        self.wait_for_order_to_fill(deadline=deadline)

    @retrying.retry(stop_max_attempt_number=10, wait_exponential_multiplier=1000)
    def buy(self, order_type, deadline=None):
        """Buys stocks in the trading list."""
        orders_table = []
        positions = self.alpaca.list_positions()
        existing_positions = {position.symbol: int(position.qty) for position in positions}
        for symbol, proportion, _ in self.trading_list:
            if proportion == 0:
                continue
            adjust = 1 if order_type == 'limit' else 0.98
            qty = int(self.cash * proportion * adjust / self.prices[symbol])
            if symbol in existing_positions:
                qty -= existing_positions[symbol]
            if qty > 0:
                orders_table.append([symbol, self.prices[symbol], qty, self.prices[symbol] * qty])
                try:
                    if order_type == 'limit':
                        self.alpaca.submit_order(symbol, qty, 'buy', 'limit', 'day',
                                                 limit_price=self.prices[symbol])
                    elif order_type == 'market':
                        self.alpaca.submit_order(symbol, qty, 'buy', 'market', 'day')
                    else:
                        raise NotImplementedError('Order type %s not recognized' % (order_type,))
                except tradeapi.rest.APIError as e:
                    logging.error('Failed to buy %s: %s', symbol, e)
        outputs = [utils.get_header('Place ' + order_type.capitalize() + ' Buy Order')]
        if orders_table:
            outputs.append(tabulate(orders_table,
                                    headers=['Symbol', 'Price', 'Quantity', 'Estimated Cost'],
                                    tablefmt='grid'))
        logging.info('\n'.join(outputs))
        self.wait_for_order_to_fill(deadline=deadline)

    @retrying.retry(stop_max_attempt_number=10, wait_exponential_multiplier=1000)
    def wait_for_order_to_fill(self, timeout=20, deadline=None):
        orders = self.alpaca.list_orders(status='open')
        wait_time = 0
        while orders:
            logging.info('Wait for orders to fill. %d open orders remaining...', len(orders))
            time.sleep(2)
            wait_time += 2
            if wait_time >= timeout:
                break
            if deadline and time.time() >= deadline:
                break
            orders = self.alpaca.list_orders(status='open')
        if not orders:
            logging.info('All orders filled')
        else:
            logging.info('Cancel %d remaining orders', len(orders))
            self.alpaca.cancel_all_orders()

    def print_trading_list(self, print_all=False):
        trading_table = []
        cost = 0
        for symbol, proportion, weight in self.trading_list[:100]:
            if proportion == 0 and not print_all:
                continue
            trading_row = [symbol, '%.2f%%' % (proportion * 100,), weight]
            price = self.prices[symbol]
            change = (price - self.closes[symbol][-1]) / self.closes[symbol][-1]
            day_range_change = price / np.max(self.closes[symbol][-utils.DATE_RANGE:]) - 1
            value = self.equity * proportion
            qty = int(value / price)
            share_cost = qty * price
            cost += share_cost
            trading_row.extend(['%+.2f%%' % (change * 100,),
                                '%+.2f%%' % (day_range_change * 100,),
                                '%+.2f%%' % (self.thresholds[symbol] * 100,), price,
                                share_cost, qty])
            trading_table.append(trading_row)
        headers = ['Symbol', 'Proportion', 'Weight', 'Today Change',
                   '%d Day Change' % (utils.DATE_RANGE,), 'Threshold', 'Price',
                   'Cost', 'Quantity']
        outputs = []
        if trading_table:
            outputs.append(tabulate(trading_table, headers=headers, tablefmt='grid'))
        else:
            logging.warning('NO stock satisfying trading criteria!')
        outputs.append(tabulate(
            [['Equity', '%.2f' % (self.equity,),
              'Estimated Cost', '%.2f' % (cost,),
              'Price Updates',
              ', '.join(['TOP ' + str(update_length) + ': ' + update_time.strftime('%T')
                         for update_length, update_time in
                         sorted(self.last_updates.items(), key=lambda t: t[0])])]], tablefmt='grid'))
        logging.info('\n'.join(outputs))


def main():
    parser = argparse.ArgumentParser(description='Stock trading realtime.')
    parser.add_argument('--api_key', default=None, help='Alpaca API key.')
    parser.add_argument('--api_secret', default=None, help='Alpaca API secret.')
    parser.add_argument('--real_trade', help='Trade with real money.',
                        action="store_true")
    parser.add_argument('-f', '--force', help='Force to run even at market close.',
                        action="store_true")
    args = parser.parse_args()

    if args.api_key and args.api_secret or args.real_trade:
        print('-' * 80)
        print('Using Alpaca API for live trading')
        print('-' * 80)
        alpaca = tradeapi.REST(args.api_key or os.environ['ALPACA_API_KEY'],
                               args.api_secret or os.environ['ALPACA_API_SECRET'],
                               utils.ALPACA_API_BASE_URL, 'v2')
        polygon = polygonapi.REST(args.api_key or os.environ['ALPACA_API_KEY'])
    else:
        print('-' * 80)
        print('Using Alpaca API for paper market')
        print('-' * 80)
        alpaca = tradeapi.REST(os.environ['ALPACA_PAPER_API_KEY'],
                               os.environ['ALPACA_PAPER_API_SECRET'],
                               utils.ALPACA_PAPER_API_BASE_URL, 'v2')
        polygon = polygonapi.REST(os.environ['ALPACA_PAPER_API_KEY'])

    if alpaca.get_clock().is_open or args.force:
        trading = TradingRealTime(alpaca, polygon)
        trading.run()
    else:
        print('Market is closed. Use "-f" flag to force run.')


if __name__ == '__main__':
    main()
