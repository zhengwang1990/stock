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
import signal
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
            self.last_update = None
        else:
            logging.info('Loading current stock prices...')
            self.update_prices(self.closes.keys(), use_tqdm=True)
            self.last_update = datetime.datetime.now()

        for symbol in self.closes.keys():
            self.closes[symbol] = np.append(self.closes[symbol], 0)
            self.opens[symbol] = np.append(self.opens[symbol], 0)
            self.volumes[symbol] = np.append(self.volumes[symbol], 0)
        self.embed_prices_to_closes()

        self.trading_list = []
        self.next_market_close = self.alpaca.get_clock().next_close.timestamp()

    def embed_prices_to_closes(self):
        for symbol, price in self.prices.items():
            if symbol in self.closes:
                self.closes[symbol][-1] = price
        for symbol, closes in self.closes.items():
            if symbol not in self.prices:
                logging.error('[%s] Stock price not found', symbol)

    def drop_low_volume_symbols(self):
        """Drops to-be-tracked symbols with low volumes."""
        dropped_keys = []
        stock_universe = self.get_stock_universe()
        for symbol in self.closes.keys():
            if symbol not in stock_universe:
                dropped_keys.append(symbol)
        for symbol in dropped_keys:
            self.closes.pop(symbol)
            self.volumes.pop(symbol)
            self.opens.pop(symbol)
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

    def update_all_prices(self):
        """Keeps updating stock prices."""
        while time.time() < self.next_market_close:
            self.update_prices(self.closes.keys())
            self.last_update = datetime.datetime.now()
            if not self.active:
                return
            if time.time() > self.next_market_close - 60 * 10:
                time.sleep(10)
            elif time.time() > self.next_market_close - 60 * 30:
                time.sleep(300)
            else:
                time.sleep(600)

    def update_trading_list_prices(self):
        """Keeps updating stock prices of symbols in the trading list."""
        while time.time() < self.next_market_close:
            self.update_prices([symbol for symbol, _, _ in self.trading_list])
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
            p = self.polygon.last_trade(sym).price
            return p

        try:
            price = _get_realtime_price_impl(symbol)
        except requests.exceptions.RequestException as e:
            logging.error('[%s] Exception raised in get_realtime_price for: %s', symbol, e)
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

    @retrying.retry(stop_max_attempt_number=10, wait_exponential_multiplier=1000)
    def update_account(self):
        account = self.alpaca.get_account()
        self.equity = float(account.equity)
        self.cash = float(account.cash)

    def run(self):
        main_threads = []
        for target in [self.update_all_prices,
                       self.update_trading_list_prices,
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
                        self.active = False
                        raise exc_obj.with_traceback(exc_trace)
            time.sleep(1)

        for t in main_threads:
            t.join()

    def update_trading_list(self):
        """Keeps updating trading list with ML models."""
        print_all = False
        while time.time() < self.next_market_close:
            # Update trading list
            trading_list = self.get_trading_list()
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
            orders = self.alpaca.list_orders(status='open')
            for _ in range(5):
                if not orders:
                    break
                logging.info('Wait for orders to cancel. %d open orders remaining...', len(orders))
                time.sleep(1)
                orders = self.alpaca.list_orders(status='open')

    def print_trading_list(self, print_all=False):
        trading_table = []
        cost = 0
        for symbol, proportion, side in self.trading_list[:100]:
            if proportion == 0 and not print_all:
                continue
            trading_row = [symbol, utils.to_percent(proportion), side]
            price = self.closes[symbol][-1]
            change = price / self.closes[symbol][-2] - 1
            value = self.equity * proportion
            qty = int(value / price)
            share_cost = qty * price
            cost += share_cost
            trading_row.extend([utils.to_percent(change, True),
                                price,
                                share_cost, qty])
            trading_table.append(trading_row)
        headers = ['Symbol', 'Proportion', 'Side', 'Today Change',
                   'Price', 'Cost', 'Quantity']
        outputs = []
        if trading_table:
            outputs.append(tabulate(trading_table, headers=headers, tablefmt='grid'))
        else:
            logging.warning('NO stock satisfying trading criteria!')
        outputs.append(tabulate(
            [['Equity', '%.2f' % (self.equity,),
              'Estimated Cost', '%.2f' % (cost,),
              'Price Updates',
              self.last_update.strftime('%T') if self.last_update else 'Not Updated']], tablefmt='grid'))
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
        api_key = args.api_key or os.environ['ALPACA_API_KEY']
        api_secret = args.api_secret or os.environ['ALPACA_API_SECRET']
        base_url = utils.ALPACA_API_BASE_URL
    else:
        print('-' * 80)
        print('Using Alpaca API for paper market')
        print('-' * 80)
        api_key = os.environ['ALPACA_PAPER_API_KEY']
        api_secret = os.environ['ALPACA_PAPER_API_SECRET']
        base_url = utils.ALPACA_PAPER_API_BASE_URL
    sys.stdout.flush()
    alpaca = tradeapi.REST(api_key, api_secret, base_url, 'v2')
    polygon = polygonapi.REST(api_key)

    if alpaca.get_clock().is_open or args.force:
        trading = TradingRealTime(alpaca, polygon)
        trading.run()
    else:
        print('Market is closed. Use "-f" flag to force run.')


if __name__ == '__main__':
    main()
