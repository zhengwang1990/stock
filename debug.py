import matplotlib.pyplot as plt
import simulate
import utils
import alpaca_trade_api as tradeapi
import time
import os
import ta
import numpy as np
import pandas as pd


def test_alpaca():
    alpaca = tradeapi.REST(os.environ['ALPACA_PAPER_API_KEY'],
                           os.environ['ALPACA_PAPER_API_SECRET'],
                           utils.ALPACA_PAPER_API_BASE_URL, 'v2')

    # Time
    clock = alpaca.get_clock()
    closingTime = clock.next_close.timestamp()
    currTime = time.time()
    print('Market is open:', clock.is_open)
    print('Market close time:', closingTime)
    print('Current time:', currTime)

    # Account info
    account = alpaca.get_account()
    print(account)

    print('Account cash:', float(account.cash))
    print('Account equity:', float(account.equity))

    # Submit buy order
    alpaca.submit_order('TAL', 1, 'buy', 'market', 'day')
    print('Buy order submitted')
    orders = alpaca.list_orders(status='open')
    while orders:
        print('Wait for order to fill...')
        time.sleep(1)
        orders = alpaca.list_orders(status='open')

    # Positions
    positions = alpaca.list_positions()
    print('Get %d positions' % (len(positions),))
    for position in positions:
        print(position)

    # Account info
    account = alpaca.get_account()
    print('Account cash:', float(account.cash))

    # Submit sell order
    alpaca.submit_order('TAL', 1, 'sell', 'market', 'day')
    print('Sell order submitted')
    orders = alpaca.list_orders(status='open')
    while orders:
        print('Wait for order to fill...')
        time.sleep(1)
        orders = alpaca.list_orders(status='open')

    # Account info
    account = alpaca.get_account()
    print('Account cash:', float(account.cash))
    print('Account buying power:', float(account.buying_power))


def main():
    test_alpaca()


if __name__ == '__main__':
    main()
