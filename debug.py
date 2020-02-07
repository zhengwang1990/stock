import matplotlib.pyplot as plt
import simulate
import utils
import alpaca_trade_api as tradeapi
import time
import os
import ta
import numpy as np
import pandas as pd


def plot_buy_points(ticker):
    series = get_series(ticker)[1][-LOOK_BACK_DAY:]
    pick_t, avg_return, threshold = get_picked_points(series)
    print(get_header(ticker))
    print('Average return: %.2f%%' % (avg_return * 100,))
    print('Threshold: %.2f%%' % (threshold * 100,))
    t = range(len(series))
    plt.plot(t, series)
    plt.plot(pick_t, series[pick_t], 'o')
    plt.show()


def test_alpaca():
    alpaca = tradeapi.REST(os.environ['ALPACA_PAPER_API_KEY'],
                           os.environ['ALPACA_PAPER_API_SECRET'],
                           utils.ALPACA_PAPER_API_BASE_URL, 'v2')

    # Time
    clock = alpaca.get_clock()
    closingTime = clock.next_close.replace().timestamp()
    currTime = time.time()
    print('Market is open:', clock.is_open)
    print('Market close time:', closingTime)
    print('Current time:', currTime)

    # Account info
    account = alpaca.get_account()
    print(account)
    exit(0)
    print('Account cash:', float(account.cash))
    print('Account equity:', float(account.equity))

    # Submit buy order
    #alpaca.submit_order('TAL', 1, 'buy', 'market', 'day')
    #print('Buy order submitted')
    #orders = alpaca.list_orders(status='open')
    #while orders:
    #    print('Wait for order to fill...')
    #    time.sleep(1)
    #    orders = alpaca.list_orders(status='open')

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
    #plot_buy_points('AMEH')
    test_alpaca()


if __name__ == '__main__':
    main()
