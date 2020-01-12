import argparse
from common import *
from tabulate import tabulate


def get_static_buy_symbols(fund=None):
    """"Gets stock symbols to buy from previous close."""
    all_series = get_all_series(MAX_HISTORY_LOAD)
    all_series = filter_all_series(all_series)
    buy_symbols = get_buy_symbols(all_series, -1)
    trading_list = get_trading_list(buy_symbols)
    trading_table = []
    cost = 0

    for ticker, proportion in trading_list:
        trading_row = [ticker, '%.2f%%' % (proportion * 100,)]
        if fund:
            price = all_series[ticker][-1]
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


def get_dynamic_buy_symbols(fund=None):
    """"Gets stock symbols to buy from previous close."""
    pass


def main():
    parser = argparse.ArgumentParser(description='Stock trading strategy.')
    parser.add_argument('--fund', default=None, help='Total fund to trade')
    args = parser.parse_args()
    get_static_buy_symbols(args.fund)


if __name__ == '__main__':
    main()
