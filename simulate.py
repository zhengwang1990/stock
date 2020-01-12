import sys
import json
from common import *
from tabulate import tabulate
from tqdm import tqdm

MAX_HISTORY_LOAD = '5y'
CACHE_DIR = 'cache'
LOOK_BACK_DAY = 250


def bi_print(message, output_file):
    """Prints to both stdout and a file."""
    print(message)
    if output_file:
        output_file.write(message)
        output_file.write('\n')
        output_file.flush()


def simulate(start_year='2018', output_file=None):
    """Simulates trading operations and outputs gains."""
    tickers = get_all_symbols()
    sl = get_series_length(MAX_HISTORY_LOAD)
    dates = get_series_dates(MAX_HISTORY_LOAD)
    dir_path = os.path.dirname(os.path.realpath(__file__))
    all_series = {}
    print('Loading stock histories...')
    for ticker in tqdm(tickers, ncols=80, bar_format='{percentage:3.0f}%|{bar}{r_bar}', file=sys.stdout):
        cache_name = os.path.join(dir_path, CACHE_DIR, 'cache-%s.json' % (ticker,))
        if os.path.isfile(cache_name):
            with open(cache_name) as f:
                series_json = f.read()
            series = np.array(json.loads(series_json))
        else:
            series = get_series(ticker, time=MAX_HISTORY_LOAD)
            series_json = json.dumps(series.tolist())
            with open(cache_name, 'w') as f:
                f.write(series_json)
        if len(series) != sl:
            continue
        all_series[ticker] = series

    to_del = []
    for ticker, series in all_series.items():
        if np.max(series) * 0.7 > series[-1]:
            to_del.append(ticker)
    for ticker in to_del:
        del all_series[ticker]

    print('Num of picked stocks: %d' % (len(all_series)))

    total_return = 1.0
    start_point = 0
    while start_year not in dates[start_point]:
        start_point += 1
    for cutoff in range(start_point - 1, sl - 1):
        bi_print('=' * 80, output_file)
        bi_print('DATE: %s' % (dates[cutoff + 1],), output_file)
        buy_symbols = []
        for ticker, series in tqdm(all_series.items(), ncols=80, bar_format='{percentage:3.0f}%|{bar}{r_bar}',
                                   leave=False, file=sys.stdout):
            avg_return, is_buy = get_buy_signal(series[cutoff-LOOK_BACK_DAY:cutoff], series[cutoff])
            if is_buy:
                buy_symbols.append((avg_return, ticker))
        buy_symbols.sort(reverse=True)
        n_symbols = 0
        while n_symbols < min(10, len(buy_symbols)) and buy_symbols[n_symbols][0] >= 0.01:
            n_symbols += 1
        ac = 0
        for i in range(n_symbols):
            ac += buy_symbols[i][0]
        day_gain = 0
        trading_table = []
        for i in range(n_symbols):
            portion = 0.75 / n_symbols + 0.25 * buy_symbols[i][0] / ac
            ticker = buy_symbols[i][1]
            series = all_series[ticker]
            gain = (series[cutoff + 1] - series[cutoff]) / series[cutoff]
            trading_table.append([ticker, '%.2f%%' % (portion * 100,), '%.2f%%' % (gain * 100,)])
            day_gain += gain * portion
        if trading_table:
            bi_print(tabulate(trading_table, headers=['Symbol', 'Proportion', 'Gain'], tablefmt="grid"), output_file)
        bi_print('DAILY GAIN: %.2f%%' % (day_gain * 100,), output_file)
        total_return *= (1 + day_gain)
        bi_print('TOTAL GAIN: %.2f%%' % ((total_return - 1) * 100,), output_file)
    bi_print('=' * 80, output_file)


def main():
    output_filename = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'outputs', 'simulate.txt')
    with open(output_filename, 'w') as f:
        simulate(start_year='2018', output_file=f)
    # with open('output.data', 'w') as f:
    #  symbols = get_all_symbols()
    #  get_static_buy_symbols(symbols, f)


if __name__ == '__main__':
    main()
