import sys
from common import *
from tqdm import tqdm


def get_static_buy_symbols(tickers, output_file=None):
    sl = get_series_length('1y')
    buy_symbols = []
    for ticker in tqdm(tickers, bar_format='{percentage:3.0f}%|{bar}{r_bar}', file=sys.stdout):
        series = get_series(ticker, time='1y')
        if len(series) != sl:
            continue
        avg_return, is_buy = get_buy_signal(series[:-1], series[-1])
        if is_buy:
            buy_symbols.append((avg_return, ticker))

    output_file.write('%s, %.2f%%\n' % (ticker, avg_return * 100))
    output_file.flush()

def main():
    output_filename = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'outputs', 'run.txt')
    with open(output_filename, 'w') as f:
        symbols = get_all_symbols()
        get_static_buy_symbols(symbols, output_file=f)


if __name__ == '__main__':
    main()
