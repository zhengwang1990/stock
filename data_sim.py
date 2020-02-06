import pandas as pd
import os
import utils
from tabulate import tabulate


def predict(X):
    return [1] * len(X)


DATA_FILE = 'simulate_stats_2016_2019.csv'


def main():
    dir_path = os.path.dirname(os.path.realpath(__file__))
    df = pd.read_csv(os.path.join(dir_path, 'data', DATA_FILE))

    keys = [key for key, _ in df.iteritems() if key not in ('Gain', 'Symbol', 'Date')]

    prev_date = ''
    X = []
    symbols = []
    gains = []
    for row in df.itertuples():
        symbol = row.Symbol
        date = row.Date
        gain = row.Gain
        if date != prev_date and prev_date:
            # do something here
            trading_table = []
            print(utils.get_header(prev_date))
            weights = predict(X)
            for symbol, weight, gain in zip(symbols, weights, gains):
                trading_table.append([symbol, weight, gain])
            print(tabulate(
                trading_table,
                headers=['Symbol', 'Weight', 'Gain'],
                tablefmt='grid'))
            X, symbols = [], []
        x_value = [getattr(row, key) for key in keys]
        X.append(x_value)
        symbols.append(symbol)
        gains.append(gain)
        prev_date = date
    print('-' * 80)


if __name__ == '__main__':
    main()
