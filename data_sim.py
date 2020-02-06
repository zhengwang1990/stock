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
    gain_transactions, loss_transactions = 0, 0
    total_value = 1
    for row in df.itertuples():
        symbol = row.Symbol
        date = row.Date
        gain = row.Gain
        if date != prev_date and prev_date:
            # do something here
            trading_table = []
            print(utils.get_header(prev_date))
            weights = predict(X)
            buy_symbols = []
            for symbol, weight, gain in zip(symbols, weights, gains):
                buy_symbols.append((symbol, weight, gain))
            buy_symbols.sort(key=lambda s: s[1], reverse=True)
            n_symbols = min(utils.MAX_STOCK_PICK, len(buy_symbols))
            daily_gain = 0
            for i in range(len(buy_symbols)):
                symbol = buy_symbols[i][0]
                weight = buy_symbols[i][1]
                gain = buy_symbols[i][2]
                proportion = 1 / n_symbols if i < n_symbols else 0
                if proportion > 0:
                    trading_table.append([symbol, weight, '%.2f%%' % (gain * 100,)])
                    daily_gain += proportion * gain
                    if gain >= 0:
                        gain_transactions += 1
                    else:
                        loss_transactions += 1
            total_value *= 1 + daily_gain
            print(tabulate(
                trading_table,
                headers=['Symbol', 'Weight', 'Gain'],
                tablefmt='grid'))
            print('DAILY GAIN: %.2f%%, TOTAL GAIN: %.2f%%' % (
                daily_gain * 100, (total_value - 1) * 100))
            print('NUM GAIN TRANSACTIONS: %d, NUM LOSS TRANSACTIONS: %d, PRECISION: %.2f%%' % (
                gain_transactions, loss_transactions,
                gain_transactions / (gain_transactions +
                                     loss_transactions + 1E-7) * 100))
            X, symbols, gains = [], [], []
        x_value = [getattr(row, key) for key in keys]
        X.append(x_value)
        symbols.append(symbol)
        gains.append(gain)
        prev_date = date


if __name__ == '__main__':
    main()
