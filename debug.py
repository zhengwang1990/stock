import matplotlib.pyplot as plt
from common import *


def plot_buy_points(ticker):
    series = get_series(ticker)
    pick_t, avg_return, threshold = get_picked_points(series)
    print(get_header(ticker))
    print('Average return: %.2f%%' % (avg_return * 100,))
    print('Threshold: %.2f%%' % (threshold * 100,))
    t = range(len(series))
    plt.plot(t, series)
    plt.plot(pick_t, series[pick_t], 'o')
    plt.show()


def main():
    plot_buy_points('SPY')


if __name__ == '__main__':
    main()
