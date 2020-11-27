import alpaca_trade_api as tradeapi
import alpaca_trade_api.polygon as polygonapi
import argparse
import datetime
import logging
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import os
import pandas as pd
import signal
import utils
from tabulate import tabulate


class TradingSimulate(utils.TradingBase):
    """Simulates trading transactions and outputs performances."""

    def __init__(self,
                 alpaca,
                 polygon,
                 start_date=None,
                 end_date=None):
        self.root_dir = os.path.dirname(os.path.realpath(__file__))
        self.output_dir = os.path.join(self.root_dir, utils.OUTPUTS_DIR,
                                       'simulate',
                                       datetime.datetime.now().strftime('%Y-%m-%d-%H-%M'))
        os.makedirs(self.output_dir, exist_ok=True)
        utils.logging_config(os.path.join(self.output_dir, 'result.txt'))

        super(TradingSimulate, self).__init__(alpaca, polygon, start_date=start_date, end_date=end_date)

        self.start_date = (start_date or
                           self.history_dates[utils.DAYS_IN_A_YEAR + 1].strftime('%F'))
        self.end_date = end_date or utils.get_business_day(1)
        self.start_point, self.end_point = 0, self.history_length - 1
        while (self.start_point < self.history_length and
               pd.to_datetime(self.start_date) > self.history_dates[self.start_point]):
            self.start_point += 1
        while (self.end_point > 0 and
               pd.to_datetime(self.end_date) < self.history_dates[self.end_point]):
            self.end_point -= 1
        self.values = {'Total': ([self.history_dates[self.start_point - 1]], [1.0])}
        self.win_trades, self.lose_trades = 0, 0
        signal.signal(signal.SIGINT, self.safe_exit)

    def safe_exit(self, signum, frame):
        logging.info('Safe exiting with signal %d...', signum)
        self.print_summary()
        exit(1)

    def analyze_date(self, sell_date, cutoff):
        outputs = [utils.get_header(sell_date.date())]
        trading_list = self.get_trading_list(cutoff=cutoff)
        trading_table = []
        daily_gain = 0
        for symbol, proportion, side in trading_list:
            if proportion == 0:
                continue
            close = self.closes[symbol]
            today_change = close[cutoff] / close[cutoff - 1] - 1
            if cutoff == self.history_length - 1:
                trading_table.append([symbol, side,
                                      utils.to_percent(proportion),
                                      utils.to_percent(today_change, True),
                                      close[cutoff]])
                continue
            gain = (close[cutoff + 1] - close[cutoff]) / close[cutoff]
            if side == 'short':
                gain *= -1
            # > 100% gain might caused by stock split. Do not calculate.
            if gain >= 1:
                continue
            if gain > 0:
                self.win_trades += 1
            elif gain < 0:
                self.lose_trades += 1
            trading_table.append([symbol, side,
                                  utils.to_percent(proportion),
                                  utils.to_percent(today_change, True),
                                  close[cutoff],
                                  close[cutoff + 1],
                                  utils.to_percent(gain, True)])
            daily_gain += gain * proportion

        if trading_table:
            outputs.append(tabulate(trading_table, headers=[
                'Symbol', 'Side', 'Proportion', 'Today Change',
                'Buy Price', 'Sell Price', 'Gain'], tablefmt='grid'))
        if cutoff < self.history_length - 1:
            self.add_profit(sell_date, daily_gain, outputs)
        else:
            logging.info('\n'.join(outputs))

    def add_profit(self, sell_date, daily_gain, outputs):
        """Adds daily gain to values memory."""
        total_value = self.values['Total'][1][-1] * (1 + daily_gain)
        self.values['Total'][0].append(sell_date)
        self.values['Total'][1].append(total_value)
        quarter = '%d-Q%d' % (sell_date.year,
                              (sell_date.month - 1) // 3 + 1)
        year = '%d' % (sell_date.year,)
        for t in [quarter, year]:
            if t not in self.values:
                self.values[t] = ([self.get_prev_market_date(sell_date)],
                                  [1.0])
            self.values[t][0].append(sell_date)
            t_value = self.values[t][1][-1] * (1 + daily_gain)
            self.values[t][1].append(t_value)
        summary_table = [['Daily Gain', '%+.2f%%' % (daily_gain * 100),
                          'Quarterly Gain', '%+.2f%%' % ((self.values[quarter][1][-1] - 1) * 100,),
                          'Yearly Gain', '%+.2f%%' % ((self.values[year][1][-1] - 1) * 100,),
                          'Total Gain', '%+.2f%%' % ((total_value - 1) * 100,)],
                         ['Win Trades', self.win_trades, 'Lose Trades', self.lose_trades]]
        outputs.append(tabulate(summary_table, tablefmt='grid'))
        logging.info('\n'.join(outputs))

    def print_summary(self):
        time_range = '%s ~ %s' % (self.start_date, self.end_date)
        summary_table = [['Time Range', time_range]]
        gain_texts = [(k + ' Gain', '%.2f%%' % ((v[1][-1] - 1) * 100,))
                      for k, v in self.values.items()]
        summary_table.extend(sorted(gain_texts))
        logging.info(utils.get_header('Summary') + '\n' + tabulate(summary_table, tablefmt='grid'))

    def plot_summary(self):
        pd.plotting.register_matplotlib_converters()
        plot_symbols = ['QQQ', 'SPY', 'TQQQ']
        color_map = {'QQQ': '#78d237', 'SPY': '#FF6358', 'TQQQ': '#aa46be'}
        for symbol in [utils.REFERENCE_SYMBOL] + plot_symbols:
            if symbol not in self.hists:
                try:
                    self.load_history(symbol)
                except Exception as e:
                    logging.exception('Can not download history of %s: %s', symbol, e)
        for k, v in self.values.items():
            dates, values = v
            if k == 'Total':
                formatter = mdates.DateFormatter('%Y-%m')
            else:
                formatter = mdates.DateFormatter('%m-%d')
            plt.figure(figsize=(10, 4))
            plt.plot(dates, values,
                     label='My Portfolio (%+.2f%%)' % ((values[-1] - 1) * 100,),
                     color='#28b4c8')
            curve_max = 1
            for symbol in plot_symbols:
                if symbol in self.hists:
                    curve = [self.hists[symbol].get('Close')[dt] for dt in dates]
                    for i in range(len(dates) - 1, -1, -1):
                        curve[i] /= curve[0]
                    curve_max = max(curve_max, np.abs(curve[-1]))
                    plt.plot(dates, curve,
                             label='%s (%+.2f%%)' % (symbol, (curve[-1] - 1) * 100),
                             color=color_map[symbol])
            text_kwargs = {'family': 'monospace'}
            plt.xlabel('Date', **text_kwargs)
            plt.ylabel('Normalized Value', **text_kwargs)
            plt.title(k, **text_kwargs, y=1.15)
            plt.grid(linestyle='--', alpha=0.5)
            plt.legend(ncol=len(plot_symbols) + 1, bbox_to_anchor=(0, 1),
                       loc='lower left', prop=text_kwargs)
            ax = plt.gca()
            ax.spines['right'].set_color('none')
            ax.spines['top'].set_color('none')
            ax.xaxis.set_major_formatter(formatter)
            if np.abs(values[-1]) > 5 * curve_max:
                plt.yscale('log')
            plt.tight_layout()
            plt.savefig(os.path.join(self.output_dir, k + '.png'))
            plt.close()

    def run(self):
        """Starts simulation."""
        # Buy on cutoff day, sell on cutoff + 1 day
        for cutoff in range(self.start_point - 1, self.end_point):
            sell_date = self.history_dates[cutoff + 1]
            self.analyze_date(sell_date, cutoff)
        if pd.to_datetime(self.end_date) > self.history_dates[-1]:
            self.analyze_date(self.history_dates[-1] + pd.tseries.offsets.BDay(1),
                              self.history_length - 1)

        self.print_summary()
        self.plot_summary()

    def get_prev_market_date(self, date):
        p = 0
        while date > self.history_dates[p]:
            p += 1
        return self.history_dates[p - 1]


def main():
    parser = argparse.ArgumentParser(description='Stock trading simulation.')
    parser.add_argument('--start_date', default=None,
                        help='Start date of the simulation.')
    parser.add_argument('--end_date', default=None,
                        help='End date of the simulation.')
    parser.add_argument('--api_key', default=None, help='Alpaca API key.')
    parser.add_argument('--api_secret', default=None, help='Alpaca API secret.')
    args = parser.parse_args()

    alpaca = tradeapi.REST(args.api_key or os.environ['ALPACA_PAPER_API_KEY'],
                           args.api_secret or os.environ['ALPACA_PAPER_API_SECRET'],
                           utils.ALPACA_PAPER_API_BASE_URL, 'v2')
    polygon = polygonapi.REST(args.api_key or os.environ['ALPACA_PAPER_API_KEY'])
    trading = TradingSimulate(alpaca, polygon, args.start_date, args.end_date)
    trading.run()


if __name__ == '__main__':
    main()
