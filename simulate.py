import alpaca_trade_api as tradeapi
import argparse
import datetime
import matplotlib.pyplot as plt
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
                 start_date=None,
                 end_date=None,
                 model=None,
                 data_file=None,
                 write_data=False):
        period = None
        if data_file:
            self.data_df = pd.read_csv(os.path.join(
                os.path.dirname(os.path.realpath(__file__)), utils.DATA_DIR, data_file))
            year_diff = (datetime.datetime.today().date().year -
                         pd.to_datetime(self.data_df.iloc[0].Date).year + 1)
            period = '%dy' % (year_diff,)
        if start_date:
            year_diff = (datetime.datetime.today().date().year -
                         pd.to_datetime(start_date).year + 2)
            year_diff = max(5, year_diff)
            period = '%dy' % (year_diff,)
        if not (data_file or start_date):
            period = utils.DEFAULT_HISTORY_LOAD
        super(TradingSimulate, self).__init__(alpaca, period=period, model=model,
                                              load_history=not bool(data_file))
        self.data_file = data_file
        if self.data_file:
            self.start_date = start_date or self.data_df.iloc[0].Date
            self.end_date = end_date or self.data_df.iloc[-1].Date
            self.values = {'Total': (
                [self.get_prev_market_date(pd.to_datetime(self.start_date))],
                [1.0])}
        else:
            self.start_date = (start_date or
                               self.history_dates[utils.DAYS_IN_A_YEAR + 1].strftime('%F'))
            self.end_date = end_date or datetime.datetime.today().strftime('%F')
            self.start_point, self.end_point = 0, self.history_length - 1
            while (self.start_point < self.history_length and
                   pd.to_datetime(self.start_date) > self.history_dates[self.start_point]):
                self.start_point += 1
            while (self.end_point > 0 and
                   pd.to_datetime(self.end_date) < self.history_dates[self.end_point]):
                self.end_point -= 1
            self.write_data = write_data
            if self.write_data:
                stats_cols = ['Symbol', 'Date'] + utils.ML_FEATURES + ['Gain']
                self.stats = pd.DataFrame(columns=stats_cols)
            self.values = {'Total': ([self.history_dates[self.start_point - 1]], [1.0])}
        self.output_detail = open(os.path.join(self.root_dir, utils.OUTPUTS_DIR,
                                               'simulate_detail.txt'), 'w')
        self.gain_transactions, self.loss_transactions = 0, 0
        signal.signal(signal.SIGINT, self.safe_exit)

    def safe_exit(self, signum, frame):
        print('\nSafe exiting with signal %d...' % (signum,))
        self.print_summary()
        exit(1)

    def analyze_date(self, sell_date, cutoff):
        utils.bi_print(utils.get_header(sell_date.date()), self.output_detail)
        buy_symbols = self.get_buy_symbols(cutoff=cutoff)
        if self.write_data and cutoff < self.history_length - 1:
            self.append_stats(buy_symbols, sell_date, cutoff)
        trading_list = self.get_trading_list(buy_symbols=buy_symbols)
        trading_table = []
        daily_gain = 0
        for symbol, proportion, weight in trading_list:
            if proportion == 0:
                continue
            close = self.closes[symbol]
            today_change = close[cutoff] / close[cutoff - 1] - 1
            day_range_change = close[cutoff] / np.max(close[cutoff - utils.DATE_RANGE:cutoff]) - 1
            if cutoff == self.history_length - 1:
                trading_table.append([symbol, '%.2f%%' % (proportion * 100,),
                                      weight,
                                      '%.2f%%' % (today_change * 100,),
                                      '%.2f%%' % (day_range_change * 100,),
                                      close[cutoff]])
                continue
            gain = (close[cutoff + 1] - close[cutoff]) / close[cutoff]
            # > 100% gain might caused by stock split. Do not calculate.
            if gain >= 1:
                continue
            trading_table.append([symbol, '%.2f%%' % (proportion * 100,),
                                  weight,
                                  '%.2f%%' % (today_change * 100,),
                                  '%.2f%%' % (day_range_change * 100,),
                                  close[cutoff],
                                  close[cutoff + 1],
                                  '%.2f%%' % (gain * 100,)])
            daily_gain += gain * proportion
            if gain >= 0:
                self.gain_transactions += 1
            else:
                self.loss_transactions += 1
        if trading_table:
            utils.bi_print(
                tabulate(trading_table, headers=[
                    'Symbol', 'Proportion', 'Weight', 'Today Change',
                    '%d Day Change' % (utils.DATE_RANGE,), 'Buy Price',
                    'Sell Price', 'Gain'], tablefmt='grid'),
                self.output_detail)
        if cutoff < self.history_length - 1:
            self.add_profit(sell_date, daily_gain)

    def analyze_rows(self, sell_date_str, rows):
        utils.bi_print(utils.get_header(sell_date_str), self.output_detail)
        ml_features, symbols, gains = [], [], {}
        for row in rows:
            ml_features.append([getattr(row, key) for key in utils.ML_FEATURES])
            symbols.append(row.Symbol)
            gains[row.Symbol] = row.Gain
        ml_features = np.array(ml_features)
        weights = self.model.predict(ml_features)
        buy_symbols = [(symbol, weight) for symbol, weight in zip(symbols, weights)]
        trading_list = self.get_trading_list(buy_symbols=buy_symbols)
        trading_table = []
        daily_gain = 0
        for symbol, proportion, weight in trading_list:
            if proportion == 0:
                continue
            gain = gains[symbol]
            # > 100% gain might caused by stock split. Do not calculate.
            if gain >= 1:
                continue
            trading_table.append([symbol, '%.2f%%' % (proportion * 100,),
                                  weight,
                                  '%.2f%%' % (gain * 100,)])
            daily_gain += gain * proportion
            if gain >= 0:
                self.gain_transactions += 1
            else:
                self.loss_transactions += 1
        if trading_table:
            utils.bi_print(tabulate(
                trading_table,
                headers=['Symbol', 'Proportion', 'Weight', 'Gain'],
                tablefmt='grid'), self.output_detail)
        self.add_profit(pd.to_datetime(sell_date_str), daily_gain)

    def add_profit(self, sell_date, daily_gain):
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
        utils.bi_print('DAILY GAIN: %.2f%%, TOTAL GAIN: %.2f%%' % (
            daily_gain * 100, (total_value - 1) * 100), self.output_detail)
        utils.bi_print('NUM GAIN TRANSACTIONS: %d, NUM LOSS TRANSACTIONS: %d, PRECISION: %.2f%%' % (
            self.gain_transactions, self.loss_transactions,
            self.gain_transactions / (self.gain_transactions +
                                      self.loss_transactions + 1E-7) * 100),
                       self.output_detail)

    def print_summary(self):
        output_summary = open(os.path.join(self.root_dir, utils.OUTPUTS_DIR,
                                           'simulate_summary.txt'), 'w')
        utils.bi_print(utils.get_header('Summary'), output_summary)
        time_range = '%s ~ %s' % (self.start_date, self.end_date)
        summary_table = [['Time Range', time_range]]
        gain_texts = [(k + ' Gain', '%.2f%%' % ((v[1][-1] - 1) * 100,))
                      for k, v in self.values.items()]
        summary_table.extend(sorted(gain_texts))
        utils.bi_print(tabulate(summary_table, tablefmt='grid'), output_summary)

        if not self.data_file and self.write_data:
            self.stats.to_csv(
                os.path.join(self.root_dir,
                             utils.OUTPUTS_DIR,
                             'simulate_stats_%s_%s.csv' % (self.start_date[:4],
                                                           self.end_date[:4])),
                index=False)

    def plot_summary(self):
        pd.plotting.register_matplotlib_converters()
        plot_symbols = ['QQQ', 'SPY', 'TQQQ']
        color_map = {'QQQ': '#78d237', 'SPY': '#FF6358', 'TQQQ': '#aa46be'}
        for symbol in [utils.REFERENCE_SYMBOL] + plot_symbols:
            if symbol not in self.hists:
                try:
                    self.load_history(symbol, self.period)
                except Exception:
                    pass
        for k, v in self.values.items():
            dates, values = v
            if k == 'Total':
                dates_str = [date.strftime('%Y-%m-%d') for date in dates]
                unit = max(len(dates_str) // 4, 1)
            else:
                dates_str = [date.strftime('%m-%d') for date in dates]
                dates_str[0] = dates[0].strftime('%Y-%m-%d')
                unit = max(len(dates_str) // 6, 1)
            plt.figure(figsize=(10, 4))
            plt.plot(dates_str, values,
                     label='My Portfolio (%+.2f%%)' % ((values[-1] - 1) * 100,),
                     color='#28b4c8')
            curve_max = 1
            for symbol in plot_symbols:
                if symbol in self.hists:
                    curve = [self.hists[symbol].get('Close')[dt] for dt in dates]
                    for i in range(len(dates) - 1, -1, -1):
                        curve[i] /= curve[0]
                    curve_max = max(curve_max, np.abs(curve[-1]))
                    plt.plot(dates_str, curve,
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
            ax.set_xticks([dates_str[1]] + dates_str[unit:-unit+1:unit] + [dates_str[-1]])
            if np.abs(values[-1]) > 5 * curve_max:
                plt.yscale('log')
            plt.tight_layout()
            plt.savefig(os.path.join(self.root_dir, utils.OUTPUTS_DIR, 'plots', k + '.png'))
            plt.close()

    def run(self):
        """Starts simulation."""
        # Buy on cutoff day, sell on cutoff + 1 day
        if self.data_file:
            rows = []
            prev_date = ''
            for row in self.data_df.itertuples():
                current_date = row.Date
                if current_date < self.start_date or current_date > self.end_date:
                    continue
                if current_date != prev_date and prev_date:
                    self.analyze_rows(prev_date, rows)
                    rows = []
                rows.append(row)
                prev_date = current_date
            self.analyze_rows(prev_date, rows)
        else:
            for cutoff in range(self.start_point - 1, self.end_point):
                sell_date = self.history_dates[cutoff + 1]
                self.analyze_date(sell_date, cutoff)
            if pd.to_datetime(self.end_date) > self.history_dates[-1]:
                self.analyze_date(self.history_dates[-1] + pd.tseries.offsets.BDay(1),
                                  self.history_length - 1)

        self.print_summary()
        self.plot_summary()

    def append_stats(self, buy_symbols, date, cutoff):
        for symbol, _, ml_feature in buy_symbols:
            close = self.closes[symbol]
            gain = (close[cutoff + 1] - close[cutoff]) / close[cutoff]
            # > 100% gain might caused by stock split. Do not calculate.
            if gain >= 1:
                continue
            stat_value = ml_feature
            stat_value['Symbol'] = symbol
            stat_value['Date'] = date
            stat_value['Gain'] = gain
            self.stats = self.stats.append(stat_value, ignore_index=True)

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
    parser.add_argument('--model', default=None, help='Keras model for weight prediction.')
    parser.add_argument('--data_file', default=None, help='Read datafile for simulation.')
    parser.add_argument("--write_data", help='Write data with ML features.',
                        action="store_true")
    args = parser.parse_args()

    alpaca = tradeapi.REST(args.api_key or os.environ['ALPACA_PAPER_API_KEY'],
                           args.api_secret or os.environ['ALPACA_PAPER_API_SECRET'],
                           utils.ALPACA_PAPER_API_BASE_URL, 'v2')
    trading = TradingSimulate(alpaca, args.start_date, args.end_date,
                              args.model, args.data_file,
                              args.write_data)
    trading.run()


if __name__ == '__main__':
    main()
