import alpaca_trade_api as tradeapi
import argparse
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
                 write_training_data=False):
        if start_date:
            year_diff = (pd.datetime.today().date().year -
                         pd.to_datetime(start_date).year + 2)
            year_diff = max(5, year_diff)
            period = '%dy' % (year_diff,)
        elif data_file:
            self.data_df = pd.read_csv(os.path.join(
                os.path.dirname(os.path.realpath(__file__)), utils.DATA_DIR, data_file))
            year_diff = (pd.datetime.today().date().year -
                         pd.to_datetime(self.data_df.iloc[0].Date).year + 1)
            period = '%dy' % (year_diff,)
        else:
            period = utils.DEFAULT_HISTORY_LOAD
        super(TradingSimulate, self).__init__(alpaca, period=period, model=model,
                                              load_history=not bool(data_file))
        self.data_file = data_file
        if data_file:
            self.values = {'Total': (
                [self._get_prev_market_date(pd.to_datetime(self.data_df.iloc[0].Date))],
                [1.0])}
        else:
            start_date = (start_date or
                          self.history_dates[5 * utils.DAYS_IN_A_YEAR + 1].date())
            end_date = end_date or pd.datetime.today().date()
            self.start_point, self.end_point = 0, self.history_length - 1
            while pd.to_datetime(start_date) > self.history_dates[self.start_point]:
                self.start_point += 1
            while pd.to_datetime(end_date) < self.history_dates[self.end_point]:
                self.end_point -= 1
            self.write_training_data = write_training_data
            if self.write_training_data:
                stats_cols = ['Symbol', 'Date'] + utils.ML_FEATURES + ['Gain']
                self.stats = pd.DataFrame(columns=stats_cols)
            self.values = {'Total': ([self.history_dates[self.start_point - 1]], [1.0])}
        self.output_detail = open(os.path.join(self.root_dir, utils.OUTPUTS_DIR,
                                               'simulate_detail.txt'), 'w')
        self.gain_transactions, self.loss_transactions = 0, 0
        signal.signal(signal.SIGINT, self.safe_exit)

    def safe_exit(self, signum, frame):
        print('\nSafe exiting with signal %d...' % (signum,))
        self._print_summary()
        exit(1)

    def _analyze_date(self, sell_date, cutoff):
        utils.bi_print(utils.get_header(sell_date.date()), self.output_detail)
        buy_symbols = self.get_buy_symbols(cutoff=cutoff)
        if self.write_training_data:
            self._append_stats(buy_symbols, sell_date, cutoff)
        trading_list = self.get_trading_list(buy_symbols=buy_symbols)
        trading_table = []
        daily_gain = 0
        for symbol, proportion, weight in trading_list:
            if proportion == 0:
                continue
            close = self.closes[symbol]
            gain = (close[cutoff + 1] - close[cutoff]) / close[cutoff]
            # > 100% gain might caused by stock split. Do not calculate.
            if gain >= 1:
                continue
            today_change = close[cutoff] / close[cutoff - 1] - 1
            day_range_change = close[cutoff] / np.max(close[cutoff - utils.DATE_RANGE:cutoff]) - 1
            trading_table.append([symbol, '%.2f%%' % (proportion * 100,),
                                  weight,
                                  '%.2f%%' % (today_change * 100,),
                                  '%.2f%%' % (day_range_change * 100,),
                                  '%.2f%%' % (gain * 100,)])
            daily_gain += gain * proportion
            if gain >= 0:
                self.gain_transactions += 1
            else:
                self.loss_transactions += 1
        if trading_table:
            utils.bi_print(tabulate(
                trading_table,
                headers=['Symbol', 'Proportion', 'Weight', 'Today Change',
                         '%d Day Change' % (utils.DATE_RANGE,), 'Gain'],
                tablefmt='grid'), self.output_detail)
        self._add_profit(sell_date, daily_gain)

    def _add_profit(self, sell_date, daily_gain):
        """Adds daily gain to values memory."""
        total_value = self.values['Total'][1][-1] * (1 + daily_gain)
        self.values['Total'][0].append(sell_date)
        self.values['Total'][1].append(total_value)
        quarter = '%d-Q%d' % (sell_date.year,
                              (sell_date.month - 1) // 3 + 1)
        year = '%d' % (sell_date.year,)
        for t in [quarter, year]:
            if t not in self.values:
                self.values[t] = ([self._get_prev_market_date(sell_date)],
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

    def _print_summary(self):
        output_summary = open(os.path.join(self.root_dir, utils.OUTPUTS_DIR,
                                           'simulate_summary.txt'), 'w')
        utils.bi_print(utils.get_header('Summary'), output_summary)
        if self.data_file:
            time_range = self.data_df.iloc[0].Date + '~' + self.data_df.iloc[-1].Date
        else:
            time_range = '%s ~ %s' % (
                self.history_dates[self.start_point].date(),
                self.history_dates[self.end_point].date())
        summary_table = [['Time Range', time_range]]
        gain_texts = [(k + ' Gain', '%.2f%%' % ((v[1][-1] - 1) * 100,))
                      for k, v in self.values.items()]
        summary_table.extend(sorted(gain_texts))
        utils.bi_print(tabulate(summary_table, tablefmt='grid'), output_summary)

        if not self.data_file and self.write_training_data:
            self.stats.to_csv(os.path.join(self.root_dir,
                                           utils.OUTPUTS_DIR,
                                           'simulate_stats.csv'),
                              index=False)

    def _plot_summary(self):
        pd.plotting.register_matplotlib_converters()
        for symbol in [utils.REFERENCE_SYMBOL, 'QQQ', 'SPY']:
            if symbol not in self.hists:
                try:
                    self.load_history(symbol, self.period)
                except Exception:
                    pass
        has_qqq = 'QQQ' in self.hists
        has_spy = 'SPY' in self.hists
        if has_qqq:
            qqq = self.hists.get('QQQ').get('Close')
        if has_spy:
            spy = self.hists.get('SPY').get('Close')
        for k, v in self.values.items():
            plt.figure(figsize=(15, 7))
            plt.plot(v[0], v[1], label='My Portfolio')
            qqq_curve = [qqq[dt] for dt in v[0]] if has_qqq else None
            spy_curve = [spy[dt] for dt in v[0]] if has_spy else None
            for i in range(len(v[0]) - 1, -1, -1):
                if has_qqq:
                    qqq_curve[i] /= qqq_curve[0]
                if has_spy:
                    spy_curve[i] /= spy_curve[0]
            if has_qqq:
                plt.plot(v[0], qqq_curve, label='QQQ')
            if has_spy:
                plt.plot(v[0], spy_curve, label='SPY')
            plt.legend()
            plt.title(k)
            if ((has_qqq and np.abs(v[1][-1]) > 10 * np.abs(qqq_curve[-1])) or
                (has_spy and np.abs(v[1][-1]) > 10 * np.abs(spy_curve[-1]))):
                plt.yscale('log')
            plt.savefig(os.path.join(self.root_dir, utils.OUTPUTS_DIR, k + '.png'))
            plt.close()

    def run(self):
        """Starts simulation."""
        # Buy on cutoff day, sell on cutoff + 1 day
        if self.data_file:
            rows = []
            prev_date = ''
            for row in self.data_df.itertuples():
                current_date = row.Date
                if current_date != prev_date and prev_date:
                    self._analyze_rows(prev_date, rows)
                    rows = []
                rows.append(row)
                prev_date = current_date
            self._analyze_rows(prev_date, rows)
        else:
            for cutoff in range(self.start_point - 1, self.end_point):
                sell_date = self.history_dates[cutoff + 1]
                self._analyze_date(sell_date, cutoff)

        self._print_summary()
        self._plot_summary()

    def _analyze_rows(self, sell_date_str, rows):
        print(utils.get_header(sell_date_str))
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
        self._add_profit(pd.to_datetime(sell_date_str), daily_gain)

    def _append_stats(self, buy_symbols, date, cutoff):
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

    def _get_prev_market_date(self, date):
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
    parser.add_argument("--write_training_data", help='Write training data.',
                        action="store_true")
    args = parser.parse_args()

    alpaca = tradeapi.REST(args.api_key or os.environ['ALPACA_PAPER_API_KEY'],
                           args.api_secret or os.environ['ALPACA_PAPER_API_SECRET'],
                           utils.ALPACA_PAPER_API_BASE_URL, 'v2')
    trading = TradingSimulate(alpaca, args.start_date, args.end_date,
                              args.model, args.data_file,
                              args.write_training_data)
    trading.run()


if __name__ == '__main__':
    main()
