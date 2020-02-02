import alpaca_trade_api as tradeapi
import argparse
import matplotlib.pyplot as plt
import ml
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
               model_name=None,
               write_training_data=False):
    if start_date:
      year_diff = (pd.datetime.today().date().year -
                   pd.to_datetime(start_date).year + 2)
      year_diff = max(5, year_diff)
      period = '%dy' % (year_diff,)
    else:
      period = utils.DEFAULT_HISTORY_LOAD
    super(TradingSimulate, self).__init__(alpaca, period=period)
    self.model = ml.load_model(model_name) if model_name else None

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
    self.output_detail = open(os.path.join(self.root_dir, utils.OUTPUTS_DIR,
                                           'simulate_detail.txt'), 'w')
    self.values = {'Total': ([self.history_dates[self.start_point - 1]], [1.0])}
    self.gain_transcations, self.loss_transactions = 0, 0
    signal.signal(signal.SIGINT, self.safe_exit)

  def safe_exit(self, signum, frame):
    print('\nSafe exiting with signal %d...' % (signum,))
    self._print_summary()
    exit(1)

  def _step(self, date, cutoff):
    utils.bi_print(utils.get_header(date.date()), self.output_detail)
    buy_symbols = self.get_buy_symbols(cutoff=cutoff, model=self.model)
    if self.write_training_data:
      self._append_stats(buy_symbols, date, cutoff)
    trading_list = self.get_trading_list(buy_symbols=buy_symbols)
    trading_table = []
    daily_gain = 0
    for ticker, proportion, weight in trading_list:
      if proportion == 0:
        continue
      series = self.closes[ticker]
      gain = (series[cutoff + 1] - series[cutoff]) / series[cutoff]
      # > 100% gain might caused by stock split. Do not calculate.
      if gain >= 1:
        continue
      trading_table.append([ticker, '%.2f%%' % (proportion * 100,), weight,
                            '%.2f%%' % (gain * 100,)])
      daily_gain += gain * proportion
      if gain >= 0:
        self.gain_transcations += 1
      else:
        self.loss_transactions += 1
    if trading_table:
      utils.bi_print(tabulate(trading_table,
                              headers=['Symbol', 'Proportion', 'Weight', 'Gain'],
                              tablefmt='grid'),
               self.output_detail)
    return daily_gain


  def _add_profit(self, cutoff, date, daily_gain):
    """Adds dailly gain to values memory."""
    total_value = self.values['Total'][1][-1] * (1 + daily_gain)
    self.values['Total'][0].append(date)
    self.values['Total'][1].append(total_value)
    quarter = '%d-Q%d' % (date.year,
                          (date.month - 1) // 3 + 1)
    year = '%d' % (date.year,)
    for t in [quarter, year]:
      if t not in self.values:
        self.values[t] = ([self.history_dates[cutoff]], [1.0])
      self.values[t][0].append(date)
      t_value = self.values[t][1][-1] * (1 + daily_gain)
      self.values[t][1].append(t_value)
    utils.bi_print('DAILY GAIN: %.2f%%, TOTAL GAIN: %.2f%%' % (
        daily_gain * 100, (total_value - 1) * 100), self.output_detail)
    utils.bi_print('NUM GAIN TRANSACATIONS: %d, NUM LOSS TRANSACATIONS: %d, PRECISION: %.2f%%' % (
        self.gain_transcations, self.loss_transactions,
        self.gain_transcations / (self.gain_transcations +
                                  self.loss_transactions + 1E-7) * 100),
             self.output_detail)


  def _print_summary(self):
    output_summary = open(os.path.join(self.root_dir, utils.OUTPUTS_DIR,
                                       'simulate_summary.txt'), 'w')
    utils.bi_print(utils.get_header('Summary'), output_summary)
    summary_table = [['Time Range', '%s ~ %s' % (
        self.history_dates[self.start_point].date(),
        self.history_dates[self.end_point].date())]]
    gain_texts = [(k + ' Gain', '%.2f%%' % ((v[1][-1] - 1) * 100,))
                  for k, v in self.values.items()]
    summary_table.extend(sorted(gain_texts))
    utils.bi_print(tabulate(summary_table, tablefmt='grid'), output_summary)

    if self.write_training_data:
      self.stats.to_csv(os.path.join(self.root_dir,
                                     utils.OUTPUTS_DIR,
                                     'simulate_stats.csv'),
                        index=False)

  def _plot_summary(self):
    pd.plotting.register_matplotlib_converters()
    qqq = self.hists['QQQ'].get('Close')
    spy = self.hists['SPY'].get('Close')
    for k, v in self.values.items():
      plt.figure(figsize=(15, 7))
      plt.plot(v[0], v[1], label='My Portfolio')
      qqq_curve = [qqq.get(dt) for dt in v[0]]
      spy_curve = [spy.get(dt) for dt in v[0]]
      for i in range(len(v[0]) - 1, -1, -1):
        qqq_curve[i] /= qqq_curve[0]
        spy_curve[i] /= spy_curve[0]
      plt.plot(v[0], qqq_curve, label='QQQ')
      plt.plot(v[0], spy_curve, label='SPY')
      plt.legend()
      plt.title(k)
      if np.abs(v[1][-1]) > 10 * np.abs(qqq_curve[-1]):
        plt.yscale('log')
      plt.savefig(os.path.join(self.root_dir, utils.OUTPUTS_DIR, k + '.png'))
      plt.close()

  def run(self):
    """Starts simulation."""
    # Buy on cutoff day, sell on cutoff + 1 day
    for cutoff in range(self.start_point - 1, self.end_point):
      current_date = self.history_dates[cutoff + 1]
      daily_gain = self._step(current_date, cutoff)
      self._add_profit(cutoff, current_date, daily_gain)

    self._print_summary()
    self._plot_summary()

  def _append_stats(self, buy_symbols, date, cutoff):
    for symbol, _, ml_feature in buy_symbols:
      series = self.closes[symbol]
      gain = (series[cutoff + 1] - series[cutoff]) / series[cutoff]
      # > 100% gain might caused by stock split. Do not calculate.
      if gain >= 1:
        continue
      stat_value = ml_feature
      stat_value['Symbol'] = symbol
      stat_value['Date'] = date
      stat_value['Gain'] = gain
      self.stats = self.stats.append(stat_value, ignore_index=True)


def main():
  parser = argparse.ArgumentParser(description='Stock trading simulation.')
  parser.add_argument('--start_date', default=None,
                      help='Start date of the simulation.')
  parser.add_argument('--end_date', default=None,
                      help='End date of the simulation.')
  parser.add_argument('--model', default='model_p573991.hdf5',
                      help='Machine learning model for prediction.')
  parser.add_argument('--api_key', default=None, help='Alpaca API key.')
  parser.add_argument('--api_secret', default=None, help='Alpaca API secret.')
  parser.add_argument("--write_training_data", help='Write tranining data.',
                      action="store_true")
  args = parser.parse_args()

  alpaca = tradeapi.REST(args.api_key or os.environ['ALPACA_API_KEY'],
                         args.api_secret or os.environ['ALPACA_API_SECRET'],
                         utils.APCA_API_BASE_URL, 'v2')
  trading = TradingSimulate(alpaca, args.start_date, args.end_date, args.model,
                            args.write_training_data)
  trading.run()


if __name__ == '__main__':
  main()
