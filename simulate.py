import argparse
import matplotlib.pyplot as plt
from common import *
from tabulate import tabulate


def simulate(start_date=None, end_date=None):
    """Simulates trading operations and outputs gains."""
    file_dir = os.path.dirname(os.path.realpath(__file__))
    output_detail = open(os.path.join(file_dir, 'outputs', 'simulate_detail.txt'), 'w')
    output_summary = open(os.path.join(file_dir, 'outputs', 'simulate_summary.txt'), 'w')

    dates = get_series_dates(MAX_HISTORY_LOAD)
    series_length = len(dates)
    all_series = filter_low_volume_series(
        filter_garbage_series(get_all_series(MAX_HISTORY_LOAD)))

    start_date = start_date or dates[LOOK_BACK_DAY+1].date()
    end_date = end_date or pd.datetime.today().date()
    start_point, end_point = 0, series_length - 1
    while pd.to_datetime(start_date) > dates[start_point]:
        start_point += 1
    if start_point - 1 < LOOK_BACK_DAY:
        raise Exception('Start date must be no early than %s' % (dates[LOOK_BACK_DAY+1].date()))
    while pd.to_datetime(end_date) < dates[end_point]:
        end_point -= 1
    values = {'Total': ([dates[start_point-1]], [1.0])}
    for cutoff in range(start_point - 1, end_point):
        current_date = dates[cutoff + 1]
        bi_print(get_header(current_date.date()), output_detail)
        buy_symbols = get_buy_symbols(all_series, cutoff)
        trading_list = get_trading_list(buy_symbols)
        trading_table = []
        day_gain = 0
        for ticker, proportion in trading_list:
            series = all_series[ticker]
            gain = (series[cutoff + 1] - series[cutoff]) / series[cutoff]
            trading_table.append([ticker, '%.2f%%' % (proportion * 100,), '%.2f%%' % (gain * 100,)])
            day_gain += gain * proportion
        if trading_table:
            bi_print(tabulate(trading_table, headers=['Symbol', 'Proportion', 'Gain'], tablefmt='grid'), output_detail)
        bi_print('DAILY GAIN: %.2f%%' % (day_gain * 100,), output_detail)
        total_value = values['Total'][1][-1] * (1 + day_gain)
        values['Total'][0].append(current_date)
        values['Total'][1].append(total_value)
        current_quarter = '%d-Q%d' % (current_date.year, (current_date.month - 1) // 3 + 1)
        current_year = '%d' % (current_date.year,)
        for current_t in [current_quarter, current_year]:
            if current_t not in values:
                values[current_t] = ([dates[cutoff]], [1.0])
            values[current_t][0].append(current_date)
            t_value = values[current_t][1][-1] * (1 + day_gain)
            values[current_t][1].append(t_value)
        bi_print('TOTAL GAIN: %.2f%%' % ((total_value - 1) * 100,), output_detail)

    bi_print(get_header('Summary'), output_summary)
    summary_table = [['Time Range', '%s ~ %s' % (dates[start_point].date(), dates[end_point].date())]]
    pd.plotting.register_matplotlib_converters()
    qqq = get_series('QQQ', time=MAX_HISTORY_LOAD)[1]
    spy = get_series('SPY', time=MAX_HISTORY_LOAD)[1]
    for k, v in values.items():
        plt.figure(figsize=(15, 7))
        plt.plot(v[0], v[1], label='My Portfolio')
        qqq_curve = [qqq.get(dt) for dt in v[0]]
        spy_curve = [spy.get(dt) for dt in v[0]]
        for i in range(len(v[0])-1, -1, -1):
            qqq_curve[i] /= qqq_curve[0]
            spy_curve[i] /= spy_curve[0]
        plt.plot(v[0], qqq_curve, label='QQQ')
        plt.plot(v[0], spy_curve, label='SPY')
        plt.legend()
        plt.title(k)
        if np.abs(v[1][-1]) > 10 * np.abs(qqq_curve[-1]):
            plt.yscale('log')
        plt.savefig(os.path.join(file_dir, 'outputs', k + '.png'))
    gain_texts = [(k + ' Gain',  '%.2f%%' % ((v[1][-1] - 1) * 100,)) for k, v in values.items()]
    summary_table.extend(sorted(gain_texts))
    bi_print(tabulate(summary_table, tablefmt='grid'), output_summary)


def main():
    parser = argparse.ArgumentParser(description='Stock trading strategy.')
    parser.add_argument('--start_date', default=None, help='Start date of the simulation.')
    parser.add_argument('--end_date', default=None, help='End date of the simulation.')
    args = parser.parse_args()
    simulate(args.start_date, args.end_date)


if __name__ == '__main__':
    main()
