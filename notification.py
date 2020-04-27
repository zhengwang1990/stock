import alpaca_trade_api as tradeapi
import alpaca_trade_api.polygon as polygonapi
import argparse
import collections
import datetime
import numpy as np
import os
import smtplib
import textwrap
import utils
import yfinance as yf
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

Order = collections.namedtuple('Order', ['price', 'qty', 'value'])


def _create_server(user, password):
    server = smtplib.SMTP('smtp.live.com', 587)
    server.starttls()
    server.ehlo()
    server.login(user, password)
    return server


def _create_message(sender, receiver):
    message = MIMEMultipart("alternative")
    message['From'] = sender
    message['To'] = receiver
    return message


def _get_trade_info(orders, side):
    trade_info = {}
    for order in orders:
        symbol = order.symbol
        filled_qty = int(order.filled_qty)
        if order.side == side and filled_qty > 0:
            old = trade_info.get(symbol, Order(0, 0, 0))
            new_qty = filled_qty + old.qty
            new_value = float(order.filled_avg_price) * filled_qty + old.value
            new_price = new_value / new_qty
            trade_info[symbol] = Order(new_price, new_qty, new_value)
    return trade_info


def _get_trading_history(historical_value, v_min, v_scale, color, line_width):
    trading_history = ''
    for i, value in enumerate(historical_value):
        y_pos = (value - v_min) / v_scale + 15
        trading_history += ('<li style="--x: %dvw; --y: %fvw; --color: %s">'
                            '<div class="data-point">') % (i + 1, y_pos, color)
        if i > 0:
            daily_gain = (value / historical_value[i - 1] - 1) * 100
            trading_history += '<span class="tooltip" style="color:%s">%.2f (%+.2f%%)</span>' % (
                'green' if daily_gain >= 0 else 'red', value, daily_gain)
        else:
            trading_history += '<span class="tooltip">1.0</span>'
        trading_history += '</div>'
        if i < len(historical_value) - 1:
            y_next_pos = (historical_value[i + 1] - v_min) / v_scale + 15
            trading_history += ('<div class="line-segment" style="--width: %fvw; height: %dpx; '
                                '--angle:%frad;"></div></li>') % (
                                   (6.4 ** 2 + (0.36 * (y_pos - y_next_pos)) ** 2) ** 0.5,
                                   line_width,
                                   np.arctan(0.36 * (y_pos - y_next_pos) / 6.4))
    return trading_history


def send_summary(sender, receiver, bcc, user, password, alpaca, polygon):
    calendar = alpaca.get_calendar(start=datetime.date.today() - datetime.timedelta(days=30),
                                   end=datetime.date.today())
    open_dates = sorted([c.date for c in calendar], reverse=True)
    if open_dates[0].strftime('%F') != datetime.date.today().strftime('%F'):
        print('Today is not a trading day')
    #    return

    server = _create_server(user, password)
    message = _create_message(sender, receiver)
    message['Subject'] = '[Summary] [%s] Trade summary of the day' % (
        datetime.date.today(),)
    orders = alpaca.list_orders(status='closed', after=open_dates[0])
    prev_orders = alpaca.list_orders(status='closed', after=open_dates[1], until=open_dates[0])
    buys = _get_trade_info(orders, 'buy')
    sells = _get_trade_info(orders, 'sell')
    prev_buys = _get_trade_info(prev_orders, 'buy')
    sell_text, sell_html = '', ''
    total_gain = 0
    for symbol, sell_info in sells.items():
        if symbol not in prev_buys:
            continue
        buy_info = prev_buys[symbol]
        gain = sell_info.value - buy_info.value
        percent = sell_info.price / buy_info.price - 1
        sell_text += '%s: buy at %g, sell at %g, quantity %d, gain/loss %+.2f (%+.2f%%)\n' % (
            symbol, buy_info.price, sell_info.price, sell_info.qty, gain, percent * 100)
        sell_html += ('<tr> <th scope="row">%s</th> <td>%g</td> <td>%g</td> <td>%d</td> '
                      '<td style="color:%s;">%+.2f (%+.2f%%)</td> </tr>') % (
                         symbol, buy_info.price, sell_info.price, sell_info.qty,
                         'green' if gain >= 0 else 'red', gain, percent * 100)
        total_gain += gain
    buy_text, buy_html = '', ''
    for symbol, buy_info in buys.items():
        buy_text += '%s: buy at %g, quantity %d, cost %.2f\n' % (
            symbol, buy_info.price, buy_info.qty, buy_info.value)
        buy_html += '<tr> <th scope="row">%s</th> <td>%g</td> <td>%d</td> <td>%.2f</td> </tr>' % (
            symbol, buy_info.price, buy_info.qty, buy_info.value)
    account = alpaca.get_account()
    account_equity = float(account.equity)
    account_text = 'Equity: %.2f\nCash: %s\nGain / Loss: %+.2f (%+.2f%%)\n' % (
        account_equity, account.cash, total_gain, total_gain / (account_equity - total_gain) * 100)
    account_html = ('<tr><th scope="row" class="narrow-col">Equity</th><td>%.2f</td></tr>'
                    '<tr><th scope="row" class="narrow-col">Cash</th><td>%s</td></tr>'
                    '<tr><th scope="row" class="narrow-col">Gain / Loss</th>'
                    '<td style="color:%s;">%+.2f (%+.2f%%)</td></tr>\n') % (
                       account_equity, account.cash, 'green' if total_gain >= 0 else 'red', total_gain,
                       total_gain / (account_equity - total_gain) * 100)

    history = alpaca.get_portfolio_history(date_start=open_dates[10].strftime('%F'),
                                           date_end=open_dates[1].strftime('%F'),
                                           timeframe='1D')
    historical_value = [equity / history.equity[0] for equity in history.equity]
    historical_value.append(account_equity / history.equity[0])
    historical_date = [datetime.datetime.fromtimestamp(timestamp).strftime('%m-%d')
                       for timestamp in history.timestamp]
    historical_date.append(open_dates[0].strftime('%m-%d'))
    historical_values = {}

    last_prices = {}
    for symbol, color in [('SPY', '#ffb04f'), ('QQQ', '#d2b8ff')]:
        ref_historical_value = yf.Ticker(symbol).history(start=open_dates[10].strftime('%F'),
                                                         end=open_dates[0].strftime('%F'),
                                                         interval='1d').get('Close')
        last_prices[symbol] = polygon.last_trade(symbol).price
        ref_historical_value = np.append(np.array(ref_historical_value),
                                         last_prices[symbol])
        for i in range(len(ref_historical_value) - 1, -1, -1):
            ref_historical_value[i] /= ref_historical_value[0]
        if len(historical_value) == len(historical_date) == len(ref_historical_value):
            historical_values[symbol] = (color, ref_historical_value)
    historical_values['My Portfolio'] = ('#276ecc', historical_value)

    v_min, v_max = 1E9, 0
    for _, value in historical_values.values():
        v_max = max(v_max, max(value))
        v_min = min(v_min, min(value))
    v_scale = (v_max - v_min) / 72
    trading_history_html = ''
    line_width = 3
    for symbol in ['SPY', 'QQQ', 'My Portfolio']:
        if symbol not in historical_values:
            continue
        color, value = historical_values[symbol]
        trading_history_html += _get_trading_history(value, v_min, v_scale, color, line_width)
    for i, symbol in enumerate(['My Portfolio', 'QQQ', 'SPY']):
        if symbol not in historical_values:
            continue
        color, _ = historical_values[symbol]
        trading_history_html += ('<li style="--y: 95vw; --color: %s;">'
                                 '<div class="data-point" style="--x: %fvw;"></div>'
                                 '<div class="line-segment" style="--x: %dvw; --width: 6vw; '
                                 '--angle:0; height: %dpx;"></div>'
                                 '<div class="legend" style="--x: %dvw;">%s</div></li>\n') % (
                                    color, i * 4 + 1.46875, i * 4 + 1, line_width, i * 4 + 2, symbol)

    grid_html = ''
    for i, date in enumerate(historical_date[:len(historical_value)]):
        grid_html += '<div class="ticker x-ticker%s" style="--x: %dvw;">%s</div>\n' % (
            ' hidden-ticker' if i % 2 == 1 else '', i + 1, date)
    for i in range(5):
        grid_html += ('<div style="--y: %dvw;"><div class ="ticker y-ticker">%.2f</div>'
                      '<div class="dashed-line-segment"></div></div>\n') % (
                         15 + 18 * i, v_min + (v_max - v_min) / 4 * i)

    account_html += ('<tr><th scope="row" class="narrow-col">Market Change</th>'
                     '<td style="padding: 0px;"><table>')
    for symbol in ['QQQ', 'SPY']:
        if symbol not in historical_values:
            continue
        _, value = historical_values[symbol]
        gain = (value[-1] / value[-2] - 1) * 100
        account_html += ('<tr><td style="border: none; padding: 0.5rem;">%s </td>'
                         '<td style="border: none; padding: 0.5rem;">'
                         '%.2f <span style="color:%s;">(%+.2f%%)</span></td></tr>') % (
                            symbol, last_prices[symbol], 'green' if gain >= 0 else 'red', gain)
    account_html += '</table></td></tr>'

    text = textwrap.dedent("""\
    [ Account Summary ]
    {account_text}
    [ Sell Summary ]
    {sell_text}
    [ Buy Summary ]
    {buy_text}
    """)
    html = textwrap.dedent("""\
    <html>
    <head>
      <style>
        html {{
          font-family: -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,"Helvetica Neue",Arial,"Noto Sans",sans-serif,"Apple Color Emoji","Segoe UI Emoji","Segoe UI Symbol","Noto Color Emoji";
        }}
        table {{
          border-collapse: collapse;
        }}
        .table {{
          width: 80%;
          margin-bottom: 1rem;
          color: #212529;
        }}
        .table th,
        .table td {{
          padding: 0.75rem;
          vertical-align: top;
          border-top: 1px solid #dee2e6;
        }}
        .table thead th {{
          vertical-align: bottom;
           border-bottom: 2px solid #dee2e6;
        }}
        .table tbody + tbody {{
           border-top: 2px solid #dee2e6;
        }}
        .table-bordered {{
           border: 1px solid #dee2e6;
        }}
        .table-bordered th,
        .table-bordered td {{
           border: 1px solid #dee2e6;
        }}
        .table-bordered thead th,
        .table-bordered thead td {{
          border-bottom-width: 2px;
        }}
        th {{
          text-align: inherit;
        }}
        .table .thead-light th {{
          background-color: #f5f5f5;
        }}
        .display {{
          font-size: 1.4rem;
          font-weight: 300;
          line-height: 1.2;
        }}
        .narrow-col {{
          background-color: #f5f5f5;
          width: 30%;
        }}
        #table-account {{
          width: 40%;
        }}
        .css-chart {{
          border-bottom: 1px solid;
          border-left: 1px solid;
          display: inline-block;
          height: 36vw;
          margin: 25px 50px;
          padding: 0;
          position: relative;
          width: 77%;
        }}
        .line-chart {{
          list-style: none;
          margin: 0;
          padding: 0;
        }}
        .data-point {{
          background-color: white;
          border: 2px solid var(--color);
          border-radius: 50%;
          bottom: calc(var(--y) * 0.36 - 6px);
          left: calc(var(--x) * 6.4 - 6px);
          position: absolute;
          height: 10px;
          width: 10px;
          z-index: 1;
        }}
        .ticker {{
          position: absolute;
          font-weight: bold;
          font-size: 14px;
          color: #666;
          font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif;
        }}
        .x-ticker {{
          bottom: 5px;
          left: calc(var(--x) * 6.4 - 18px);
        }}
        .y-ticker {{
          bottom: calc(var(--y) * 0.36 - 7px);
          left: 5px;
        }}
        .hidden-ticker {{
          visibility: visible;
        }}
        .line-segment {{
          background-color: var(--color);
          left: calc(var(--x) * 6.4);
          bottom: calc(var(--y) * 0.36);
          position: absolute;
          transform: rotate(var(--angle));
          transform-origin: left bottom;
          width: var(--width);
        }}
        .dashed-line-segment {{
          border: none;
          border-top: 1px dashed #c6c6c6;
          position: absolute;
          left: 40px;
          bottom: calc(var(--y) * 0.36);
          width: 70vw;
          z-index: -1;
        }}
        .data-point .tooltip {{
          visibility: hidden;
          width: 120px;
          background-color: #fff;
          font-weight: bold;
          font-size: 14px;
          color: #666;
          font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif;
          text-align: center;
          border-radius: 6px;
          border: 1px solid #e1e1e1;
          padding: 5px 0;
          position: absolute;
          z-index: 2;
          bottom: 135%;
          left: 50%;
          margin-left: -61px;
          opacity: 0;
          transition: opacity 0.3s;
        }}
        .data-point .tooltip::after {{
          content: "";
          position: absolute;
          top: 100%;
          left: 50%;
          margin-left: -5px;
          border-width: 5px;
          border-style: solid;
          border-color: #555 transparent transparent transparent;
        }}
        .data-point:hover .tooltip {{
          visibility: visible;
          opacity: 1;
        }}
        .legend {{
          color: var(--color);
          left: calc(var(--x) * 6.4);
          bottom: calc(var(--y) * 0.36 - 7px);
          position: absolute;
        }}
        @media screen and (max-width: 1200px) {{
          #table-account {{
            width: 55%;
          }}
        }}
        @media screen and (max-width: 800px) {{
          .table {{
            width: 100%;
          }}
          #table-account {{
            width: 70%;
          }}
          .css-chart {{
            height: 45vw;
            width: 100%;
            margin: 12px 0px;
          }}
          .data-point {{
            left: calc(var(--x) * 8 - 4px);
            bottom: calc(var(--y) * 0.45 - 4px);
            height: 6px;
            width: 6px;
          }}
          .line-segment {{
            left: calc(var(--x) * 8);
            bottom: calc(var(--y) * 0.45);
            width: calc(var(--width) * 1.25);
            transform: rotate(var(--angle));
          }}
          .x-ticker {{
           left: calc(var(--x) * 8 - 18px);
          }}
          .y-ticker {{
            bottom: calc(var(--y) * 0.45 - 7px);
          }}
          .hidden-ticker {{
            visibility: hidden;
          }}
          .dashed-line-segment {{
            bottom: calc(var(--y) * 0.45);
            width: 82vw;
          }}
          .legend {{
            left: calc(var(--x) * 8);
            bottom: calc(var(--y) * 0.45 - 7px);
          }}
        }}
        @media screen and (max-width: 600px) {{
          #table-account {{
            width: 100%;
          }}
        }}
      </style>
    </head>
    <body>
      <h1 class="display">Account Summary</h1>
      <table class="table table-bordered" id="table-account">
        {account_html}
      </table>
      <h1 class="display">Sell Summary</h1>
      <table class="table table-bordered">
        <thead class="thead-light">
          <tr>
            <th scope="col">Symbol</th>
            <th scope="col">Buy Price</th>
            <th scope="col">Sell Price</th>
            <th scope="col">Quantity</th>
            <th scope="col">Gain / Loss</th>
          </tr>
        </thead>
        <tbody>
          {sell_html}
        </tbody>
      </table>
      <h1 class="display">Buy Summary</h1>
      <table class="table table-bordered">
        <thead class="thead-light">
          <tr>
            <th scope="col">Symbol</th>
            <th scope="col">Buy Price</th>
            <th scope="col">Quantity</th>
            <th scope="col">Cost</th>
          </tr>
        </thead>
        <tbody>
          {buy_html}
        </tbody>
      </table>
      <h1 class="display">10-day History</h1>
      <figure class="css-chart">
        <ul class="line-chart">
          {trading_history_html}
        </ul>
        {grid_html}
      </figure>
    </body>
    </html>
    """)
    message.attach(MIMEText(text.format(
        account_text=account_text, sell_text=sell_text, buy_text=buy_text), 'plain'))
    message.attach(MIMEText(html.format(
        account_html=account_html, sell_html=sell_html, buy_html=buy_html,
        trading_history_html=trading_history_html, grid_html=grid_html), 'html'))
    with open('test1.html', 'w') as f:
        f.write(html.format(account_html=account_html, sell_html=sell_html, buy_html=buy_html,
                            trading_history_html=trading_history_html, grid_html=grid_html))
    server.sendmail(sender, [receiver] + bcc, message.as_string())
    server.close()
    print('Email summary sent')


def send_alert(sender, receiver, user, password, exit_code):
    server = _create_server(user, password)
    message = _create_message(sender, receiver)
    message['Subject'] = '[Error] [%s] Trade running exited with code %d' % (
        datetime.date.today(), exit_code)
    text = 'Trade running exited with code %d' % (exit_code,)
    message.attach(MIMEText(text, 'plain'))
    server.sendmail(sender, receiver, message.as_string())
    server.close()
    print('Email alert sent')


def main():
    parser = argparse.ArgumentParser(description='Stock trading notification.')
    parser.add_argument('--exit_code', default=0, type=int,
                        help='Exit code of the trading run.')
    parser.add_argument('--api_key', default=None, help='Alpaca API key.')
    parser.add_argument('--api_secret', default=None, help='Alpaca API secret.')
    parser.add_argument('--sender', required=True, help='Email sender name.')
    parser.add_argument('--receiver', required=True, help='Email receiver address.')
    parser.add_argument('--user', required=True, help='Email user name.')
    parser.add_argument('--password', required=True, help='Email password.')
    parser.add_argument('--bcc', nargs='*', default=[], help='Email bcc address.')
    args = parser.parse_args()
    if args.exit_code == 0:
        alpaca = tradeapi.REST(args.api_key or os.environ['ALPACA_API_KEY'],
                               args.api_secret or os.environ['ALPACA_API_SECRET'],
                               utils.ALPACA_API_BASE_URL, 'v2')
        polygon = polygonapi.REST(os.environ['ALPACA_API_KEY'])
        send_summary(args.sender, args.receiver, args.bcc, args.user, args.password,
                     alpaca, polygon)
    else:
        send_alert(args.sender, args.receiver, args.user, args.password, args.exit_code)


if __name__ == '__main__':
    main()
