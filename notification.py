import alpaca_trade_api as tradeapi
import argparse
import collections
import datetime
import os
import smtplib
import textwrap
import utils
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


def send_summary(sender, receiver, bcc, user, password, alpaca):
    calendar = alpaca.get_calendar(start=datetime.date.today() - datetime.timedelta(days=15),
                                   end=datetime.date.today())
    open_dates = sorted([c.date for c in calendar], reverse=True)
    if open_dates[0].strftime('%y%m%d') != datetime.date.today().strftime('%y%m%d'):
        print('Today is not a trading day')
        return

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
                      '<td style="color:%s">%+.2f (%+.2f%%)</td> </tr>') % (
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
    equity = float(account.equity)
    account_text = 'Equity: %.2f\nCash: %s\nGain / Loss: %+.2f (%+.2f%%)\n' % (
        equity, account.cash, total_gain, total_gain / (equity - total_gain) * 100)
    account_html = ('<tr><th scope="row" class="narrow-col">Equity</th><td>%.2f</td></tr>'
                    '<tr><th scope="row" class="narrow-col">Cash</th><td>%s</td></tr>'
                    '<tr><th scope="row" class="narrow-col">Gain / Loss</th>'
                    '<td style="color:%s">%+.2f (%+.2f%%)</td></tr>') % (
        equity, account.cash, 'green' if total_gain >= 0 else 'red', total_gain,
        total_gain / (equity - total_gain) * 100)

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
        @media screen and (max-width: 800px) {{
          .table {{
            width: 100%;
          }}
          #table-account {{
            width: 70%;
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
    </body>
    </html>
    """)
    message.attach(MIMEText(text.format(
        account_text=account_text, sell_text=sell_text, buy_text=buy_text), 'plain'))
    message.attach(MIMEText(html.format(
        account_html=account_html, sell_html=sell_html, buy_html=buy_html), 'html'))
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
        send_summary(args.sender, args.receiver, args.bcc, args.user, args.password, alpaca)
    else:
        send_alert(args.sender, args.receiver, args.user, args.password, args.exit_code)


if __name__ == '__main__':
    main()
