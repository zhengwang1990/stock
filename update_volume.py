from common import *


def get_volume(symbol):
    url = 'https://finance.yahoo.com/quote/%s'%(symbol,)
    prefixes = ['"averageVolume"', '"regularMarketVolume"']
    try:
        volume = int(web_scraping(url, prefixes))
    except Exception as e:
        print(e)
        volume = None
    return symbol, volume

def main():
    dir_path = os.path.dirname(os.path.realpath(__file__))
    symbols = []
    for f in os.listdir(os.path.join(dir_path, 'data')):
        if f.endswith('csv'):
            df = pd.read_csv(os.path.join('data', f))
            symbols.extend([row.Symbol for row in df.itertuples() if re.match('^[A-Z]*$', row.Symbol)])

    pool = futures.ThreadPoolExecutor(max_workers=MAX_THREADS)
    threads = []
    for symbol in symbols:
        t = pool.submit(get_volume, symbol)
        threads.append(t)

    res = {}
    for t in tqdm(threads, ncols=80):
        symbol, volume = t.result()
        if volume:
            res[symbol] = volume

    pool.shutdown()

    s = json.dumps(res)
    with open(os.path.join(dir_path, 'data', 'volumes.json'), 'w') as f:
        f.write(s)


if __name__ == '__main__':
  main()
