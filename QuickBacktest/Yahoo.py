import requests
import pandas as pd
from datetime import datetime
from multiprocessing.dummy import Pool as ThreadPool

# valid intervals: 1m,2m,5m,15m,30m,60m,90m,1h,1d,5d,1wk,1mo,3mo
# valid periods: 1d,5d,1mo,3mo,6mo,1y,2y,5y,10y,ytd,max
default_setting = {'interval': '1d',
                   'range': '10y'}

def make_param(start=None, end=None, interval='1d', prepost='false', events=False, range='15y'):
    if start and end:
        period1 = str(int(datetime.strptime(start, "%Y-%m-%d %H:%M:%S").timestamp()))
        period2 = str(int(datetime.strptime(end, "%Y-%m-%d %H:%M:%S").timestamp()))
        param = {'period1': period1,
                 'period2': period2,
                 'interval': interval,
                 'event': str(events),
                 'includePrePost':str(prepost)}
    else:
        param = {'range': range,
                 'interval': interval,
                 'event': str(events),
                 'includePrePost': str(prepost)}
    return param


class Stock:
    def __repr__(self):
        return 'Yahoo Finance Object: {}'.format(self.code)

    def __init__(self, stock, param=None):
        self.code = stock
        self.url_v8 = 'https://query1.finance.yahoo.com/v8/finance/chart/'
        self.url_v10 = 'https://query1.finance.yahoo.com/v10/finance/quoteSummary/'
        self.param = param if isinstance(param, dict) else default_setting

    def price(self, param=None):
        error = 'Connection Failed'
        try:
            param = param if param else self.param
            url = '{}{}'.format(self.url_v8, self.code)
            data = requests.get(url, params=param).json()
            error = data['chart']['error']
            data = data['chart']['result'][0]
            date = [datetime.fromtimestamp(x) for x in data['timestamp']]
            opn = data['indicators']['quote'][0]['open']
            high = data['indicators']['quote'][0]['high']
            low = data['indicators']['quote'][0]['low']
            close = data['indicators']['quote'][0]['close']
            volume = data['indicators']['quote'][0]['volume']
            try:
                adjusted_c = data['indicators']['adjclose'][0]['adjclose']
                df = pd.DataFrame({'Date': date,
                                   'Open': opn,
                                   'High': high,
                                   'Low': low,
                                   'Close': close,
                                   'AdjClose': adjusted_c,
                                   'Volume': volume})
            except KeyError:
                df = pd.DataFrame({'Date': date,
                                   'Open': opn,
                                   'High': high,
                                   'Low': low,
                                   'Close': close,
                                   'Volume': volume})
            dropped_df = df.dropna()
            dropped_df = dropped_df.rename(columns={'Date': 'datetime'})
            dropped_df.columns = [x.lower() for x in dropped_df.columns]

            if dropped_df.shape[0] != df.shape[0]:
                print(f'Dropped {df.shape[0] - dropped_df.shape[0]} rows {self.code}')

            # before_drop_zero = dropped_df.shape[0]
            # dropped_df = dropped_df.loc[dropped_df['volume'] > 0]
            # if dropped_df.shape[0] != before_drop_zero:
            #     print(f'Dropped {before_drop_zero - dropped_df.shape[0]} rows of zero volume {self.code}')


            # if self.param['interval'] == '1d':
            #     dropped_df['datetime'] = pd.to_datetime([x.strftime('%Y-%m-%d') for x in dropped_df['datetime']])
            return dropped_df

        except Exception:
            print(f'Failed to download {self.code}')
            print(f'Reason {error}')
            raise
            return None


class Stocks:
    def __repr__(self):
        return 'Yahoo Finance Stocks: {}'.format(' '.join(self.stocks))

    def __init__(self, stocks, param=None):
        self.stocks = stocks
        self.param = param if isinstance(param, dict) else default_setting
        self.url_v8 = 'https://query1.finance.yahoo.com/v8/finance/chart/'
        self.url_v10 = 'https://query1.finance.yahoo.com/v10/finance/quoteSummary/'
        self.objects = [Stock(x, param) for x in self.stocks]

    def prices(self, param=None, threads=4):

        def stock_price(stock_object, _param=param):
            return stock_object.price(_param)

        threads = min(threads, len(self.stocks))
        pool = ThreadPool(threads)
        prices = pool.map(stock_price, self.objects)
        pool.close()
        pool.join()
        result = dict()
        for tk, df in zip(self.stocks, prices):
            if df is not None:
                result[tk] = df
        return result


