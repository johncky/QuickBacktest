import pandas as pd
import numpy as np
import quantstats as qs

class BacktestResult:
    def __init__(self, result_df, trade_df):
        self.result_df = result_df
        self.trade_df = trade_df

    def report(self, benchmark):
        import webbrowser

        PV = self.result_df.set_index('datetime')['pv'].resample('1D').last().dropna()
        html = 'backtest result.html'
        qs.reports.html(PV, benchmark, output=html, title=f'Backtest result')
        webbrowser.open(html)

    def plot(self):
        df = self.result_df.copy()
        df['trade_qty'] = df['quantity'].diff()
        df['buy_pt'] = [1 if qty_dif > 0 else None for qty_dif in df['trade_qty']]
        df['sell_pt'] = [1 if qty_dif < 0 else None for qty_dif in df['trade_qty']]
        df['buy_y'] = df['buy_pt'] * df['close']
        df['sell_y'] = df['sell_pt'] * df['close']
        df['x'] = df['datetime']
        from matplotlib import pyplot as plt
        plt.scatter(x=df['x'], y=df['buy_y'].values, marker='o', color='green', s=100)
        plt.scatter(x=df['x'], y=df['sell_y'].values, marker='o', color='red', s=100)
        plt.ylabel(f'price')
        plt.plot(df['x'], df['close'])
        plt.title(f'Entry-exit points')

    def stats(self):
        stats = dict()
        pv = self.result_df.set_index('datetime')['pv']
        tk = self.result_df.set_index('datetime')['adjclose'] if 'adjclose' in self.result_df.columns else \
        self.result_df.set_index('datetime')['close']
        pv = pv.resample('1D').last().dropna()
        tk = tk.resample('1D').last().dropna()
        ret = np.log(pv).diff()
        tk_ret = np.log(tk).diff()
        ret = ret.resample('1D').last().dropna()
        tk_ret = tk_ret.resample('1D').last().dropna()
        stats['N'] = pv.shape[0] - 1
        stats['Sharpe'] = qs.stats.sharpe(pv)
        stats['Sortino'] = qs.stats.sortino(pv)
        stats['CAGR %'] = (qs.stats.cagr(pv)) * 100
        stats['Cum Return %'] = (pv.iloc[-1] / pv.iloc[0] - 1) * 100
        stats['Cum Return (Stock) %'] = (tk.iloc[-1] / tk.iloc[0] - 1) * 100
        stats['Daily Ret %'] = (qs.stats.avg_return(pv)) * 100
        stats['Daily Vol %'] = (qs.stats.volatility(pv) / (251 ** (1 / 2))) * 100
        stats['Monthly Ret %'] = (stats['Daily Ret %'] * 21)
        stats['Monthly Vol %'] = (qs.stats.volatility(pv) / (21 ** (1 / 2))) * 100
        stats['Annual Ret %'] = (stats['Daily Ret %'] * 251)
        stats['Annual Vol %'] = (qs.stats.volatility(pv)) * 100
        stats['Win Days %'] = (qs.stats.win_rate(pv)) * 100
        stats['Max Drawdown %'] = (qs.stats.max_drawdown(pv)) * 100
        stats['Daily VAR %'] = (qs.stats.var(pv)) * 100
        stats['Beta'] = np.cov(ret.dropna(), tk_ret.dropna())[0][1] / np.var(tk_ret.dropna())
        stats['Alpha'] = stats['Cum Return %'] - stats['Cum Return (Stock) %']
        stats['No Trades'] = self.trade_df.shape[0]
        return stats


class BacktestResults:
    def __init__(self, backtest_results: dict):
        self.all_results = backtest_results

    def ticker_report(self, ticker, benchmark):
        self.all_results[ticker].report(benchmark=benchmark)

    def ticker_plot(self, ticker):
        self.all_results[ticker].plot()

    def ticker_stats(self, ticker):
        return self.all_results[ticker].stats()

    def stats(self):
        tk_stats = dict()
        for tk in self.all_results.keys():
            tk_stats[tk] = self.all_results[tk].stats()
        return np.round(pd.DataFrame(tk_stats), 2)

    def portfolio_report(self, benchmark, allocations=None):
        import webbrowser
        portfolio_pv = self.make_portfolio(allocations=allocations)['pv'].resample('1D').last().dropna()
        html = 'backtest result.html'
        qs.reports.html(portfolio_pv, benchmark, output=html, title=f'Backtest result')
        webbrowser.open(html)

    def make_portfolio(self, allocations=None):
        df = pd.concat([x.result_df.set_index('datetime')['pv'] for x in self.all_results.values()], axis=1)
        df.columns = self.all_results.keys()
        df = df.fillna(method='bfill')
        df = df.resample('1D').last().dropna()
        allocations = [1 / len(self.all_results)] * len(self.all_results) if allocations is None else allocations
        return pd.DataFrame({'datetime': df.index, 'pv': np.dot(df, allocations)}).set_index('datetime')
