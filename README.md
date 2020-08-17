# QuickBacktest

Backtest technical indicators based trading strategies.
Automatically download data from Yahoo Finance, backtest strategy, and produce performance statistics and report.

# Example
```python
    from quickBacktest import Strategy
    
    class SMA(Strategy):
        def init(self):
            self.data['sma16'] = self.data['adjclose'].rolling(16).mean()
            self.data['sma32'] = self.data['adjclose'].rolling(32).mean()
            self.data['dif'] = self.data['sma16'] - self.data['sma32']
            self.data['pre_dif'] = self.data['dif'].shift(1)

        def signal(self):
            if self.dif > 0 and self.pre_dif <= 0:
                self.long()

            elif self.dif < 0 and self.pre_dif >= 0:
                self.exit_long()
            else:
                pass


    sma = SMA()
    result = sma.backtest(tickers=tickers,
                           capital=1000000,
                           start_date="2015-01-01",
                           end_date="2020-07-31",
                           buy_at_open=True,
                           bid_ask_spread=0.0,
                           fee_mode='FIXED:0',
                           data_params={'interval': '1d', 'range': '10y'})
                      
    # if "allocations" is not specified, default equal weightings
    # performance report of a portfolio that invested 25% of capital in each ticker
    result.portfolio_report(benchmark="^IXIC", allocations=[0.25,0.25,0.25,0.25])

    # stats of all tickers
    result.stats()
    
    # plot exit-entry points of a ticker
    result.ticker_plot('FB')
    
    # report of a ticker
    result.ticker_report('FB', benchmark='^IXIC')
    
    # df that contains all trades
    trade_df = result.all_results['FB'].trade_df
```

### Trade functions
1. long: buy minimum(x, max affordable qty) qty
2. exit_long: sell stocks to cover long position
3. cover_and_long: cover short position, and buy minimum(x, max affordable qty) qty
4. short: short sell maximum(x, max qty to short) qty; max qty to short = int((cash + equity value) / trade price)
5. cover_short: buy stocks to cover short position
6. exit_and_short: exit long position, and short sell maximum(x, max qty to short) qty
7. liquidate: liquidate position, hold cash


```python
# use this to convert % invested to corresponding signal and quantity to trade
signal, qty = quickBacktest.convert_percent_to_qty(percentage, cur_cash, cur_qty, trade_price)

# for example, if you are currently 15% in stock, and you want to calculate
# the signal and qty required to be 35% invested in stock based on your current cash and quantity: 
signal, qty = quickBacktest.convert_percent_to_qty(percentage=0.35,
                                                    states['cash'], 
                                                    states['quantity'],
                                                    states['trade_price'])
# this would return ("LONG", quantity required to buy on top of current qty)
```

