# QuickBacktest

Backtest technical indicators based trading strategies.
Automatically download data from Yahoo Finance, backtest strategy, and produce performance statistics and report.

# Example
```python
    # states are preserved in loop
    # put your variables/objects inside states
    # cash, quantity, trade price, trade date are preserved names in states
    # cash: current cash; quantity: current quantity; 

    def sma_crossover_signal(df, states):
        # a function that takes in df of market data(datetime, open, high, low, close, adjclose(null for intraday data)
        # and return signals and quantity to trade
        df['sma16'] = df['adjclose'].rolling(16).mean()
        df['sma32'] = df['adjclose'].rolling(32).mean()
        df['dif'] = df['sma16'] - df['sma32']
        df['pre_dif'] = df['dif'].shift(1)
        row = df.iloc[-1]
        
        # modified states will be preserved
        states['var1'] += 1
        if row['dif'] > 0 and row['pre_dif'] <= 0:
            return 'COVER AND LONG', 'ALL'

        elif row['dif'] < 0 and row['pre_dif'] >= 0:
            return 'EXIT AND SHORT', 'ALL'
        else:
            return 'PASS', ''

    # tickers to backtest
    tickers = ['FB', 'AMZN', 'AAPL', 'GOOG']

    result = backtest(tickers=tickers, capital=1000000,strategy_func=sma_crossover_signal, 
                        start_date="2015-01-01",
                        end_date="2020-07-31", states={'var1': 0, 'var2': list()}, 
                        buy_at_open=True, bid_ask_spread= 0.0,
                        fee_mode= 'FIXED:0', max_rows=None)
                      
    # if "allocations" is not specified, default equal weightings
    # performance report of a portfolio where 25% of capital is invested in each ticker
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

### Signals
1. LONG: buy min(x, max affordable qty) qty
2. COVER AND LONG: cover short position, and buy min(x, max affordable qty) qty
3. SHORT: short sell max(x, max qty to short) qty; max qty to short = int((cash + equity value) / trade price)
4. EXIT AND SHORT: exit long position, and short sell max(x, max qty to short) qty
5. LIQUIDATE: liquidate position, hold cash

