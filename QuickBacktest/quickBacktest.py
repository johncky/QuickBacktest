import pandas as pd
import numpy as np
from result import BacktestResult, BacktestResults
from QuickBacktest import yh
from abc import abstractmethod


class Strategy:
    def __init__(self):
        self.cash = None
        self.quantity = None
        self._trade_price = None
        self._trade_datetime = None
        self._bid_ask_spread = None
        self._cal_fee = None
        self._trades = None
        self.data = None
        self._bt_result = None

    def init(self):
        pass

    @abstractmethod
    def signal(self):
        pass

    def long(self, qty=None):
        if self.cash > 0:
            action_price = self._trade_price * (1 + self._bid_ask_spread)
            max_qty = int(self.cash / action_price)
            bot_qty = max_qty if qty is None else min(max_qty, abs(int(qty)))

            cash_proceeds = -bot_qty * action_price
            txn_fee = self._cal_fee(bot_qty, action_price)
            self.cash = self.cash + cash_proceeds - txn_fee
            self.quantity += bot_qty

            self._trades.loc[self._trade_datetime] = ['BUY', action_price, bot_qty, cash_proceeds, txn_fee]

    def exit_long(self):
        if self.quantity > 0:
            action_price = self._trade_price * (1 - self._bid_ask_spread)
            exit_qty = self.quantity

            cash_proceeds = self.quantity * action_price
            txn_fee = self._cal_fee(exit_qty, action_price)

            self.cash = self.cash + cash_proceeds - txn_fee
            self.quantity = 0
            self._trades.loc[self._trade_datetime] = ['SELL', action_price, exit_qty, cash_proceeds, txn_fee]

    def short(self, qty=None):
        action_price = self._trade_price * (1 - self._bid_ask_spread)

        ev = self.quantity * action_price
        pv = self.cash + ev
        min_short = -int(pv / action_price)

        short_qty = (self.quantity - min_short) if qty is None else abs(int(qty))
        if self.quantity - short_qty < min_short:
            short_qty = self.quantity - min_short

        cash_proceeds = short_qty * action_price
        txn_fee = self._cal_fee(short_qty, action_price)
        self.cash = self.cash + cash_proceeds - txn_fee
        self.quantity -= short_qty
        self._trades.loc[self._trade_datetime] = ['SELL', action_price, short_qty, cash_proceeds, txn_fee]

    def cover_short(self):
        if self.quantity < 0:
            action_price = self._trade_price * (1 + self._bid_ask_spread)

            bot_qty = self.quantity * -1

            cash_proceeds = self.quantity * action_price
            txn_fee = self._cal_fee(bot_qty, action_price)
            self.cash = self.cash + cash_proceeds - txn_fee
            self.quantity = 0
            self._trades.loc[self._trade_datetime] = ['BUY', action_price, bot_qty, cash_proceeds, txn_fee]

    def cover_and_long(self, qty=None):
        if self.quantity <= 0:
            action_price = self._trade_price * (1 + self._bid_ask_spread)

            pv = self.quantity * action_price + self.cash
            max_qty = int(pv / action_price)
            long_qty = max_qty if qty is None else min(max_qty, int(qty))
            bot_qty = -1 * self.quantity + long_qty

            cash_proceeds = -action_price * bot_qty
            txn_fee = self._cal_fee(bot_qty, action_price)

            self.cash = self.cash + cash_proceeds - txn_fee
            self.quantity += bot_qty
            self._trades.loc[self._trade_datetime] = ['BUY', action_price, bot_qty, cash_proceeds, txn_fee]

    def exit_and_short(self, qty=None):
        if self.quantity >= 0:
            action_price = self._trade_price * (1 - self._bid_ask_spread)

            pv = self.quantity * action_price + self.cash
            max_qty = int(pv / action_price)
            short_qty = max_qty if qty is None else min(max_qty, abs(int(qty)))
            sell_qty = -short_qty - self.quantity

            cash_proceeds = - sell_qty * action_price
            txn_fee = self._cal_fee(sell_qty, action_price)
            self.cash = self.cash + cash_proceeds - txn_fee

            self.quantity += sell_qty
            self._trades.loc[self._trade_datetime] = ['SELL', action_price, -sell_qty, cash_proceeds, txn_fee]

    def liquidate(self):
        if self.quantity == 0:
            return
        if self.quantity > 0:
            action_price = self._trade_price * (1 - self._bid_ask_spread)
        else:
            action_price = self._trade_price * (1 + self._bid_ask_spread)

        action_qty = -self.quantity
        cash_proceeds = self.quantity * action_price

        txn_fee = self._cal_fee(action_qty, action_price)
        self.cash = self.cash + cash_proceeds - txn_fee

        self.quantity = 0
        trd_side = 'SELL' if action_qty < 0 else 'BUY'
        self._trades.loc[self._trade_datetime] = [trd_side, action_price, abs(action_qty), cash_proceeds, txn_fee]

    def set_attrs(self, df):
        for col in df.columns:
            setattr(self, col, df[col].iloc[-1])

    def loop(self, df, capital, buy_at_open, bid_ask_spread, fee_mode):
        records = pd.DataFrame(columns=['cash', 'quantity'])
        records.index.name = 'datetime'
        self._trades = pd.DataFrame(
            columns=['trade side', 'trade price', 'trade quantity', 'trade proceeds', 'transaction fee'])
        self._trades.index.name = 'datetime'

        self.quantity = 0
        self.cash = capital
        self._bid_ask_spread = bid_ask_spread

        df = df.copy()
        df.columns = [x.lower() for x in df.columns]
        df['next open'] = df['open'].shift(-1)
        df['next datetime'] = df['datetime'].shift(-1)
        df.reset_index(drop=True, inplace=True)
        last_id = df.index[-1]
        action_price_label = 'next open' if buy_at_open else 'close'
        action_datetime = 'next datetime' if buy_at_open else 'datetime'

        try:
            mode, fee = fee_mode.split(':')
            fee = float(fee)
            if mode.upper() == 'FIXED':
                def cal_fee(qty, price):
                    return fee
            elif mode.upper() == 'QTY':
                def cal_fee(qty, price):
                    return abs(int(qty)) * fee
            elif mode.upper() == 'PERCENT':
                def cal_fee(qty, price):
                    return abs(int(qty)) * price * fee
            else:
                raise Exception()

        except Exception:
            print(f'Invalid fee mode {fee_mode}, using zero fees')

            def cal_fee(qty, price):
                return 0.0

        self._cal_fee = cal_fee

        for id, row in df.iterrows():
            if buy_at_open:
                records.loc[row['datetime']] = [self.cash, self.quantity]

            if (id == last_id) and buy_at_open:
                continue

            self._trade_price = row[action_price_label]
            self._trade_datetime = row[action_datetime]

            self.set_attrs(df.iloc[id:id + 1])
            self.signal()

            if not buy_at_open:
                records.loc[row['datetime']] = [self.cash, self.quantity]

        completed_df = df.copy()
        completed_df = completed_df.merge(records.reset_index(), on=['datetime'], how='right')
        completed_df['ev'] = (completed_df['quantity'] * completed_df['close']).astype(float)
        completed_df['pv'] = (completed_df['ev'] + completed_df['cash']).astype(float)
        completed_df['interval return'] = np.log(completed_df['pv']).diff()
        completed_df['datetime'] = pd.to_datetime(completed_df['datetime'])
        self._trades = self._trades.reset_index()
        self._trades['datetime'] = pd.to_datetime(self._trades['datetime'])
        return BacktestResult(result_df=completed_df, trade_df=self._trades.copy())

    def backtest(self, tickers, capital=1000000, buy_at_open=True,
                 bid_ask_spread: float = 0.0,
                 fee_mode: str = 'FIXED:0', start_date=None, end_date=None, data_params=None):
        bt_results = dict()
        print('Downloading data from Yahoo...')
        dfs = yh.Stocks(tickers, param=data_params).prices()
        for tk in tickers:
            if tk not in dfs.keys():
                print(f'Pass backtesting of {tk}...')
                continue
            df = dfs[tk]
            if start_date:
                df = df.loc[df['datetime'] >= start_date]
            if end_date:
                df = df.loc[df['datetime'] <= end_date]

            self.data = df.copy()
            self.init()
            self.data = self.data.dropna()

            if self.data.shape[0] == 0:
                print(f'Zero {tk} data left after init(), skipped backtesting...')
            else:
                print(f'Backtesting {tk}...')
                tk_result = self.loop(df=self.data, capital=capital, buy_at_open=buy_at_open,
                                      bid_ask_spread=bid_ask_spread,
                                      fee_mode=fee_mode)
                bt_results[tk] = tk_result
        print('Completed!')

        self._bt_result = BacktestResults(backtest_results=bt_results)

        # clear
        self.data = None
        self.cash = None
        self.quantity = None
        return self._bt_result

    def hold(self, qty):
        return

    def convert_percent_to_qty(self, percentage):
        if percentage == '':
            return "PASS", 0
        ev = self.quantity * self._trade_price
        pv = self.cash + ev
        try:
            percent = float(percentage)
            if percent > 1 or percent < -1:
                raise Exception()
        except Exception:
            print(f'Invalid % invested {percentage}, return PASS signal')
            return "PASS", 0

        required_qty = int(pv * percent / self._trade_price)
        action_qty = required_qty - self.quantity
        if action_qty == 0:
            return self.hold, 0
        if action_qty > 0:
            return self.long, action_qty
        else:
            return self.short, abs(action_qty)

if __name__ == '__main__':

    tickers = ['0700.HK']

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
                self.liquidate()
            else:
                pass


    sma = SMA()
    result2 = sma.backtest(tickers=tickers,
                           capital=200000,
                           buy_at_open=True,
                           bid_ask_spread=0.0,
                           fee_mode='FIXED:0',
                           start_date='2015-01-01')

    result2.portfolio_report('0700.HK')