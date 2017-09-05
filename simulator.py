import math
import numpy as np
import pandas as pd
import random

class market_position(object):
    def __init__(self, v=0.5):
        '''Initialize the simulator, v is the spread, we assume it is constant for now.'''
        self.spread = v
        self.P_L = [10000]
        self.historical_skew = [0.5]
        self.historical_x = [10]
        self.historical_bid = [9.75]
        self.historical_ask = [10.25]
        self.historical_q = [0]
        self.historical_cash = [10000]
        self.transaction_p = 0.7
        self.Pbuy = 0.5
        # probability of sell will be 1 - self.Pbuy, no need to define again

    def __repr__(self):
        '''Representation'''
        result = ""
        result += "Current time is: " + str(self.x) + "\n"
        result += "Current stock price is: " + str(self.x) + "\n"
        result += "Current inventory is: " + str(self.q) + "\n"
        result += "Current cash account is: " + str(self.x) + "\n"
        return result

    def set_skew(self, skew):
        '''Set a new skewness of bid ask spread'''
        self.historical_skew.append(skew)

    def get_accumulated_P_L(self, t):
        '''Show historical P&L'''
        return self.P_L

    def transaction_coming(self, p):
        '''Return True as a transaction has arrived,
        Return False as no transaction coming.'''
        return np.random.binomial(1, p) == 1

    def buy_sell(self, Pbuy):
        '''Return True as buy transaction,
        Return False as sell transaction'''
        return np.random.binomial(1, Pbuy) == 1

    def simulate_one_3tree(self):
        '''Simulate one time foreward. This method simulated stock price in 3 branches tree'''
        # simulate incoming transaction, if there is, adjust inventory and cash account
        if self.transaction_coming(self.transaction_p):
            if self.buy_sell(self.Pbuy):
                self.historical_q.append(self.historical_q[-1] + 1)
                self.historical_cash.append(self.historical_cash[-1] - self.historical_bid[-1])
            else:
                self.historical_q.append(self.historical_q[-1] - 1)
                self.historical_cash.append(self.historical_cash[-1] + self.historical_ask[-1])
        else:
            self.historical_q.append(self.historical_q[-1])
            self.historical_cash.append(self.historical_cash[-1])

        # simulate stock price moving path, record the P&L
        # assume the stock price moves as random walk 0f -0.1, 0, 0.1
        self.historical_x.append(self.historical_x[-1] + random.randint(-1,1) * 0.1)
        self.P_L.append(self.historical_q[-1] * self.historical_x[-1] + self.historical_cash[-1])

        ####### implement skew strategy here: #######
        self.historical_skew.append(0.5)

        self.historical_ask.append(self.historical_x[-1] + self.spread * (3 / 2 - 2 * self.historical_skew[-1]))
        self.historical_bid.append(self.historical_x[-1] + self.spread * (1 / 2 - 2 * self.historical_skew[-1]))

    def n_simulate(self, n):
        '''simulate n steps'''
        for i in range(n):
            self.simulate_one_3tree()

if __name__ == "__main__":
    a = market_position()

    a.n_simulate(50)

    # print("skew: ")
    # print(a.historical_skew)
    # print("stock price: ")
    # print(a.historical_x)
    # print("bid")
    # print(a.historical_bid)
    # print("ask")
    # print(a.historical_ask)
    # print("q")
    # print(a.historical_q)
    # print("cash")
    # print(a.historical_cash)
    print("coming transaction probability")
    print(a.transaction_p)
    print("buy transaction probability")
    print(a.Pbuy)

    df = pd.DataFrame()
    df['skew'] = a.historical_skew
    df['price'] = a.historical_x
    df['bid'] = a.historical_bid
    df['ask'] = a.historical_ask
    df['inventory'] = a.historical_q
    df['cash'] = a.historical_cash
    df['P&L'] = np.array(a.P_L) - 10000

    print(df)
