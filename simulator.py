import math
import numpy as np
import pandas as pd
import random

class market_position(object):
    def __init__(self, v=0.01):
        '''Initialize the simulator, v is the spread, we assume it is constant for now.'''

        # constant parameters
        self.mu = 0.0001
        self.sigma = 0.01
        self.ell = 5000
        self.dt = 1
        self.spread = v
        self.transaction_p = 0.7

        self.P_L = [10000]
        self.hist_x = [10]
        self.hist_bid = [9.75]
        self.hist_ask = [10.25]
        self.hist_q = [0]
        self.hist_cash = [10000]
        self.Pbuy = [0.5]
        initial_skew = findMaxSkew(self.mu, self.sigma, self.hist_x[-1], self.ell, self.hist_cash[-1], self.hist_q[-1], self.spread, self.dt)
        self.hist_skew = [initial_skew]
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
        self.hist_skew.append(skew)
        
    def get_accumulated_P_L(self, t):
        '''Show hist P&L'''
        return self.P_L
    
    def transaction_coming(self):
        '''Return True as a transaction has arrived,
        Return False as no transaction coming.'''
        return np.random.binomial(1, self.transaction_p) == 1
    
    def buy_sell(self):
        '''Return True as buy transaction,
        Return False as sell transaction'''
        return np.random.binomial(1, self.Pbuy[-1]) == 1
    
    def new_stock_price(self):
        '''Return stock price change'''
        return self.hist_x[-1] * math.exp((self.mu - (self.sigma ** 2) / 2) * self.dt + self.sigma * np.random.normal())
    
    def simulate_one_3tree(self):
        '''Simulate one time foreward. This method simulated stock price in 3 branches tree'''
        # simulate incoming transaction, if there is, adjust inventory and cash account
        if self.transaction_coming():
            if self.buy_sell():
                self.hist_q.append(self.hist_q[-1] + 1)
                self.hist_cash.append(self.hist_cash[-1] - self.hist_bid[-1])
            else:
                self.hist_q.append(self.hist_q[-1] - 1)
                self.hist_cash.append(self.hist_cash[-1] + self.hist_ask[-1])
        else:
            self.hist_q.append(self.hist_q[-1])
            self.hist_cash.append(self.hist_cash[-1])
            
        # simulate stock price moving path, record the P&L
        # assume the stock price moves as random walk 0f -0.1, 0, 0.1
        # self.hist_x.append(self.hist_x[-1] + random.randint(-1,1) * 0.1)
        
        self.hist_x.append(self.new_stock_price())
        
        self.P_L.append(self.hist_q[-1] * self.hist_x[-1] + self.hist_cash[-1])
        
        new_skew = findMaxSkew(self.mu, self.sigma, self.hist_x[-1], self.ell, self.hist_cash[-1], self.hist_q[-1], self.spread, self.dt)
        self.hist_skew.append(new_skew)
        self.Pbuy.append(self.hist_skew[-1])
        
        self.hist_ask.append(self.hist_x[-1] + self.spread * (3 / 2 - 2 * self.hist_skew[-1]))
        self.hist_bid.append(self.hist_x[-1] + self.spread * (1 / 2 - 2 * self.hist_skew[-1]))
        
    def n_simulate(self, n):
        '''simulate n steps'''
        for i in range(n):
            self.simulate_one_3tree()
            
def findMaxSkew(mu,sigma,x,ell,cash,Q,vx,dt):
    # Set all relevant parameters
    u = np.exp(sigma*np.sqrt(dt))
    d = 1/u
    p_up = (np.exp(mu*dt)-d)/(u-d)
    p_dn = 1-p_up

    #p_buy = p*s
    #p_sell = p*(1-s)
    #price_buy =

    # Scenario S1 & S4:
    if(cash+u*x*Q<ell or cash+d*x*Q<ell):
        return -1

    else:
        upper1 = (-ell+cash+1.5*vx+x*(u*Q+u-1))/(2*vx) # S2
        upper2 = (-ell+cash+1.5*vx+x*(d*Q+d-1))/(2*vx) # S5
        upper = min(upper1,upper2)

        lower1 = (ell-cash+0.5*vx-x*(u*Q-u+1))/(2*vx) # S3
        lower2 = (ell-cash+0.5*vx-x*(d*Q-d+1))/(2*vx) # S6
        lower = min(lower1,lower2)

        if(upper<lower or upper<0 or lower>1): return -1
        else:
            upper = min(upper,1)
            lower = min(lower,0)

            #  we have to: max -4*vx*(s^2) + (4*V-2*X+2*X*exp(mu*t))*s
            #  s = -b/2a
            s = 0.5+0.25*(np.exp(mu*dt)-1)*x/vx
            if(s<=upper and s>=lower): return s
            elif(lower > s): return lower
            else: return upper
            
def findMaxSkew(mu,sigma,x,ell,cash,Q,vx,dt):
    # Set all relevant parameters
    u = np.exp(sigma*np.sqrt(dt))
    d = 1/u
    p_up = (np.exp(mu*dt)-d)/(u-d)
    p_dn = 1-p_up

    #p_buy = p*s
    #p_sell = p*(1-s)
    #price_buy =

    # Scenario S1 & S4:
    if(cash+u*x*Q<ell or cash+d*x*Q<ell):
        return -1

    else:
        upper1 = (-ell+cash+1.5*vx+x*(u*Q+u-1))/(2*vx) # S2
        upper2 = (-ell+cash+1.5*vx+x*(d*Q+d-1))/(2*vx) # S5
        upper = min(upper1,upper2)

        lower1 = (ell-cash+0.5*vx-x*(u*Q-u+1))/(2*vx) # S3
        lower2 = (ell-cash+0.5*vx-x*(d*Q-d+1))/(2*vx) # S6
        lower = max(lower1,lower2)

        if(upper<lower or upper<0 or lower>1): return -1
        else:
            upper = min(upper,1)
            lower = max(lower,0)

            #  we have to: max -4*vx*(s^2) + (4*V-2*X+2*X*exp(mu*t))*s
            #  s = -b/2a
            s = 0.5+0.25*(np.exp(mu*dt)-1)*x/vx
            if(s<=upper and s>=lower): return s
            elif(lower > s): return lower
            else: return upper

if __name__ == "__main__":    
    a = market_position()
    
    a.n_simulate(3600)
    
    # print("skew: ")
    # print(a.hist_skew)
    # print("stock price: ")
    # print(a.hist_x)
    # print("bid")
    # print(a.hist_bid)
    # print("ask")
    # print(a.hist_ask)
    # print("q")
    # print(a.hist_q)
    # print("cash")
    # print(a.hist_cash)
    print("coming transaction probability")
    print(a.transaction_p)
    print("buy transaction probability")
    
    df = pd.DataFrame()
    df['skew'] = a.hist_skew
    df['price'] = a.hist_x
    df['bid'] = a.hist_bid
    df['ask'] = a.hist_ask
    df['inventory'] = a.hist_q
    df['cash'] = a.hist_cash
    df['P&L'] = np.array(a.P_L) - 10000
    df['Prob_Buy'] = a.Pbuy
    
    print(df[-100:-1])
    
    
    
# rewrite price move
# add the skew function