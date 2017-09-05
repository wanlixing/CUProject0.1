import pandas as pd
import numpy as np

# Single Asset Optimization_Calculate s

# Parameters:
# p : Prob of a Trade coming
# mu : stock price drift
# sigma: stock price diffusion
# x : Current stock price
# Q : Current cumulated positions (before this period's trade)
# s : Ask-bid skewness, s>=0 and s<=1
# vx : Ask-bid spread
# dt : time slice
# mu : Mean of stock price
# cash: Current cumulated cash position (before this period's trade)



# Optimization Problem:
# max E(cash + cash from current trade + StockPriceNextPeriod * PositionsAfterCurrentTrade)
# s.t. P&L in the worst scenario > Limit

# Define Scenarios:
# S1: PriceUp & NoTrade: cash+u*x*Q
# S2: PriceUp & WeBuy: cash-x-2*vx*s+(3/2)*vx+u*x*(Q+1)
# S3: PriceUp & WeSell: cash+x+2*vx*s-(1/2)*vx+u*vx*(Q-1)
# S4: PriceDown & NoTrade: cash+d*x*Q
# S5: PriceDown & WeBuy: cash-x-2*vx*s+(3/2)*vx+d*x*(Q+1)
# S6: PriceDown & WeSell: cash+X+2*vx*s-(1/2)*vx+d*x*(Q-1)

# If feasible set is empty, return s = -1
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

cash = 0
ell = -500
mu = -0.1
dt = 1
sigma = 0.1
x = 5
vx = 0.5
Q = 0

s = findMaxSkew(mu,sigma,x,ell,cash,Q,vx,dt)
print(s)