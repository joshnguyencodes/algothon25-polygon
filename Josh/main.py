import numpy as np

nInst = 50
currentPos = np.zeros(nInst)
entryPrice = np.full(nInst, np.nan)
stopPrice = np.full(nInst, np.nan)
positionSide = np.zeros(nInst)  # 1 for long, -1 for short, 0 for flat

# --- Strategy Parameters ---
N = 100  # Donchian channel lookback (can optimize 20-100)
ATR_period = 20
stop_multiplier = 2.0  # ATR multiple for trailing stop
risk_pct = 0.01  # 1% of equity per trade

def getMyPosition(prcSoFar):
    global currentPos, entryPrice, stopPrice, positionSide
    (nins, nt) = prcSoFar.shape
    equity = 50000 # Assume starting equity (can be tracked dynamically)
    newPos = np.array(currentPos)

    if nt < max(N, ATR_period) + 1:
        return np.zeros(nins)

    # Calculate Donchian Channel
    upper = np.max(prcSoFar[:, -N:], axis=1) *0.99
    lower = np.min(prcSoFar[:, -N:], axis=1) *0.99
    middle = (upper + lower) / 2
    price = prcSoFar[:, -1]

    # Calculate ATR
    high = np.max(prcSoFar[:, -ATR_period:], axis=1)
    low = np.min(prcSoFar[:, -ATR_period:], axis=1)
    prev_close = prcSoFar[:, -ATR_period-1:-1]
    tr1 = high - low
    tr2 = np.abs(high - prev_close[:, -1])
    tr3 = np.abs(low - prev_close[:, -1])
    tr = np.maximum.reduce([tr1, tr2, tr3])
    ATR = tr + 1e-6  # avoid zero

    for i in range(nins):
        # --- Entry Logic ---
        if positionSide[i] == 0:
            # Long breakout
            if price[i] > upper[i]:
                positionSide[i] = 1
                entryPrice[i] = price[i]
                stopPrice[i] = price[i] - stop_multiplier * ATR[i]
            # Short breakout
            elif price[i] < lower[i]:
                positionSide[i] = -1
                entryPrice[i] = price[i]
                stopPrice[i] = price[i] + stop_multiplier * ATR[i]

        # --- Exit Logic ---
        if positionSide[i] == 1:
            # Exit long if price < middle band or hit stop
            if price[i] < middle[i] or price[i] < stopPrice[i]:
                positionSide[i] = 0
                entryPrice[i] = np.nan
                stopPrice[i] = np.nan
        elif positionSide[i] == -1:
            # Exit short if price > middle band or hit stop
            if price[i] > middle[i] or price[i] > stopPrice[i]:
                positionSide[i] = 0
                entryPrice[i] = np.nan
                stopPrice[i] = np.nan

        # --- ATR Trailing Stop Update ---
        if positionSide[i] == 1:
            stopPrice[i] = max(stopPrice[i], price[i] - stop_multiplier * ATR[i])
        elif positionSide[i] == -1:
            stopPrice[i] = min(stopPrice[i], price[i] + stop_multiplier * ATR[i])

        # --- Position Sizing ---
        if positionSide[i] != 0:
            risk_per_trade = equity * risk_pct
            if positionSide[i] == 1:
                stop_dist = price[i] - stopPrice[i]
            else:
                stop_dist = stopPrice[i] - price[i]
            if stop_dist > 0:
                size = int(risk_per_trade / stop_dist)
                newPos[i] = positionSide[i] * size
            else:
                newPos[i] = 0
        else:
            newPos[i] = 0

    currentPos = newPos
    return currentPos