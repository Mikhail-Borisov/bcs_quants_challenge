# -*- coding: utf-8 -*-
# Это упрощенная версия бэктеста, который используется на сервере https://bcsquants.com
# Разработчик Дмитрий Ивановский <dima-iv@mail.ru>

from __future__ import print_function
from datetime import datetime as dt
import numpy as np
import csv, os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as plotticker

orderList = []
orderEvent = False
tickSizeToSeconds = {'s1': 1, 's5': 5, 'm1': 60, 'm5': 300}
dataKeys = ['time', 'open', 'high', 'low', 'close', 'volume', 'count']
tickKeys = ['direct', 'takeProfit', 'stopLoss', 'holdPeriod', 'datetime']
FEE = 0.0002 # комиссия
allTickers = ['ALRS', 'SNGS', 'MGNT', 'ROSN', 'MOEX', 'VTBR', 'LKOH', 'GAZP', 'SBERP', 'SBER', # акции
              'USD000UTSTOM', # валюта
              'RTSI', 'MICEXINDEXCF', # индексы
              'GZX', 'SIX', 'BRX'] # фьючерсы
_tickSize = 'm5'

FIRST_TEST_DAYTIME = dt(2015, 1, 5, 10, 0, 0)
LAST_TEST_DAYTIME = dt(2016, 8, 31, 18, 35, 0)

def showBacktestResult(result):
    return pd.DataFrame(result, index=[x['ticker'] for x in result],
                                columns=['sumProcent', 'std', 'sharpe',
                                         'minV', 'numDeals',
                                         'sumTakeProfit', 'sumHoldPeriod', 'sumStopLoss'])

def getBacktestResult(init, tick, tickers=allTickers, skipMessage=False, progressBar=True):
    result = []
    if not isinstance(tickers, list):
        tickers = [tickers]
    for ticker in tickers:
        orderList, orderEvent = [], False
        _tickSize, orderList, data = runTick(init, tick, ticker)
        res = runOrder(ticker, _tickSize, orderList, data)
        res['ticker'] = ticker
        res['_tickSize'] = _tickSize
        result.append(res)
        if progressBar:
            print(ticker, end='\n' if ticker == tickers[-1] else ', ')

    if not skipMessage:
        print('Не забудьте вы можете посмотреть тики и заявки в соответствующих файлах')
        print('tickFile = data/order/TICKER_{0}_tick.csv'.format(_tickSize))
        print('orderFile = data/order/TICKER_{0}_order.csv'.format(_tickSize))

    return result

def order(direct, takeProfit, stopLoss, holdPeriod):
    global orderList, orderEvent, tickSizeToSeconds, _tickSize
    if not isinstance(holdPeriod, int):
        raise Exception('Hold period must be int type. If you use division with operator /, ' +
                        'remember in python3 this operation converts result to float type, ' +
                        'use // instead or convert to int directly')
    if holdPeriod * tickSizeToSeconds[_tickSize] < 300:
        raise Exception('Hold period must be not less than 300 seconds')
    if takeProfit < 0.0004:
        raise Exception('Take profit must be not less than 0.0004')
    if stopLoss < 0.0004:
        raise Exception('Stop loss must be not less than 0.0004')
    orderList.append([direct, takeProfit, stopLoss, holdPeriod])
    orderEvent = True

def runTick(init, tick, ticker):
    global orderList, orderEvent, _tickSize
    orderList, orderEvent = [], False

    class Empty:
        pass
    self = Empty()
    init(self)

    _tickSize = getattr(self, '_tickSize', 'm5')
    _window = getattr(self, '_window', None)
    if _window is not None:
        _window = int(_window)

    data = {key: np.load('data/{0}/{1}/{2}.npy'.format(_tickSize, ticker, key), encoding='bytes') for key in dataKeys }

    for ind in range(1, len(data['time'])):
        if data['time'][ind] < FIRST_TEST_DAYTIME:
            continue
        if data['time'][ind] > LAST_TEST_DAYTIME:
            break

        if _window:
            if ind < _window:
                continue
            else:
                tick(self, { key: data[key][ind - _window:ind] for key in dataKeys })
        else:
            tick(self, { key: data[key][:ind] for key in dataKeys })

        if orderEvent:
            for jnd in range(len(orderList) - 1, -1, -1): # [len(orderList) - 1, ..., 0]
                if len(orderList[jnd]) == 4:
                    orderList[jnd].append(data['time'][ind])
                else:
                    break
            orderEvent = False

    if not os.path.exists('data/order'):
        os.makedirs('data/order')
    with open('data/order/{0}_{1}_tick.csv'.format(ticker, _tickSize), 'w') as file:
        file.write(';'.join(tickKeys) + '\n')
        for order in orderList:
            file.write(';'.join([str(elem) for elem in order]) + '\n')

    return _tickSize, orderList, data

def runOrder(ticker, _tickSize, orderList, dataNpy):
    measure = {'deals': [], 'sumProcent': 0.0, 'sumTakeProfit': 0, 'sumStopLoss': 0, 'sumHoldPeriod': 0, 'numDeals': 0}
    currentDataNum, firstTime, preLastCandle = -1, True, False

    sharpeArray = []
    sharpeOrderArray = [[0., None]]

    for order in orderList:
        if preLastCandle:
            break
        order = dict(zip(tickKeys, order))

        mode = 'findOrder'
        if firstTime or data['time'] <= order['datetime']:
            while (not preLastCandle) and mode != 'Exit':
                currentDataNum += 1
                if currentDataNum >= len(dataNpy['time']) - 2:
                    preLastCandle = True

                data = {key: dataNpy[key][currentDataNum] for key in dataKeys}

                if mode == 'findOrder':
                    if data['time'] >= order['datetime']:
                        priceEnter = dataNpy['close'][currentDataNum + 1]
                        numEnter = currentDataNum + 1
                        datetimeEnter = dataNpy['time'][currentDataNum + 1]
                        mode = 'doOrder'
                elif mode == 'doOrder':
                    if order['direct'] == 'buy':
                        directOrder = 1
                    else:
                        directOrder = -1

                    datetimeExit = dataNpy['time'][currentDataNum + 1]
                    nextClose = dataNpy['close'][currentDataNum + 1]
                    procentUp = data['high'] / priceEnter - 1.
                    procentDown = data['low'] / priceEnter - 1.
                    holdPeriod = order['holdPeriod']
                    isPreLastTestDatetime = (datetimeExit == LAST_TEST_DAYTIME)
                    isHoldPeriod = preLastCandle or (currentDataNum - numEnter + 1 > holdPeriod)

                    if data['time'].second == 0 and data['time'].minute == 0:
                        plOrderFromTime = directOrder * (nextClose / priceEnter - 1.)
                        sharpeOrderArray.append([plOrderFromTime, data['time']])

                    if order['direct'] == 'buy':
                        takeProfit = (procentUp >= order['takeProfit'])
                        stopLoss = (procentDown <= -order['stopLoss'])
                    else: # order['direct'] == 'sell'
                        takeProfit = (procentDown <= -order['takeProfit'])
                        stopLoss = (procentUp >= order['stopLoss'])

                    if takeProfit or stopLoss or isHoldPeriod or isPreLastTestDatetime:
                        event = 'holdPeriod'
                        nextClose = dataNpy['close'][currentDataNum + 1]
                        direct = {'buy': 1, 'sell': -1}[order['direct']]
                        procent = (nextClose / priceEnter - 1.) * direct - 2 * FEE
                        if takeProfit:
                            event = 'takeProfit'
                        if stopLoss:
                            event = 'stopLoss'
                        measure['deals'].append({
                            'procent': procent,
                            'event': event,
                            'direct': order['direct'],
                            'datetimeEnter': datetimeEnter,
                            'datetimeExit': datetimeExit,
                            'priceEnter': priceEnter,
                            'priceExit': nextClose,
                            'datetimeEnterInd': numEnter,
                            'datetimeExitInd': currentDataNum + 1,
                        })
                        mode = 'Exit'

                        if not(data['time'].second == 0 and data['time'].minute == 0):
                            plOrderFromTime = directOrder * (nextClose / priceEnter - 1.)
                            sharpeOrderArray.append([plOrderFromTime, data['time']])
                        sharpeOrderArray[0][0] -= FEE
                        sharpeOrderArray[-1][0] -= FEE
                        sharpeArray += [(sharpeOrderArray[ind][0] - sharpeOrderArray[ind - 1][0], sharpeOrderArray[ind][1])
                                        for ind, _ in enumerate(sharpeOrderArray) if ind > 0]
                        sharpeOrderArray = [[0., None]]

        firstTime = False

    mapEventDirect = {'takeProfit': 'sumTakeProfit', 'holdPeriod': 'sumHoldPeriod', 'stopLoss': 'sumStopLoss'}

    measure['sharpeArray'] = sharpeArray

    portfolio = {
        'deals': [],
        'sharpeArray': [0.]
    }
    for deal in measure['deals']:
        portfolio['deals'].append(deal['procent'])
        measure[mapEventDirect[deal['event']]] += deal['procent']
    for delta, time in sharpeArray:
        portfolio['sharpeArray'].append(delta)

    def calcMeasures(portfolio):
        deals = portfolio['deals']
        sharpeArray = portfolio['sharpeArray']
        res = {};
        pnl = np.cumsum(deals)

        res['std'] = 0.
        if sharpeArray:
            res['std'] = np.std(sharpeArray)

        res['minV'] = min(np.min(pnl), 0)
        #res['maxDrawdown'] = maxDrawdown(pnl)
        res['sumProcent'] = pnl[-1]
        res['numDeals'] = len(deals)

        res['sharpe'] = 0.
        if res['std'] > 0:
            res['sharpe'] = np.average(sharpeArray) / res['std']

        return res

    measure['sumProcent'] = measure['minV'] = measure['maxDrawdown'] = 0
    measure['std'] = measure['numDeals'] = 0
    if portfolio:
        measureTest = calcMeasures(portfolio)
        measure.update(measureTest)

    toCSV = [deal for deal in measure['deals']]
    fieldnames = ['datetimeEnter', 'direct', 'priceEnter', 'procent', 'event', 'datetimeExit', 'priceExit']
    with open('data/order/{0}_{1}_order.csv'.format(ticker, _tickSize), 'w') as output_file:
        dict_writer = csv.DictWriter(output_file, fieldnames=fieldnames, delimiter=';', extrasaction='ignore')
        dict_writer.writeheader()
        dict_writer.writerows(toCSV)

    def myKeyView(key, value):
        value = value or 0
        if key in ("sumProcent",
                   "maxDrawdown", "std", "minV",
                   "sumTakeProfit", "sumHoldPeriod", "sumStopLoss"):
            if value:
                return round(float(value) * 100, 2)
        if key in ("testResult", "controlResult", 'sharpe',):
            if value:
                return round(float(value), 2)
        return value

    return {key:myKeyView(key, measure[key]) for key in measure}

def plotChart(result, ticker):
    for res in result:
        if res['ticker'] == ticker:
            break
    _tickSize = res['_tickSize']

    data = {key: np.load('data/{0}/{1}/{2}.npy'.format(_tickSize, ticker, key), encoding='bytes') for key in dataKeys }

    N = len(data['time'])
    ind = np.arange(N)
    def format_date(x, pos=None):
        thisind = np.clip(int(x + 0.5), 0, N - 1)
        return data['time'][thisind].strftime('%Y-%m-%d %H:%M:%S')
    fig, ax = plt.subplots()
    ax.plot(ind, data['close'], 'b-')

    ax.plot([x['datetimeEnterInd']
              for ind, x in enumerate(res['deals'])
              if x['direct'] == 'buy'],
             [x['priceEnter']
              for x in res['deals']
              if x['direct'] == 'buy'],
             'go', marker='^', ms=10)

    ax.plot([x['datetimeEnterInd']
              for x in res['deals']
              if x['direct'] == 'sell'],
             [x['priceEnter']
              for x in res['deals']
              if x['direct'] == 'sell'],
             'ro', marker='v', ms=15)

    ax.plot([x['datetimeExitInd']
              for x in res['deals']
              if x['event'] == 'takeProfit'],
             [x['priceExit']
              for x in res['deals']
              if x['event'] == 'takeProfit'],
             'go', marker='$ P $', ms=15)

    ax.plot([x['datetimeExitInd']
              for x in res['deals']
              if x['event'] == 'stopLoss'],
             [x['priceExit']
              for x in res['deals']
              if x['event'] == 'stopLoss'],
             'ro', marker='$ S $', ms=15)

    ax.plot([x['datetimeExitInd']
              for x in res['deals']
              if x['event'] == 'holdPeriod' and x['procent'] > 0],
             [x['priceExit']
              for x in res['deals']
              if x['event'] == 'holdPeriod' and x['procent'] > 0],
             'go', marker='$ H $', ms=15)

    ax.plot([x['datetimeExitInd']
              for x in res['deals']
              if x['event'] == 'holdPeriod' and x['procent'] <= 0],
             [x['priceExit']
              for x in res['deals']
              if x['event'] == 'holdPeriod' and x['procent'] <= 0],
             'ro', marker='$ H $', ms=15)

    ax.xaxis.set_major_formatter(plotticker.FuncFormatter(format_date))
    fig.autofmt_xdate()

    plt.show()