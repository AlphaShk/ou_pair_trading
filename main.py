from alpaca.trading.requests import MarketOrderRequest,OrderSide,TimeInForce,GetAssetsRequest,AssetClass
from alpaca.trading.client import TradingClient

from alpaca.data.live import StockDataStream,CryptoDataStream

from model import OU_Trading_Model
from analysis import *
from alpaca_tools import *

import yaml
        
dt_from_start = 0 
model = None

work_data = {}
status = {}
past_data = []
pair = []

trading_client = None

async def quote_data_handler(data):

    global pair,work_data,dt_from_start,model,status,past_data,trading_client,capital 

    with open("config.yaml") as stream:
        try:
            data = yaml.safe_load(stream)
            interval = data['interval']
            window = data['window']
        except yaml.YAMLError as exc:
            print(exc)

    tickers = list(work_data.keys())
    
    if not data.symbol in tickers:
        work_data[data.symbol] = [data.close]
    else: 
        work_data[data.symbol].append(data.close)
    
    status[data.symbol] = True
    
    print(work_data)

    # if len(tickers) == 2 and len(work_data[tickers[0]]) == len(work_data[tickers[1]]): # First idea

    print(status)
    if len(tickers) == 2 and all(list(status.values())): # Better at handling gaps when streaming data
        n_closes = len(past_data)
        dt_from_start += 1 

        prices = [work_data[tickers[0]][-1],work_data[tickers[1]][-1]]
        print(prices)
        past_data.append(prices)

        model.capital = trading_client.get_account().portfolio_value

        if n_closes >= window and dt_from_start % interval == 0:
            print('setup',past_data)
            model.setup(past_data)
        try: 
            signal = model.trade(prices)
            if signal: 
                spread(pair,model.is_long,[model.amount_a,model.amount_b])
            print('trade',signal,model.is_long)
        except:
            pass

        for ticker in tickers: 
            status[ticker] = False

if __name__ == '__main__':

    dt = 1/252

    base_url = 'https://paper-api.alpaca.markets' 
    api_key = ''
    secret_key = ''

    with open("config.yaml") as stream:
        try:
            data = yaml.safe_load(stream)
            api_key = data['api_key']
            secret_key = data['secret_key']
        except yaml.YAMLError as exc:
            print(exc)

    account = trading_client.get_account()
    print('Account cash:',account.cash)
    model = OU_Trading_Model(int(account.cash)*0.1)

    search_params = GetAssetsRequest(asset_class=AssetClass.US_EQUITY)
    assets = trading_client.get_all_assets(search_params)

    symbols = [asset.symbol for asset in assets if asset.tradable and asset.shortable][:10]
    print(symbols)
    price_data = get_price_data(symbols,api_key,secret_key,test=True)
    result = analyze_tickers(price_data,dt)[:10]

    for i,out in enumerate(result): 
        print(f"{i+1}. {out['t1']}-{out['t2']}: {out['likl']}")

    ind = input('Choose pair:')
    try: 
        ind = int(ind)-1 
        print(f"Trading pair: {result[ind]['t1']}-{result[ind]['t2']}")

        pair = [result[ind]['t1'],result[ind]['t2']]

        past_data = [list(price_data[pair[0]]),list(price_data[pair[1]])]
        
        # pair = ['BTC/USDT','SOL/USDT'] # For test 
        # stream = CryptoDataStream(api_key, secret_key)

        stream = StockDataStream(api_key, secret_key)
        stream.subscribe_bars(quote_data_handler, *pair)
        stream.run()
    except ValueError: 
        print('Not an integer.')


def spread(pair,long,qty):
    trading_client.close_all_positions(cancel_orders=True)

    market_order_data = MarketOrderRequest(
                        symbol=pair[0],
                        qty=qty[0],
                        side=OrderSide.BUY if long else OrderSide.SELL,
                        time_in_force=TimeInForce.DAY
                        )

    trading_client.submit_order(
                    order_data=market_order_data
                )
    
    market_order_data = MarketOrderRequest(
                        symbol=pair[1],
                        qty=qty[1],
                        side=OrderSide.SELL if long else OrderSide.BUY,
                        time_in_force=TimeInForce.DAY
                        )

    trading_client.submit_order(
                    order_data=market_order_data
                )