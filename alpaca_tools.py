from alpaca.data.historical.stock import StockBarsRequest,StockHistoricalDataClient
from alpaca.data.historical.crypto import CryptoBarsRequest,CryptoHistoricalDataClient
from alpaca.data.timeframe import TimeFrame
from datetime import datetime
import pandas as pd 

def get_1y_data(ticker,api_key,secret_key,test):
    if test: 
        request_params = CryptoBarsRequest(
                            symbol_or_symbols=ticker,
                            timeframe=TimeFrame.Day,
                            start=datetime.now().replace(year=datetime.now().day - 10),
                            end=datetime.now().replace(day=datetime.now().day - 1)
                    )
        client = CryptoHistoricalDataClient(api_key,secret_key)
        bars = client.get_crypto_bars(request_params)
        return bars.df
    else: 
        request_params = StockBarsRequest(
                        symbol_or_symbols=ticker,
                        timeframe=TimeFrame.Day,
                        start=datetime.now().replace(year=datetime.now().year - 1),
                        end=datetime.now().replace(day=datetime.now().day - 1)
                )
        client = StockHistoricalDataClient(api_key,secret_key)
        bars = client.get_stock_bars(request_params)
        return bars.df
        

def get_price_data(tickers,api_key,secret_key,test=False): 
    data = {ticker: get_1y_data(ticker,api_key,secret_key,test)['close'].to_numpy() for ticker in tickers if 'close' in get_1y_data(ticker,api_key,secret_key,test).columns}
    return data