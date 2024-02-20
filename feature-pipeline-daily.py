import configparser 
from binance import Client
import pandas as pd
import numpy as np
import datetime
import hopsworks
from dotenv import load_dotenv
import os
#from binance.spot import Spot

load_dotenv()
api_key = os.getenv("HOPSWORKS_API_KEY")

project = hopsworks.login(api_key_value=api_key, project="project0")
fs = project.get_feature_store()

#read configs
config = configparser.ConfigParser()
config_file_path = "C:/Users/User/Desktop/mldl/project/config.ini"  # Replace with the actual path to your config.ini file
config.read(config_file_path)
config.read("config.ini") #The Config.ini file we just created

api_key = config["binance"]["api_key"]
api_secret = config["binance"]["api_secret"]

client = Client(api_key, api_secret)
#client = Spot()

# get today's data
today = datetime.datetime.now()

symbol = 'BTCUSDT' 

todays_kline = client.get_historical_klines(symbol, Client.KLINE_INTERVAL_1DAY, today.strftime("%d %b, %Y"))
#todays_kline = client.klines("BTCUSDT", "1d",limit = 1)

# necessary arrangements for today's data
todays_df = pd.DataFrame(todays_kline)

todays_df.columns = ["Open time","Open","High","Low","Close","Volume","Close time","Quote asset volume","Number of trades",
                   "Taker buy base asset volume","Taker buy quote asset volume","Ignore"]
                  

#We need to convert the Time from Object to datetime format
todays_df["Open time"] = pd.to_datetime(todays_df["Open time"]/1000, unit="s")
todays_df["Close time"] = pd.to_datetime(todays_df["Close time"]/1000, unit="s")

#Covert others to float datatype
numeric_columns = ["Open","High","Low","Close","Volume","Quote asset volume",
                   "Taker buy base asset volume","Taker buy quote asset volume","Ignore"]
todays_df[numeric_columns] = todays_df[numeric_columns].apply(pd.to_numeric, axis=1)

todays_df = todays_df.drop('Ignore', axis=1)
todays_df.columns = todays_df.columns.str.lower().str.replace(' ', '_')

# print(todays_df)

#Store today's data in hopsworks
btc_fg = fs.get_feature_group(name="btc",version=1)
btc_fg.insert(todays_df)
