import sys
sys.path.append("D://Users//F.Moraglio//Documents//python_forecasting//stage_3//libs")
import ais_lib as al
from utils_lib import mape
import pandas as pd
import numpy as np
#Plots
import matplotlib.pyplot as plt
import seaborn as sns
from pandas.plotting import register_matplotlib_converters

#Warnings
'''
import warnings
warnings.filterwarnings("ignore")
'''

#Plot configuration
plt.ioff()
register_matplotlib_converters()
sns.set_style('darkgrid')
plt.rc('figure',figsize=(32,24))
plt.rc('font', size=16)
#%%
data = pd.read_csv("D:/Users/F.Moraglio/Documents/python_forecasting/stage_2/data/transformed/load.csv",
					sep = ";", #specify separator
					parse_dates = True,
					dayfirst= True, #To parse
					decimal=",",
					index_col = 0,
					squeeze = True
					) 
egea_forecast =  pd.read_csv("D:/Users/F.Moraglio/Documents/python_forecasting/stage_2/data/transformed/forecast.csv",
					sep = ";", #specify separator
					parse_dates = True,
					#NO DAYFIRST!
					decimal=",",
					index_col = 0,
					squeeze = True
					) 
#Adjust Egea forecast datetime index
egea_forecast.index = pd.to_datetime(egea_forecast.index, utc=True)
egea_forecast = egea_forecast.tz_convert("UTC")
#%%
zone = 'NORD'
ndata = data[zone]
negea = egea_forecast[zone]
train = ndata["2018-08":"2020-07"]
#%%
#Parallelized version
#Dudek suggests sigma = 1.9, 3>= delta >=2, c almost 1
parallel_model = al.ParallelAIS(data = train, delta = 3, c = 0.95, max_iter=15, sigma = 2, S=1)
#%%
testseries = negea["2020-09-01":"2020-11-01"]
#Predict
forecast = parallel_model.predict(testseries) 
#Comparison
true_load = ndata[forecast.index]
egea_forecast = negea[forecast.index]
err = mape(true_load, forecast)
plt.plot(true_load, label = "Real", color = "black", linewidth = 4, alpha = 0.5)
plt.plot(forecast, label = "Modello", color = "green", linestyle="--", linewidth = 2)
plt.plot(egea_forecast, label = "Forecast Egea", color = "blue", linestyle = "-.", linewidth = 2)
plt.title("Artificial Immune System test - MAPE " +str(np.round(err,1)) +"%")
plt.ylabel("Quantità [MWh]")
plt.legend()
plt.show()