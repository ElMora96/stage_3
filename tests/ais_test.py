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
plt.rc('figure',figsize=(16,12))
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
train = ndata[:"2020-07"]
#09-18 & 08-02: ok for demo
#test = ndata["2020-08-02"] 
test = negea["2020-09-13"]
compare = ndata[test.index + pd.Timedelta(1, "D")]
egea_compare = egea_forecast.loc[compare.index, zone]
#%%
#Model initializaton
model = al.AIS(train, S = 2, delta = 2.5, sigma = 3)
#%%
testAG = al.ForecastAntiGen(test, tau = 1)
#%%
#Test prediction
pred = model.predict(testAG, train = True)
#%%
#Plot Routine
err = mape(compare, testAG.output_data)
plt.plot(compare, label = "Real", color = "black", linewidth = 4, alpha = 0.5)
plt.plot(testAG.output_data, label = "Modello", color = "green", linestyle="--", linewidth = 2)
plt.plot(egea_compare, label = "Forecast Egea", color = "blue", linestyle = "-.", linewidth = 2)
plt.title("Artificial Immune System test - MAPE " +str(np.round(err,1)) +"%")
plt.ylabel("Quantità [MWh]")
plt.legend()
plt.show()