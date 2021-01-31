#Testing trend model
import sys
sys.path.append("D://Users//F.Moraglio//Documents//python_forecasting//stage_3//libs")
import trend_lib as tl
from utils_lib import  mape
import pandas as pd
import numpy as np
from statsmodels.tsa.seasonal import STL

#Plots
import matplotlib.pyplot as plt
import seaborn as sns
from pandas.plotting import register_matplotlib_converters
plt.ioff()
register_matplotlib_converters()
sns.set_style('darkgrid')
plt.rc('figure',figsize=(16,12))
plt.rc('font', size=16)

#Warnings
import warnings
warnings.filterwarnings("ignore")

#%%
#Load Datasets
temp = pd.read_csv("D:/Users/F.Moraglio/Documents/python_forecasting/stage_2/data/transformed/temperatures.csv",
					sep = ";", #specify separator
					parse_dates = True,
					dayfirst= True, #To parse
					decimal=",",
					index_col = 0,
					squeeze = True
					) 
temp = temp.tz_convert("UTC")

solar = pd.read_csv("D:/Users/F.Moraglio/Documents/python_forecasting/stage_2/data/transformed/solar.csv",
					sep = ";", #specify separator
					parse_dates = True,
					dayfirst= True, #To parse
					decimal=",",
					index_col = 0,
					squeeze = True
					) 
solar = solar.tz_convert("UTC")

#Forecast Aziendale
egea_forecast =  pd.read_csv("D:/Users/F.Moraglio/Documents/python_forecasting/stage_2/data/transformed/forecast.csv",
					sep = ";", #specify separator
					parse_dates = True,
					#NO DAYFIRST!
					decimal=",",
					index_col = 0,
					squeeze = True
					)
egea_forecast.index = pd.to_datetime(egea_forecast.index, utc=True)
egea_forecast = egea_forecast.tz_convert("UTC") 

#Consumi da fatture terna
true_load = pd.read_csv("D:/Users/F.Moraglio/Documents/python_forecasting/stage_2/data/transformed/fatture.csv",
					sep = ";", #specify separator
					parse_dates = True,
					dayfirst= True, #To parse
					decimal=",",
					index_col = 0,
					squeeze = True
					)
true_load = true_load.tz_localize("Europe/Rome", ambiguous="infer")
true_load = true_load.tz_convert("UTC") 

#Flag vacanze
holiday = pd.read_csv("D:/Users/F.Moraglio/Documents/python_forecasting//data/flags/holiday.csv",
					sep = ";", #specify separator
					parse_dates = True,
					dayfirst= True, #To parse
					decimal=",",
					index_col = 0,
					squeeze = True,
					)

#Flag lockdown
lockdown = pd.read_csv("D:/Users/F.Moraglio/Documents/python_forecasting//data/flags/lockdown.csv",
					sep = ";", #specify separator
					parse_dates = True,
					dayfirst= True, #To parse
					decimal=",",
					index_col = 0,
					squeeze = True,
					)

#%%
#Consider zone of interest
zone = 'NORD'
z_temp = temp[zone]
z_solar = solar[zone]
z_egea = egea_forecast[zone]
z_true = true_load[zone]

#Generate model load dataset according to desired forecast
last_bill = "2020-07" #last bill to consider (Month N-2)
last_date = "2020-12-13 23:00:00+00:00" #Last available date in load dataset. RK specify it in UTC format
true_base = z_true[:last_bill] #Take all "available" bills 
forecast_completion = z_egea[true_base.index[-1] + pd.Timedelta(1,"H"): last_date] #Complete with corporate forecast
z_load = pd.concat([true_base, forecast_completion], axis = 0)[1:] #Join to obtain model load

#Leave out old values - ensure have same length for all model datasets
first_date = "2018-01-01 00:00:00+00:00"
z_temp = z_temp[first_date:last_date]
z_solar = z_solar[first_date:last_date]
z_load = z_load[first_date:last_date]

#%%
#Perform decomposition to extract trend
load_list = [z_load, z_egea, z_true]
trend_list = [] #store extracted trends
for load in load_list:
	#load = (load - load.mean())/load.std() #standardize
	dec_1 = STL(load, period = 24, seasonal = 25).fit()
	intermediate = dec_1.resid + dec_1.trend
	dec_2 = STL(intermediate, period = 168, seasonal = 169).fit()
	trend = dec_2.resid + dec_2.trend
	trend_list.append(trend)

z_load, z_egea, z_true = trend_list #Overwrite

#%%
#Test set
test_range = pd.date_range(start = '2020-10-02 00:00:00+00:00',
						   end = '2020-10-02 23:00:00+00:00',
						   freq = 'H',
						   tz = 'UTC'
						   )
true_series = z_true[test_range]
egea_series = z_egea[test_range]

#Model & Predicion
model = tl.ModelTrend(z_load, z_temp, z_solar, holiday, lockdown, M = 75, rest = False )
pred_series = model.predict(test_range, recursive = False)
#%%
#Feature importances
FI = model.feat_imp
for i, test in enumerate(FI):
	vec = test[-7:]
	plt.bar(range(1,8), vec)
	tot = np.sum(vec)
	plt.title("Metadata " +str(i) + " Total " + str(tot))
	plt.show()

#%%
#Evaluation & Plot Routine
full_err = mape(true_series, pred_series)
egea_err = mape(true_series, egea_series)


plt.plot(true_series, label = "Consuntivo", color = "black", linewidth = 4, alpha = 0.5)
plt.plot(pred_series, label = "Modello", color = "red", linestyle="--", linewidth = 2)
plt.plot(egea_series, label = "Egea", color = "blue", linestyle = "-.", linewidth = 2)
plt.ylabel("Quantità [MWh]")
plt.title("Test Modello Trend NN  - "+str(zone)+"\nErrore Medio Modello: " + 
		  str(np.round(full_err, 1))+ "%\n Errore Medio Egea (Teorico): " + str(np.round(egea_err, 1)) +"%")
plt.legend()
plt.show()
	