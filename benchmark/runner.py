import sys
import os
sys.path.append("D://Users//F.Moraglio//Documents//python_forecasting//stage_3//libs")
import stl_rf_lib as rf 
from utils_lib import mape
import pandas as pd
import numpy as np

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

#%%
#Parameters
last_bill = "2020-05" #last bill to consider (Month N-2) - Name for folder
first_date = "2018-01-01 00:00:00+00:00" #Leave out old values - ensure have same length for all model datasets
last_date = "2020-12-01 23:00:00+00:00" #Last available date in load dataset. RK specify it in UTC format
test_range = pd.date_range(start = '2020-07-01 00:00:00+00:00',
						   end = '2020-07-31 23:00:00+00:00', 
						   freq = 'H',
						   tz = 'UTC'
						   ) #(Month N)
fc_month = str(test_range.month[0]) #Forecast month label
#&&
#Create Directory to store results
os.mkdir('D:/Users/F.Moraglio/Documents/python_forecasting/data/'+ fc_month)
#%%
#Runner
zonelist = true_load.columns.tolist()
predlist = [] #Store predictions for later joining
for zone in zonelist: #Run forecast for each zone
	#Settings
	z_temp = temp[zone]
	z_solar = solar[zone]
	z_egea = egea_forecast[zone]
	z_true = true_load[zone]
	z_temp = z_temp[first_date:last_date]
	z_solar = z_solar[first_date:last_date]
	true_base = z_true[:last_bill]
	forecast_completion = z_egea[true_base.index[-1] + pd.Timedelta(1,"H"): last_date]
	z_load = pd.concat([true_base, forecast_completion], axis = 0)[1:]
	z_load = z_load[first_date:last_date]
	true_series = z_true[test_range]
	egea_series = z_egea[test_range]
	#Prediction
	model = rf.ModelRF(z_load, z_temp, z_solar, M = 100 )
	pred_series = model.predict(test_range, recursive = False)
	model_err = mape(true_series, pred_series)
	egea_err = mape(true_series, egea_series)
	#Plot routine
	plt.plot(true_series, label = "Consuntivo", color = "black", linewidth = 4, alpha = 0.5)
	plt.plot(egea_series, label = "Egea", color = "blue", linestyle = "-.", linewidth = 2, alpha = 1)
	plt.plot(pred_series, label = "Modello", color = "red", linestyle="--", linewidth = 2, alpha = 1)
	plt.ylabel("Quantità [MWh]")
	plt.title("Test Modello Random Forest V2 - " +str(zone)+"\nErrore Medio Modello: " + 
			  str(np.round(model_err, 1))+ "%\n Errore Medio Egea: " + str(np.round(egea_err, 1)) +"%")
	plt.legend()
	plt.savefig('D:/Users/F.Moraglio/Documents/python_forecasting/data/'+fc_month+'/'+zone+'.png')
	plt.show()
	#Write forecast to csv
	pred_series.to_csv("D:/Users/F.Moraglio/Documents/python_forecasting/data/"+fc_month+'/'+zone+".csv",
						sep = ";",
						decimal = ",")
	#Store prediction
	predlist.append(pred_series)

#%%
#Join forecasts into single dataftame
full_df = pd.DataFrame()
for zone, fcast in zip(zonelist, predlist):
	full_df[zone] = fcast

#Write to csv
full_df.to_csv("D:/Users/F.Moraglio/Documents/python_forecasting/data/"+fc_month+'/'+fc_month+".csv",
						sep = ";",
						decimal=",")