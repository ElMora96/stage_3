 #Script to generate Holiday Flag Series and lockdown flag series
#Libs
import pandas as pd
import  datetime as dt
import numpy as np 
#Setup
target_dir = "D:/Users/F.Moraglio/Documents/python_forecasting/data/flags/"
time_index = pd.date_range(start="2018-01-01 00:00", 
							end="2020-12-31 23:00",
							freq="H",
							tz="Europe/Rome")

#Initialize	
initial_vals_holiday, initial_vals_lockdown = np.zeros(len(time_index)), np.zeros(len(time_index)) #pay attention to this initialization
holiday = pd.Series(initial_vals_holiday, index = time_index)
lockdown = pd.Series(initial_vals_lockdown, index = time_index) 

#Lockdown
lockdown_index_1 = pd.date_range(start="2020-03-09 00:00", 
								end="2020-05-18 23:00",
								freq="H",
								tz="Europe/Rome")
lockdown_index_2 = pd.date_range(start="2020-11-05 00:00", 
								end="2020-12-31 23:00",
								freq="H",
								tz="Europe/Rome")
lockdown_index = lockdown_index_1.union(lockdown_index_2)
lockdown[lockdown_index] = 1

#Holydays 
feste = {dt.date(2018,1,1), dt.date(2018,1,6), dt.date(2018,4,1), dt.date(2018,4,2), dt.date(2018,4,25), dt.date(2018,5,1), dt.date(2018,6,2), dt.date(2018,8,15), dt.date(2018,11,1), dt.date(2018, 12, 8), dt.date(2018,12,25), dt.date(2018,12,26),
		dt.date(2019,1,1), dt.date(2019,1,6), dt.date(2019,4,21), dt.date(2019,4,22), dt.date(2019,4,25), dt.date(2019,5,1), dt.date(2019,6,2), dt.date(2019,8,15), dt.date(2019,11,1), dt.date(2019, 12, 8), dt.date(2019,12,25), dt.date(2019,12,26),
		dt.date(2020,1,1), dt.date(2020,1,6), dt.date(2020,4,12), dt.date(2019,4,13), dt.date(2020,4,25), dt.date(2020,5,1), dt.date(2020,6,2), dt.date(2020,8,15), dt.date(2020,11,1), dt.date(2020, 12, 8), dt.date(2020,12,25), dt.date(2020,12,26),			
		}


for index, _ in holiday.iteritems():
	if index.date() in (feste):
		holiday[index] = 1 #flag holydays
	if index.weekday() == 6:
		holiday[index] = 0.75 #Flag sunday

#Set settimana di ferragosto as holiday
holiday["2020-08-10":"2020-08-16"] = 0.5

#Cast to UTC
holiday = holiday.tz_convert("UTC")
lockdown = lockdown.tz_convert("UTC")
#Write to csv
holiday.to_csv(target_dir + "holiday.csv", sep = ";", decimal = ",")
lockdown.to_csv(target_dir + "lockdown.csv", sep = ";", decimal = "," )
