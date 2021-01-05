#Script to generate Holiday Flag Series and lockdown flag series
#Libs
import pandas as pd
import numpy as np 
#Setup
target_dir = "D:/Users/F.Moraglio/Documents/python_forecasting/data/flags"
time_index = pd.date_range(start="2018-01-01 00:00", 
							end="2020-12-31 23:00",
							freq="H",
							tz="UTC")

#Initialize	
initial_vals_holiday, initial_vals_lockdown = np.zeros(len(time_index)), np.zeros(len(time_index)) #pay attention to this initialization
holiday = pd.Series(initial_vals_holiday, index = time_index)
lockdown = pd.Series(initial_vals_lockdown, index = time_index) 

#Lockdown
lockdown_index_1 = pd.date_range(start="2020-03-09 00:00", 
								end="2020-05-18 23:00",
								freq="H",
								tz="UTC")
lockdown_index_2 = pd.date_range(start="2020-11-05 00:00", 
								end="2020-12-31 23:00",
								freq="H",
								tz="UTC")
lockdown_index = lockdown_index_1.union(lockdown_index_2)
lockdown[lockdown_index] = 1

#Holydays 
print(holiday[-10:])