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
#Primo lockdown, molesto
lockdown[lockdown_index_1] = 1
#Nuovo lockdown, meno molesto
lockdown[lockdown_index_2] = 0.5


#Holydays 
#2018
#Prima settimana dell'anno
holiday["2018-01-02":"2018-01-05"] = 0.5
#Settimana di pasqua
holiday["2018-04-03":"2018-04-07"] = 0.5
#Ponte 25 aprile
holiday["2018-04-26":"2018-04-28"] =0.5
#Ponte primo maggio
holiday["2018-04-30"] = 0.7
#Primo giugno
holiday["2018-06-01"] = 0.3
#Settimana di Ferragosto
holiday["2018-08-13":"2018-08-18"] = 0.5
#Ponte dei santi
holiday["2018-11-02"] = 0.5
#Settimana di Natale
holiday["2018-12-24":"2018-12-19"] = 0.5

#2019
#Prima settimana dell'anno
holiday["2019-01-02":"2019-01-05"] = 0.5
#Settimana di pasqua e 25 aprile
holiday["2019-04-23":"2019-04-27"] = 0.5
#Snitch ponte 1 maggio
holiday["2019-04-29":"2019-04-30"] = 0.25
#Settimana di ferragosto
holiday["2019-08-12":"2019-08-17"] = 0.5
#Settimana di Natale
holiday["2019-12-23":"2019-12-28"] = 0.5
#Ultimi dell'anno
holiday["2019-12-30":"2019-12-31"] = 0.5

#2020
#Prima settimana dell'anno
holiday["2020-01-02":"2020-01-04"] = 0.5
#Settimana di Pasqua - lockdwon
holiday["2020-04-14":"2020-04-17"] = 0.25
#Ponte primo giugnp
holiday["2020-06-01"] = 0.5
#Settimana di ferragosto
holiday["2020-08-10":"2020-08-14"] = 0.5
#Settimana dopo ferragosto
holiday["2020-08-17":"2020-08-21"] = 0.5
#Ponte immacolata
holiday["2020-12-07"] = 0.5
#Settimana di Natale
holiday["2020-12-21":"2020-12-24"] = 0.5
#Ultimi dell'anno
holiday["2020-12-28":"2020-12-31"] = 0.5
 
#Feste vere e proprie
feste = {dt.date(2018,1,1), dt.date(2018,1,6), dt.date(2018,4,1), dt.date(2018,4,2), dt.date(2018,4,25), dt.date(2018,5,1), dt.date(2018,6,2), dt.date(2018,8,15), dt.date(2018,11,1), dt.date(2018, 12, 8), dt.date(2018,12,25), dt.date(2018,12,26),
		dt.date(2019,1,1), dt.date(2019,1,6), dt.date(2019,4,21), dt.date(2019,4,22), dt.date(2019,4,25), dt.date(2019,5,1), dt.date(2019,6,2), dt.date(2019,8,15), dt.date(2019,11,1), dt.date(2019, 12, 8), dt.date(2019,12,25), dt.date(2019,12,26),
		dt.date(2020,1,1), dt.date(2020,1,6), dt.date(2020,4,12), dt.date(2019,4,13), dt.date(2020,4,25), dt.date(2020,5,1), dt.date(2020,6,2), dt.date(2020,8,15), dt.date(2020,11,1), dt.date(2020, 12, 8), dt.date(2020,12,25), dt.date(2020,12,26),			
		}
for index, _ in holiday.iteritems():
	if index.date() in (feste):
		holiday[index] = 1 #flag holydays


#Cast to UTC
holiday = holiday.tz_convert("UTC")
lockdown = lockdown.tz_convert("UTC")
#Write to csv
holiday.to_csv(target_dir + "holiday.csv", sep = ";", decimal = ",")
lockdown.to_csv(target_dir + "lockdown.csv", sep = ";", decimal = "," )
