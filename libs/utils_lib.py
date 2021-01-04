import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error
from itertools import groupby
import datetime
import calendar

###ERRORS###

#Compute mean absolute percentage errore (MAPE)
def mape(y_true, y_pred): 
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

#Compute symmetric MAPE (sMAPE)
def smape(y_true, y_pred):
	y_true, y_pred = np.array(y_true), np.array(y_pred)
	return np.mean(np.abs(y_true - y_pred)/(np.abs(y_true) + np.abs(y_pred))) * 200

#Compute rooted mean squared error (RMSE)
def rmse(y_true, y_pred):
	y_true, y_pred = np.array(y_true), np.array(y_pred)
	return np.sqrt(mean_squared_error(y_true, y_pred))

#Compute mean absolute error (MAE)
def mae(y_true, y_pred):
	y_true, y_pred = np.array(y_true), np.array(y_pred)
	return mean_absolute_error(y_true, y_pred)


###TS UTILITIES###

#Create differenced series
def difference(dataset, interval=1):
	diff = list()
	new_index = dataset.index[interval:]
	for i in range(interval, len(dataset)):
		value = dataset[i] - dataset[i - interval]
		diff.append(value)
	return pd.Series(diff, index = new_index)

#Reverse differencing of series
def reverse_difference(dataset, beginning):
	period = len(beginning)
	inv_diff = np.empty(len(dataset) + period)
	new_index = beginning.index.union(dataset.index)
	for i in range(len(inv_diff)):
		if i < period:
			inv_diff[i] = beginning[i]
		else:
			inv_diff[i] = dataset[i - period] + inv_diff[i - period]	
	return pd.Series(inv_diff, index = new_index)
 
#Perform exponential filtering
def es_noise_removal(series, alpha = 0.9):
	#alpha: float in (0,1) - smoothing parameter
	sm_series = pd.Series(index = series.index, dtype = 'float64')
	sm_series[0] = alpha*series[0]
	for j in range(1, len(series)):
		sm_series[j] = alpha*series[j] + (1 - alpha)*sm_series[j-1]
	return sm_series


###MISC###

#Check if all elements in iterable are equal
def all_equal(iterable):
    g = groupby(iterable)
    return next(g, True) and not next(g, False)


#Add months to given datetime
def add_months(sourcedate, months):
    month = sourcedate.month - 1 + months
    year = sourcedate.year + month // 12
    month = month % 12 + 1
    day = min(sourcedate.day, calendar.monthrange(year,month)[1])
    return datetime.date(year, month, day)