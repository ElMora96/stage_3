#STL plus Random Forest
import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import NearestNeighbors

from statsmodels.tsa.holtwinters import Holt
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.seasonal import STL

from utils_lib import es_noise_removal

class Pattern():
	#Pattern representation for time series residuals
	#each cycle is a pandas TS of length 24
	#Nodes of Singly Linked List
	def __init__(self, data, temp, solar):       
		
		#Load Residuals Values
		self.load_resid = data

		#Temperatures
		self.temp = temp #temperature cycle

		#Solar data
		self.solar = solar

		#Metadata
		self.date = self.load_resid.index[0].date()
		#Day of week
		self.weekday = self.load_resid.index[0].dayofweek 
		#Day of the year
		self.dayofyear = self.load_resid.index[0].dayofyear 
		#Month number
		self.month = self.load_resid.index[0].month
		#Period of year array encoding
		self.period = self.compute_period()

		#Pointer to next node
		self.next = None

	def compute_period(self):
		#Encode period of year
		p_1 = np.sin(2*np.pi*self.dayofyear*(1/366.0))
		p_2 = np.cos(2*np.pi*self.dayofyear*(1/366.0))
		return np.array([p_1, p_2])

class  PatternList():
	"""Minimal, Singly Linked list for Patterns"""
	def __init__(self):
		self.head = None
		self.length = 0

	def push(self, new_data): 
		new_node = Pattern(*new_data)          
		new_node.next = self.head 	          
		self.head = new_node 
		self.length += 1

	def append(self, new_data): 
		new_node = Pattern(*new_data) #unpack data tuple
		if self.head is None: 
			self.head = new_node
			self.length += 1
			return
		last = self.head 
		while (last.next): 
			last = last.next
		last.next =  new_node 
		self.length += 1

	def pop(self):
		trav = self.head
		for i in range(self.length - 2):
			trav = trav.next
		trav.next = None
		self.length -= 1


class  Predictor():
	'''Predictors for Random Forest Model, for better working of NN'''
	def __init__(self, model, query, t = None):
		'''query: Pattern -- Target to predict
		   model: ModelRF -- Parent forecasting model
		   t: int -- Forecast hour 
		'''
		#Traverse linked list up to day previous query
		trav = model.patterns.head
		while trav.next.date != query.date:
			trav = trav.next
		#Previous day load
		self.load = trav.load_resid.values
		#Forecast Temperatures
		self.target_temps = query.temp.values
		#Forecast Solar 
		self.target_solar = query.solar.values[6:21]
		#Metadata
		self.target_month = np.array(query.month).reshape(1) #label encoding
		self.target_weekday = np.array(query.weekday).reshape(1) #label encoding
		self.target_hour = np.array(t).reshape(1) #label encoding
		self.target_period = query.period
		self.target_lockdown = np.array(model.lockdown_series[query.load_resid.index[t]]).reshape(1)
		self.target_holiday = np.array(model.holiday_series[query.load_resid.index[t]]).reshape(1)

	def to_array(self):
		'''Return predictor in np.array format'''
		x_list = [self.load,
			self.target_temps,
			self.target_solar,
			self.target_period,
			#self.target_hour, #useless for forecast
			self.target_weekday,
			self.target_month,
			self.target_lockdown,
			self.target_holiday
			]
		return np.concatenate(x_list)
	
	def to_reduced_array(self):
		'''Returned Shortened (NN-friendly) predictors in np.array format'''
		x_list = [self.load,
			self.target_temps,
			self.target_solar
			]
		return np.concatenate(x_list)

class ModelRF():
	def __init__(self, time_series, temp_series, solar_series, holiday_series, lockdown_series,  n = 24, M = 100, rest = True):
		'''time_series: pd.Series with datetime index
			load TS under consideration
			temp_series : pd.Series with datetime index
			associated weather series (temperature)
			solar_series: pd.Series -- Solar Production
			holiday_series: pd.Series -- 0/1 hourly flags for holidays
			lockdown_series: pd.Series -- 0/1 hourly flags for lockdown periods
			n: int
				cycle length (default = 24)
			M: int
			Number of neighbors to consider (in nearest-neighbor generation; default = 100)
			rest: Boolean -- flag specifying wether or not to perform M-NN restriction

		'''
		#Parameters
		self.n = n #cycle length
		self.M = M #Restrict to M neighbors
		self.rest = rest
		
		#Base Data
		self.time_series = time_series
		self.temp_series = temp_series
		self.solar_series = solar_series
		self.holiday_series = holiday_series
		self.lockdown_series = lockdown_series

		#ETS decomposition of diffed time series
		decomposition = self.decompose(self.time_series)
		self.trend = decomposition.trend
		self.seasonal = decomposition.seasonal
		self.residuals = decomposition.resid
		#Generate pattern linked list of (differenced) residuals		
		self.patterns = self.generate_patterns(self.residuals, self.temp_series, self.solar_series, self.n)

		#Store Precitions (complete & ETS)
		self.full_prediciton = None
		self.trend_prediction = None
		self.seas_prediction = None
		self.resid_prediction = None

		#Partial test series for comparison
		self.test_trend = None
		self.test_season = None
		self.test_resid = None

	def decompose(self, series):
		#For later, custom, decomposition
		#Decompose time series using STL
		decomp = STL(series, period = 168, seasonal = 169).fit()
		return decomp

	def to_bit_array(self, value, array_size):
		'''
		value: int
			value to format as bit array
		array_size: int
			length of binary array (e.g. 3 for weekday, 4 for month, 5 for hour of the day)
		Representation for neural network forecasting tasks
		returns:
			binary numpy array
		'''
		val = format(value, '0' + str(array_size) + 'b')
		bit_array = np.array(tuple(int(i) for i in val))
		return bit_array

	def generate_patterns(self, resid_series, temp_series, solar_series, n):
		'''
		series: pd Series
			TS to be encoded
		n: int
			pattern length (default : 24)
		Returns patterns in linked list
		'''
		pattern_linked_list = PatternList() #generate linked list
		res_list = np.split(resid_series, len(resid_series)/n)
		temp_list = np.split(temp_series, len(temp_series)/n)
		solar_list = np.split(solar_series, len(solar_series)/n)
		for res, temp, solar in zip(res_list, temp_list, solar_list):
			pattern_linked_list.append((res, temp, solar))

		return pattern_linked_list

	def restrict_to_NN(self, X, y, query, n_neighbors):
		#Restrict neural netwotk training set to nearest neighbors; naive
		#Consider only load resid components
		X_load = np.array([vec[0:24] for vec in X])
		Neighbors = NearestNeighbors(n_neighbors, metric = 'euclidean')#kshape_lib.sbd
		Neighbors.fit(X_load)
		#traverse linked list up to day previous to query
		trav = self.patterns.head
		while trav.next.date != query.date:
			trav = trav.next
		#List of NN indices
		element = trav.load_resid.values.reshape(1,-1)
		neigh = list(Neighbors.kneighbors(element, return_distance = False))
		X = X[neigh]
		y = y[neigh]

		return X, y

	def compute_predictors(self, query, t): #RF version: no standardization
		'''
		query: Pattern
		compute predictor array for given pattern
		t: int
		target forecast hour
		'''
		X = Predictor(self, query, t)
		return X.to_array()

	def learning_set(self, query, t, restrict):
		#Training set generator for each forecasting task
		'''
		query: Pattern
			Pattern of now, predict tomorrow		
		t: int
			timestep 
		restrict: Boolean
			Whether or not to restric to (M) nearest neighbors
		Returns X, y training set arrays in sklearn format.
		'''
		X, y = [],[]
		trav = self.patterns.head.next #skip first day as it has no day before
		while trav.date != query.date: #retain one more day
			#predictors
			x_val = self.compute_predictors(trav, t)
			#target
			y_val = trav.load_resid.values[t] #HERE
			
			#Store values
			X.append(x_val)
			y.append(y_val)

			trav = trav.next

		X = np.array(X)
		y = np.array(y)

		#Restrict to NN
		if restrict:
			target_index = trav.load_resid.index[t] #target time index
			if self.holiday_series[target_index] == 1:
				K = int(np.ceil(0.35 * self.M))
			else:
				K = self.M #Vary neighborhood size according to daytype
			X, y = self.restrict_to_NN(X, y, query, K) 
		return X, y

	def daily_trend_pred(self, now, delta = pd.Timedelta(3, "W")):
		train_time = pd.date_range(start = now - delta , end = now - pd.Timedelta(1, "H"), freq = "H")
		model_series = self.trend[train_time]
		h_model = Holt(model_series,
				   		exponential=True,
				   		damped_trend = True)
		#Fit
		fitted_h = h_model.fit(optimized=True)   
		#Predict tomorrow
		pred = fitted_h.predict(start = now, 
						  end = now + pd.Timedelta(23, "H"))
		return pred

	def daily_season_pred(self, now, delta = pd.Timedelta(3, "W")):
		train_time = pd.date_range(start = now - delta , end = now - pd.Timedelta(1, "H"), freq = "H")
		model_series = self.seasonal[train_time]
		hw_model = ExponentialSmoothing(model_series,
									trend = "add",
									seasonal='add', #multiplicative
									seasonal_periods= 168 ,
									initialization_method = 'estimated',
									freq='H',
									)
		fitted_hw = hw_model.fit()
		pred = fitted_hw.predict(start = now,
									end = now + pd.Timedelta(23, "H"))

		return pred

	def hourly_resid_pred(self, query, t):
		'''
		query: pattern
		predict following day load at time t
		t: int
		hour of day
		'''
		#Local RF model to predict residuals
		model = RandomForestRegressor(n_estimators = 300, #100
									   max_features = 'auto', #17
									   bootstrap = True,
									   max_samples = 0.5, #low bootstrap size 0.3
									   max_depth =None, #5
									   min_samples_split = 2,
									   n_jobs = 4,
									   ccp_alpha = 0.1,
									   verbose = 0)
		#Learrning sets
		X, y = self.learning_set(query, t, restrict = self.rest)
		#Model
		model = model.fit(X, y)
		#Compute predictor
		predictor = self.compute_predictors(query, t).reshape(1,-1)
		#Predict
		prediction = model.predict(predictor)

		return prediction	

	def predict(self, test_time_range, recursive = False):
		'''
		test_time_range: datetime index --- Forecast days specfied by test_time_range
		recursive: Boolean --- Flag to run recursive prediction 
		'''
		#Boundaries for test
		first_date = test_time_range[0].date()
		last_date = test_time_range[-1].date()
		#Prediction Routine
		#test_patterns = self.test_list(test_time_range) #Test patterns (sub) linked list
		predictions = [] #list to store daily predictions
		trend_pred_series = pd.Series() #Empty series to store trend predictions
		seas_pred_series = pd.Series() #Empty smeries to store seas predictions
		trav = self.patterns.head

		k = 1 #Prediction day counter
		#traverse liste up to first test date
		while trav.date != first_date:
			trav = trav.next

		while trav and trav.date != last_date + pd.Timedelta(1, "D"):
			print("Predicting day ", k)
			#print(trav.load_resid.index[0])
			r_pred = []
			#Hourly predictions - Resids
			for t in range(24): #iterate over time
				#Predict residuals
				t_pred = self.hourly_resid_pred(trav, t)
				#Store prediction
				r_pred.append(t_pred)
			
			#Daily prediction - residuals
			r_pred = np.array(r_pred).reshape(24) #Cast as array
			#Compute trend & seasonalty predictions
			t_pred = self.daily_trend_pred(trav.load_resid.index[0])
			s_pred = self.daily_season_pred(trav.load_resid.index[0])
			r_pred = pd.Series(r_pred, index = trav.load_resid.index)
			next_day = r_pred + t_pred + s_pred#three-component prediction
		
			#Recursive forecast
			if recursive:
				self.trend[trav.load_resid.index] = t_pred.values
				self.seasonal[trav.load_resid.index] = s_pred.values
				self.residuals[trav.load_resid.index] = r_pred.values
				trav.load_resid = r_pred
			'''
			#Semi-recursive forecast: average recursion and egea forecast
			if recursive:
				self.trend[trav.load_resid.index] = 0.5*(t_pred.values + self.trend[trav.load_resid.index])
				self.seasonal[trav.load_resid.index] = 0.5*(s_pred.values + self.seasonal[trav.load_resid.index])
				self.residuals[trav.load_resid.index] = 0.5*(r_pred.values + self.residuals[trav.load_resid.index])
				trav.load_resid = 0.5*(r_pred + trav.load_resid)
				'''				 
			#Store prediction
			predictions.append(next_day) #Full
			trend_pred_series = pd.concat([trend_pred_series, t_pred])
			seas_pred_series = pd.concat([seas_pred_series, s_pred])
			trav = trav.next
			k += 1
		#Convert back to unique series; add index
		full_pred_series = pd.concat(predictions)
		#Store predicitions 
		self.full_prediciton = full_pred_series
		self.trend_prediction = trend_pred_series
		self.seas_prediction = seas_pred_series
		self.resid_prediction = full_pred_series - trend_pred_series - seas_pred_series

		return self.full_prediciton