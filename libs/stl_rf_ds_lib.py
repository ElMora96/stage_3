#Double STL plus Random Forest
import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import NearestNeighbors
from sklearn.utils import resample

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
		self.temp = temp 
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
	def __init__(self, model, query, t):
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
			self.target_temps
			#self.target_solar #cluster based on solar is merda!
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
		self.trend = decomposition[0]
		self.seasonal_1 = decomposition[1]
		self.seasonal_2 = decomposition[2]
		self.residuals = decomposition[3]
		#Generate pattern linked list of (differenced) residuals		
		self.patterns = self.generate_patterns(self.residuals, self.temp_series, self.solar_series, self.n)

		#Store Precitions (complete & ETS)
		self.full_prediciton = None
		self.trend_prediction = None
		self.seas_1_prediction = None
		self.seas_2_prediction_prediction = None
		self.resid_prediction = None


	def decompose(self, series):
		#For later, custom, decomposition
		#Decompose time series using STL twice
		decomp_1 = STL(series, period = 24, seasonal = 25).fit()
		seasonal_1 = decomp_1.seasonal #weekly seasonalty
		intermediate_series = decomp_1.trend + decomp_1.resid
		decomp_2 = STL(intermediate_series, period = 168, seasonal = 169).fit()
		trend = decomp_2.trend #daily seasonalty
		seasonal_2 = decomp_2.seasonal
		resid = decomp_2.resid
		return [trend, seasonal_1, seasonal_2, resid]

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

	def restrict_trainset(self, predictors, targets, query, n_neighbors):
		'''NN restriction of RF training set.
		Parameters:
		predictors: list of Predictor
		targets: list of y-values (response variables)
		query: Pattern. -- Forecast Day of Interest
		n_neighbors: int -- Number o Neighbors for standard day
		'''
		if len(predictors) != len(targets):
			raise ValueError("Predictor list and Targets list must have same length!")
		t = int(predictors[0].target_hour) #Little bit ugly
		q_predictor = Predictor(self, query, t)
		#Sklearn euclidean Neighbors
		base_model = NearestNeighbors(n_neighbors, metric = "euclidean")
		X = [P.to_reduced_array() for P in predictors]
		X = np.array(X)
		#Basic Neighbors
		base_model.fit(X)
		q_x = q_predictor.to_reduced_array().reshape(1,-1) #Query array
		neigh = base_model.kneighbors(q_x, return_distance = False).reshape(self.M)
		predictors = [predictors[i] for i in neigh]
		targets = [targets[i] for i in neigh]
		#Holiday filtering - to be adjusted
		'''
		if q_predictor.target_holiday > 0:
			neigh_2 = []
			for ix, item in enumerate(predictors):
				if item.target_holiday > 0:
					neigh_2.append(ix) #retain maledetti
			predictors = [predictors[i] for i in neigh_2]
			targets = [targets[i] for i in neigh_2]
		'''
		#...
		#Return trainset as array
		predictors = np.array([P.to_array() for P in predictors])
		#print("Trainset length ", len(predictors))
		targets = np.array(targets)
		#Resample if dataset is too short
		if len(targets) < 0.2*self.M:
			resampling = resample(predictors, targets, n_samples=int(0.2*self.M))
			predictors, targets = resampling
			#print("New Trainset length ", len(predictors))


		return predictors, targets

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
			x_val = Predictor(self, trav, t)
			#target
			y_val = trav.load_resid.values[t] #HERE
			#Store values
			X.append(x_val)
			y.append(y_val)

			trav = trav.next
		#Perform trainset restriction
		if restrict:
			X, y = self.restrict_trainset(X, y, query, self.M)
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

	def daily_season_pred_1(self, now, delta = pd.Timedelta(2, "W")):
		#Predict weekly seasonalty day by day
		train_time = pd.date_range(start = now - delta , end = now - pd.Timedelta(1, "H"), freq = "H")
		model_series = self.seasonal_1[train_time]
		hw_model = ExponentialSmoothing(model_series,
									trend = "add",
									seasonal='add', #multiplicative
									seasonal_periods= 24 ,
									initialization_method = 'estimated',
									freq='H',
									)
		fitted_hw = hw_model.fit()
		pred = fitted_hw.predict(start = now,
									end = now + pd.Timedelta(23, "H"))

		return pred

	def daily_season_pred_2(self, now, delta = pd.Timedelta(3, "W")):
		#Predict daily seasonalty day by day
		train_time = pd.date_range(start = now - delta , end = now - pd.Timedelta(1, "H"), freq = "H")
		model_series = self.seasonal_2[train_time]
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
									   max_features = 'sqrt', #17
									   bootstrap = True,
									   max_samples = None, #low bootstrap size 0.3
									   max_depth =None, #5
									   min_samples_split = 2,
									   n_jobs = 4,
									   ccp_alpha = 0.0, #no cost-complexity pruning
									   verbose = 0)
		#Learrning sets
		X, y = self.learning_set(query, t, restrict = self.rest)
		#Model
		model = model.fit(X, y)
		#Compute predictor
		predictor = Predictor(self, query, t).to_array().reshape(1,-1)
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
		seas_1_pred_series = pd.Series() #Empty smeries to store weekly seas predictions
		seas_2_pred_series = pd.Series() #Empty smeries to store daily seas predictions
		trav = self.patterns.head

		k = 1 #Prediction day counter
		#traverse liste up to first test date
		while trav.date != first_date:
			trav = trav.next

		while trav and trav.date != last_date + pd.Timedelta(1, "D"): #Notice here 'and' SHORT CIRCUITS
			print("Predicting day ", k)
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
			t_pred = self.daily_trend_pred(trav.load_resid.index[0]) #Trend
			s_1_pred = self.daily_season_pred_1(trav.load_resid.index[0]) #Weekly seasonalty
			s_2_pred = self.daily_season_pred_2(trav.load_resid.index[0]) #Daily seasonalty
			r_pred = pd.Series(r_pred, index = trav.load_resid.index) #Resid
			next_day = r_pred + t_pred + s_1_pred + s_2_pred #four-component prediction
		
			#Recursive forecast
			if recursive:
				u = np.random.uniform(0.25, 0.75)
				averager = lambda x, y : u*x + (1-u)*y #Stochastic averaging lambda
				self.trend[trav.load_resid.index] = averager(self.trend[trav.load_resid.index], t_pred.values)
				self.seasonal_1[trav.load_resid.index] = averager(self.seasonal_1[trav.load_resid.index], s_1_pred.values)
				self.seasonal_2[trav.load_resid.index] = averager(self.seasonal_2[trav.load_resid.index], s_2_pred.values)
				self.residuals[trav.load_resid.index] = averager(self.residuals[trav.load_resid.index], r_pred.values)
				trav.load_resid = averager(trav.load_resid, r_pred)

			#Store prediction
			predictions.append(next_day) #Full
			trend_pred_series = pd.concat([trend_pred_series, t_pred])
			seas_1_pred_series = pd.concat([seas_1_pred_series, s_1_pred])
			seas_2_pred_series = pd.concat([seas_2_pred_series, s_2_pred])
			trav = trav.next
			k += 1
		#Convert back to unique series; add index
		full_pred_series = pd.concat(predictions)
		#Store predicitions 
		self.full_prediciton = full_pred_series
		self.trend_prediction = trend_pred_series
		self.seas_1_prediction = seas_1_pred_series
		self.seas_2_prediction = seas_2_pred_series
		self.resid_prediction = full_pred_series - trend_pred_series - seas_1_pred_series - seas_2_pred_series

		return self.full_prediciton