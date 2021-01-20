#Trend forecasting using double seasonalty decomposition
import pandas as pd
import numpy as np
from  sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.neighbors import  NearestNeighbors
#MLP model
from sklearn.neural_network import MLPRegressor
from sklearn.svm import  SVR

class Pattern():
	#Pattern representation for time series residuals
	#each cycle is a pandas TS of length 24
	#Nodes of Singly Linked List
	def __init__(self, data, temp, solar):       
		
		#Load Values
		self.load = data
		#Temperatures
		self.temp = temp 
		#Solar data
		self.solar = solar

		#Metadata
		self.date = self.load.index[0].date()
		#Day of week
		self.weekday = self.load.index[0].dayofweek 
		#Day of the year
		self.dayofyear = self.load.index[0].dayofyear 
		#Month number
		self.month = self.load.index[0].month
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
		self.load = trav.load.values
		#Forecast Temperatures
		self.target_temps = query.temp.values
		#Forecast Solar 
		self.target_solar = query.solar.values[6:21] #other values are useless
		#Metadata
		self.target_month = np.array(query.month).reshape(1) #label encoding
		self.target_weekday = np.array(query.weekday).reshape(1) #label encoding
		self.target_hour = np.array(t).reshape(1) #used only in computations - completely useless in forecasting
		self.target_period = query.period
		self.target_lockdown = np.array(model.lockdown_series[query.load.index[t]]).reshape(1)
		self.target_holiday = np.array(model.holiday_series[query.load.index[t]]).reshape(1)

	def to_array(self):
		x_list = [self.load, #0:24
			self.target_temps, #24:48
			self.target_solar, #48:72
			self.target_period, #1-2
			#self.target_hour, #3 - useless in forecasting
			self.target_weekday, #4
			self.target_month, #5
			self.target_lockdown, #6
			self.target_holiday #7
			]
		return np.concatenate(x_list)
	
	def to_reduced_array(self):
		'''Returned Shortened (NN-friendly) predictors in np.array format'''
		x_list = [self.load,
			self.target_temps
			#self.target_solar #cluster based on solar is merda!
			]
		return np.concatenate(x_list)

class  ModelTrend():
	def __init__(self, trend_series, temp_series, solar_series, holiday_series, lockdown_series,  n = 24, M = 100, rest = True):
		#Parameters
		self.n = n #cycle length
		self.M = M #Restrict to M neighbors
		self.rest = rest 
		
		#Base Data
		self.trend_series = trend_series
		self.temp_series = temp_series
		self.solar_series = solar_series
		self.holiday_series = holiday_series
		self.lockdown_series = lockdown_series

		#Feature importances container
		self.feat_imp = [] #empty list
		#Patterns
		self.patterns = self.generate_patterns(self.trend_series, self.temp_series, self.solar_series, self.n)


	def generate_patterns(self, trend_series, temp_series, solar_series, n):
		'''
		series: pd Series
			TS to be encoded
		n: int
			pattern length (default : 24)
		Returns patterns in linked list
		'''
		pattern_linked_list = PatternList() #generate linked list
		res_list = np.split(trend_series, len(trend_series)/n)
		temp_list = np.split(temp_series, len(temp_series)/n)
		solar_list = np.split(solar_series, len(solar_series)/n)
		for res, temp, solar in zip(res_list, temp_list, solar_list):
			pattern_linked_list.append((res, temp, solar))

		return pattern_linked_list

	def restrict_trainset(self, predictors, targets, query, n_neighbors):
		'''NN restriction of model training set.
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
		predictors = np.array([P.to_array() for P in predictors])
		targets = np.array(targets)
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
			y_val = trav.load.values[t] #HERE
			#Store values
			X.append(x_val)
			y.append(y_val)

			trav = trav.next
		#Perform trainset restriction
		if restrict:
			X, y = self.restrict_trainset(X, y, query, self.M)
		else:
			X = [P.to_array() for P in X]
		return X, y

	def hourly_trend_pred(self, query, t):
		'''
		query: pattern
		predict following day load at time t
		t: int
		hour of day
		'''
		#Local RF model to predict trend
		
		model = GradientBoostingRegressor(n_estimators = 700, #number of boosting iterations to perform
											loss = 'ls',
											learning_rate = 0.05, #shrinkage
											min_samples_split = 2,
											min_samples_leaf = 5, #default
											max_depth = None,
											subsample = 1, #stochastic gradient boosting
											max_features = 'auto',
											verbose = 1,
											ccp_alpha = 0.0,
											n_iter_no_change = 50,#Early stopping 1
											validation_fraction = 0.1, #Early stopping 2
											tol = 1) #Early stopping 3 [MWh]
		
		#Learrning sets
		X, y = self.learning_set(query, t, restrict = self.rest)
		#Model
		model = model.fit(X, y)
		#Compute predictor
		predictor = Predictor(self, query, t).to_array().reshape(1,-1)
		#Predict
		prediction = model.predict(predictor)
		feature_importances_ = model.feature_importances_
		self.feat_imp.append(feature_importances_) #Store feature importances array
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
		predictions = [] #list to store daily predictions
		trend_pred_series = pd.Series() #Empty series to store trend predictions
		trav = self.patterns.head
		k = 1 #Prediction day counter
		#traverse liste up to first test date
		while trav.date != first_date:
			trav = trav.next
		#Run Predictions
		while trav and trav.date != last_date + pd.Timedelta(1, "D"):
			daily_prediction = [] #store daily prediction
			print("Predicting day ", k)
			for t in range(24):
				t_pred = self.hourly_trend_pred(trav,t)
				daily_prediction.append(t_pred)
			daily_prediction = pd.Series(daily_prediction, index=trav.load.index)
			if recursive:
				trav.load = daily_prediction
			trend_pred_series = pd.concat([trend_pred_series, daily_prediction])
			trav = trav.next
			k +=1

		return trend_pred_series
