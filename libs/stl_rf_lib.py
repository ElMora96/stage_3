from numpy import sin, cos, pi, array, split, empty, concatenate
from pandas import Timedelta, Series, to_datetime, date_range
from statsmodels.tsa.seasonal import STL
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import NearestNeighbors

class ForecastModel:
	"""Model to produce short-term load forecasts"""
	#------------------------Nested Pattern class---------------------
	class _Pattern:
		"""Pattern representation of load time series under consideration.
		Consider load cycles of given length"""
		def __init__(self, load, temp, solar, holiday, lockdown):
			"""Constructor should not be invoked by user.
			load, temp and solar are pd.Series with datetime index"""
			#Data
			self._load = load.values #Electric load
			self._temp = temp.values #Forecasted temperatures
			self._solar = solar.values #Forecast solar production
			self._holiday = holiday.values #Holiday score
			self._lockdown = lockdown.values #Lockdown score
			#Metadata
			self._date = load.index[0].date() #Date
			self._dayofweek = load.index[0].dayofweek #Day of week
			self._dayofyear = load.index[0].dayofyear #Day of year - used to compute period vector
			self._month = load.index[0].month #Month number
			self._period = self._compute_period() #2-D array representation of period of year
			#Predictor array
			self._predictor = None #Computed only on call
			#Links
			self._prev = None
			self._next = None

		def _compute_period(self):
			"""Encode period of year"""
			p_1 = sin(2*pi*self._dayofyear*(1/366.0))
			p_2 = cos(2*pi*self._dayofyear*(1/366.0))
			return array([p_1, p_2])

		def _compute_predictor(self):
			"""Compute predictor for hours of day in present pattern"""
			x_list = [self._prev._load, #previous day load
					self._temp, #forecast temperatures
					self._solar[6:21], #forecast solar
					self._period, #target period
					array(self._dayofweek).reshape(1), #target weekday
					array(self._month).reshape(1), #target month
					array(self._lockdown[12]).reshape(1), #lockdown score
					array(self._holiday[12]).reshape(1) #holiday score
					]
			self._predictor = concatenate(x_list)	#total size: 70

	#----------------Nested Doubly linked list to store patterns-------
	class _PatternList:
		"""Minimal Doubly LL to store patterns in a convenient way"""
		def __init__(self):
			"""Create an empty list. Constructor should not be invoked by user."""
			self._head = None
			self._tail = None
			self._size = 0

		def __len__(self):
			"""Return number of elements in list"""
			return self._size

		def is_empty(self):
			"""Return True if list is empty"""
			return len(self) == 0

		def append_after_tail(self, node):
			"""append new node after tail"""
			if self.is_empty():
				self._head = node #set new head
			else:
				self._tail._next = node
				node._prev = self._tail
				node._compute_predictor() #Compute predictor
			self._tail = node #set new tail
			self._size += 1 #Increase list size

	#---------------------------Model Methods - Public---------------------------
	def __init__(self,
				load_series,
				temp_series,
				solar_series,
				holiday_series,
				lockdown_series,
				cycle_length = 24,
				restrict_to_nn = True,
				n_neighbors = 80
				):
		"""Initialize forecasting model.
		restrict_to_nn : Boolean --- whether or not to perform NN trainset restriction
		n_neighbors: int --- Number of neighbors to consider. Used only if restrict_to_nn = True.
		"""
		#Parameters
		self._cycle_length = cycle_length 
		self._restrict_to_nn = restrict_to_nn
		self._n_neighbors = n_neighbors
		#Data
		self._load_series = load_series
		self._temp_series = temp_series
		self._solar_series = solar_series
		self._holiday_series = holiday_series
		self._lockdown_series = lockdown_series
		#Seasonal Adjustment of load
		self._deseasonalized, self._daily_seasonal, self._weekly_seasonal  = self._decompose(load_series)
		#Generate pattern linked list of adjusted load
		self._patterns = self._generate_patterns()
		#Store forecasts
		self._full_forecast = None #full
		self._daily_seasonal_forecast = None
		self._weekly_seasonal_forecast = None
		self._deseasonalized_forecast = None

	def fit_predict(self, test_time_range):
		"""Generate & Fit submodels; generate forecast of test_time_range.
		test_time_range: daterange. Return full forecast."""
		#Date boundaries for test
		first_date = test_time_range[0].date()
		last_date = test_time_range[-1].date()
		n_days = (last_date - first_date).days + 1 #retrive days attribute from timedelta
		#Empty arrays to store forecast
		forecast_length = n_days * self._cycle_length #Length of forecast vectors
		daily_seasonal_forecast = empty(forecast_length)
		weekly_seasonal_forecast = empty(forecast_length)
		deseasonalized_forecast = empty(forecast_length)
		#Align linked list with beginning of test
		trav = self._patterns._tail 
		while trav._date != first_date:
			trav = trav._prev 
		#Forecast routine
		k = 0 #Forecast day counter
		while trav and trav._date != last_date + self._cycle_length * Timedelta(1, 'H'): #Traverse patterns day by day
			print("Forecasting day ", k+1, " of ", n_days) #Progress
			#deseasonalized component forecast
			deseasonalized_comp = empty(self._cycle_length) #Empty vector to store values
			for t in range(self._cycle_length):
				element = self._point_deseasonalized_forecast(trav, t)
				deseasonalized_forecast[k*self._cycle_length + t] = element	
			daily_seasonal_forecast[k * self._cycle_length: (k+1) * self._cycle_length] = self._season_forecast(trav, 24) #First seasonal	component
			weekly_seasonal_forecast[k * self._cycle_length: (k+1) * self._cycle_length] = self._season_forecast(trav, 168) #Second seasonalcomponent	
			trav = trav._next #Move to next day
			k = k + 1 #Increase forecast day counter
		#Store predicitions in model - convert to pandas Series
		self._full_forecast = Series(deseasonalized_forecast + weekly_seasonal_forecast + daily_seasonal_forecast, index=test_time_range)
		self._deseasonalized_forecast = Series(deseasonalized_forecast, index=test_time_range)
		self._daily_seasonal_forecast = Series(daily_seasonal_forecast, index=test_time_range)
		self._weekly_seasonal_forecast = Series(weekly_seasonal_forecast, index=test_time_range)

		return self._full_forecast

	#---------------------------Model Methods - Non-Public----------------------
	def _decompose(self, series):
		"""Utility to deseasonalize series under consideration twice:
			-weekly seasonality;
			-daily seasonality.
		"""
		first_decomp = STL(series, period = 24, seasonal = 25).fit()
		daily_seasonal = first_decomp.seasonal
		intermediate_series = series - daily_seasonal #Filter out daily seasonalty
		second_decomp = STL(intermediate_series, period = 168, seasonal = 169).fit()
		weekly_seasonal = second_decomp.seasonal
		deseasonalized = intermediate_series - weekly_seasonal
		return deseasonalized, daily_seasonal, weekly_seasonal

	def _generate_patterns(self):
		"""Routine to generate (adjusted) load pattern representation"""
		patterns = self._PatternList() #Create linked list
		pattern_split = lambda series : split(series, len(series)/self._cycle_length) #lambda to perform splitting
		load = pattern_split(self._deseasonalized)
		temp = pattern_split(self._temp_series)
		solar = pattern_split(self._solar_series)
		holiday = pattern_split(self._holiday_series)
		lockdown = pattern_split(self._lockdown_series)
		for ld, t, s, h, lk in zip(load, temp, solar, holiday, lockdown):
			new_node = self._Pattern(ld, t, s, h, lk) #Create new pattern
			patterns.append_after_tail(new_node) #Append to linked list
		return patterns 

	def _point_deseasonalized_forecast(self, query, t):
		"""Generate forecast of seasonally adjusted series. Take as input query pattern;
		generate load forecast for following day at hour t."""
		model = RandomForestRegressor(n_estimators = 128, #100
									   max_features = 'sqrt', #17
									   bootstrap = True,
									   max_samples = None, #low bootstrap size 0.3
									   max_depth =None, #5
									   min_samples_split = 2,
									   n_jobs = 4,
									   ccp_alpha = 0.0, #no cost-complexity pruning
									   verbose = 0)
		X, y = self._learning_set(query, t) #(local) learning set
		model.fit(X,y) #Fit local model
		x_query = query._predictor.reshape(1,-1) #Predictor
		yhat = model.predict(x_query) 
		return yhat

	def _learning_set(self, query, t):
		"""Generate learning set for _point_deseasonalized_forecast method"""
		X, y = [], [] #Lists to store values
		trav = self._patterns._head._next #Skip first day (it has no predictor)
		while trav._date != query._date:
			x_val = trav._predictor #Previously computed predictor
			y_val = trav._load[t] #Load at time t
			X.append(x_val)
			y.append(y_val)
			trav = trav._next
		X = array(X) #cast as array
		if self._restrict_to_nn: #Perform Nearest Neighbors trainset restriction
			X_neigh = X[:,0:24] #Consider only load 
			model = NearestNeighbors(self._n_neighbors, metric = 'euclidean')
			model.fit(X_neigh)
			x_query = query._predictor[0:24].reshape(1,-1)
			neighbors = model.kneighbors(x_query, return_distance = False).tolist()[0]
			X = [X[i] for i in neighbors]
			y = [y[i] for i in neighbors]
		return X,y

	def _season_forecast(self, query, seasonal_period, look_back = Timedelta(3,'W')):
		"""Produce a seasonal component forecast for a given pattern (24h forecast)"""
		time_0 = to_datetime(query._date, utc = True)
		train_time = date_range(start = time_0 - look_back, 
								end = time_0 - Timedelta(1,'H'), 
								freq = 'H',
								tz='UTC')
		#print(self._daily_seasonal.index)
		if seasonal_period == 24:
			data = self._daily_seasonal[train_time]
		elif seasonal_period == 168:
			data = self._weekly_seasonal[train_time]
		else:
			raise ValueError("Seasonal period must be 24 or 168")
		model = ExponentialSmoothing(data,
			trend = 'add',
			seasonal = 'add',
			seasonal_periods = seasonal_period,
			initialization_method = 'estimated',
			freq = 'H'
			)
		fit = model.fit()
		forecast = fit.predict(start = time_0, end = time_0 + Timedelta(23, 'H'))
		return forecast

		


