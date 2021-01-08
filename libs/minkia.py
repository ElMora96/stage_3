class  Predictor():
	'''Predictors for Random Forest Model, for better working of NN'''
	def __init__(self, model, query, t):
		'''query: Pattern. -- Target to predict
		   model: Parent forecasting model
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
		self.target_solar = query.solar.values
		#Metadata
		self.target_month = np.array(query.month).reshape(1) #label encoding
		self.target_weekday = np.array(query.weekday).reshape(1) #label encoding
		self.target_hour = np.array(t).reshape(1) #label encoding
		self.target_period = query.target_period
		self.target_lockdown = np.array(model.lockdown_series[query.load_resid.index[t]]).reshape(1)
		self.target_holiday = np.array(model.holiday_series[query.load_resid.index[t]]).reshape(1)

	def to_array(self):
		'''Return predictor in np.array format'''
		x_list = [self.load,
			self.target_temps,
			self.target_solar,
			self.target_period,
			self.target_hour,
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


#Restrict Predictors to NN -- To be inserted into ModelRF
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
	q_predictor = Predictor(self, query)
	#Sklearn euclidean Neighbors
	base_model = NearestNeighbors(n_neighbors, metric = "euclidean")
	X = [P.to_reduced_array() for P in predictors]
	X = np.array(X)
	#Basic Neighbors
	base_model.fit(X)
	q_x = q_predictor.to_reduced_array().reshape(1,-1) #Query array
	neigh = list(Neighbors.kneighbors(q_x, return_distance = False))
	#First filtering
	predictors = [predictors[i] for i in neigh]
	targets = [targets[i] for i in neigh]
	#Holiday filtering
	if q_predictor.target_holiday > 0:
		neigh_2 = []
		for ix, item in enumerate(predictors)
			if item.target_holiday > 0:
				neigh_2.append(ix) #retain maledetti
		predictors = [predictors[i] for i in neigh_2]
		targets = [targets[i] for i in neigh_2]

	#...
	#Return trainset as array
	predictors = np.array([P.to_array for P in predictors])
	targets = np.array(targets)
	return predictors, targets
			








