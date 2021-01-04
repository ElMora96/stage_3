#Artificial Immune System - Local Feature Selection
#Libraries
import pandas as pd
import numpy as np
import copy as cp 
import  warnings as wr
from numba import jit
#Mean absolute percentage error
from utils_lib import mape, all_equal
#Minkowski distance
from scipy.spatial.distance import minkowski


class AntiGen():  #Training Antigens
	def __init__(self, input_data, output_data):
		#Load values
		self.input_data = input_data
		self.output_data = output_data
		#Encoding-Decoding parameters
		self.mean = np.mean(input_data)
		self.sd = np.std(input_data)
		#Standardized values
		self.x = self.standardize(input_data)
		self.y = self.standardize(output_data)
		#Metadata
		self.output_day = output_data.index[0].weekday() #pd weekday

	def standardize(self, vec):
		#Standardization (encoding)
		std_vec = (vec - self.mean)/self.sd
		return std_vec

	def rescale(self, vec):
		#Invert standardization (decoding)		
		rescaled_vec = vec*self.sd + self.mean
		return rescaled_vec


class ForecastAntiGen(AntiGen): #Forecast (or Test) Antigens - No output data at init
	def __init__(self, input_data, tau = 1):
		#Load values 
		self.input_data = input_data
		self.output_data = None #No forecast for now
		#Encoding-Decoding parameters
		self.mean = np.mean(input_data)
		self.sd = np.std(input_data)
		#Standardized values
		self.x = self.standardize(input_data)
		self.y = None #No y-pattern for now
		#Metadata
		self.tau = tau
		self.input_day = input_data.index[0].dayofweek
		self.output_day = (self.input_day + tau) % 7 #Pandas format

	def set_label(self, label):
		#Set forecasted values and y-pattern
		self.y = label
		if label is not None:
			self.output_data = pd.Series(self.rescale(label), index = self.input_data.index + pd.Timedelta(self.tau, 'D'))
		else:
			self.output_data = label #if label is None
			wr.warn("Unable to compute forecast.")


class AntiBody(): #Virtual antibody
	def __init__(self, p, q):
		'''Parameters:
		P: standardized input vector
		q: standardized output vector (label)
		'''
		#Standardized load values
		self.p = p
		self.q = q
		#Parameters
		self.n = len(p)
		#Paratope 
		self.omega = self.initialize_paratope()
		#Cross-reactivity threshold
		self.r = 0
		#Power of antibody
		self.P = 0
		#Pointer to initializing antigen
		self.init_antigen = None
		#Pointer to parent (to identify clones)
		self.parent = None

	def initialize_paratope(self):
		#Random inizialization 
		paratope = np.random.randint(low = 0, high = 2, size = self.n)
		return paratope


class AIS():
	def __init__(self, data, tau = 1, Z = 10, S = 3 , delta = 2, c = 0.5 , sigma = 2):
		'''Parameters.
		data: pd.Series - Load TS under consideration
		tau: int -- Forecast horizon
		Z: int -- Clone population size
		S: int -- Stop clonal selection loop after s iterations w/o improvements
		delta: float >=0 -- Threshold error
		c: float in [0,1) -- Adjust cross-reactivity threshold
		sigma: float >=0 -- St. deviation of Normal distibution controlling hypermutation range 
		'''
		#Data
		self.data = data #Pretend for now day separation has already been performed
		#Parameters
		self.tau = tau
		self.Z = Z
		self.S = int(S)
		self.delta = delta
		self.c = c
		self.sigma = sigma
		#Cycle length
		self.n = 24
		#Population size
		self.N = int(len(data)/self.n)#daily patterns
		#Target weekday of model
		self.target_day = None
		#Training antigen population - generated for each forecasting task
		self.PopAG = None
		#Antibody population
		self.PopAB = None

	def load_antigens(self, target_day):
		'''Load training set in the form of AGs.
		Parameters.
		target_day: build day-specific AG training set
		'''
		population = []
		daily_cycles = np.split(self.data, self.N)
		for j in range(self.N):
			cycle = daily_cycles[j]
			if target_day == cycle.index[0].weekday():
				try:
					input_day = daily_cycles[j - self.tau]
					AG = AntiGen(input_day, cycle) #Load antigen
					population.append(AG)
				except:
					continue
		return population

	def generate_antibodies(self):
		'''Initialize AB population w/ AG data'''
		population = []
		for AG in self.PopAG:
			AB = AntiBody(AG.x, AG.y)
			AB.init_antigen = AG #set initialization antigen
			population.append(AB)
		return population

	#@jit(nopython = True)
	def paratope_dist(self, AB, AG, e = 2):
		'''Compute distance between specified AB and AG,
		taking into account only position specified in the paratope of AB.
		Parameters:
		e: int, 1 or 2 -- Specify Manhattan or euclidean distance
		'''
		paratope = AB.omega
		p = np.multiply(AB.p, paratope)
		x = np.multiply(AG.x, paratope)
		dist = minkowski(u = p,v = x, p = e)
		return dist

	def generate_clones(self, AB):
		'''Generate clones of given antibody AB'''
		clones = []
		for l in range(self.Z):
			clone = cp.deepcopy(AB) #create deep clone
			clone.parent = AB #Pointer to parent clone for later usage
			clones.append(clone)
		return clones

	#@jit(nopython = True)
	def hypermutaion(self, clone_AB):
		'''Run SHM over clone AB'''
		#Compute m:
		m = np.random.normal(loc = 0.0, scale = self.sigma)
		m = np.ceil(np.abs(m))
		if m == 0:
			m = 1
		elif m > self.n:
			m = m - np.floor((m-1)/self.n) * self.n

		#Cast as integer
		m = int(m)
		#Swap m random bits of AB paratope
		old_paratope = clone_AB.omega
		inversion_indices = np.random.choice(self.n, replace = False, size = m)
		new_paratope = old_paratope
		for j in range(self.n):
			if j in inversion_indices:
				new_paratope[j] = new_paratope[j]^1 #bitwise xor
		#Set new paratope
		clone_AB.omega = new_paratope

	#@jit(nopython = True)
	def cross_reactivity_threshold(self, clone_AB):
		'''compute cross-reactivity threshold r of clone AB, keep trace of parent'''
		#Split antigens into two classes
		class_1 = []
		class_2 = []
		true_load_index = self.PopAB.index(clone_AB.parent) #Hope this fuckin works!
		true_load = self.PopAG[true_load_index].output_data #(Compare with real load)
		'''
		#Alternative implementation using pointer to initializing AG
		true_load = clone_AB.parent.init_antigen.output_data
		'''
		ave = self.PopAG[true_load_index].mean
		sd = self.PopAG[true_load_index].sd
		for AG in self.PopAG: #Only train AGs
			pred_load = AG.y * sd + ave #decode 
			error = mape(true_load, pred_load) 
			if error <= self.delta:
				class_1.append(AG)
			else:
				class_2.append(AG)
		#Compute cross-reactivity threshold
		#Find smallest distance between clone_AB and AGs in class 2 (d_A)
		distances_A = []
		for AG in class_2:
			d = self.paratope_dist(clone_AB, AG)
			distances_A.append(d)
		d_A = min(distances_A) #extract minimum
		#Find largest distance between clone_AB and AGs in class 1 (d_B)
		distances_B = []
		for AG in class_1:
			d = self.paratope_dist(clone_AB, AG)
			distances_B.append(d)
		d_B = max(distances_B) #extract maximum
		#Threshold
		r_clone = d_A + self.c * (d_B - d_A)
		clone_AB.r = r_clone

	def affinity(self, AB, AG):
		'''affinity of the AB for the AG'''
		if self.paratope_dist(AB, AG) >= AB.r or AB.r == 0:
			aff = 0
		else:
			aff = 1 - (self.paratope_dist(AB, AG)/AB.r)
		return aff

	#@jit(nopython = True)
	def label_and_power(self, clone_AB):
		'''compute prediction label for given clone antibody, together with its power'''
		#Find AGs which lie in recognition region of clone_AB
		affinities = [] #affinities of recognized AGs
		y_list = [] #y-patterns of recognized AGs
		for AG in self.PopAG:
			if self.paratope_dist(clone_AB, AG) < clone_AB.r:
				y_list.append(AG.y)
				aff = self.affinity(clone_AB, AG)
				affinities.append(aff)
		#AB label - weighted average with affinities
		try:
			label = np.average(a = y_list, axis = 0, weights = affinities)
		except: #Might be the case that weights (affinities) are all zero
			label = clone_AB.q
		#AB power - cardinality
		power = len(y_list)
		#Set values
		clone_AB.q = label
		clone_AB.P = power

	#@jit(nopython = True)
	def winner_clone(self, clones):
		'''Find best clone'''
		#First selection - Highest power
		powers = [AB.P for AB in clones]
		best_power = np.argwhere(powers == np.max(powers))
		best_power = best_power.flatten().tolist() #cast as list
		if len(best_power) == 1: #if single, winner found
			best_power_index = best_power[0]
			winner = clones[best_power_index]
			#...
		else:
			#Second selection - Samllest paratope size
			paratope_sizes = []
			for index in best_power:
				size = clones[index].omega.sum() #all ones - sum to get size
				paratope_sizes.append(size)
			best_size = np.argwhere(paratope_sizes == np.max(paratope_sizes))
			best_size = best_size.flatten().tolist()
			if len(best_size) == 1:
				best_size_index = best_power[best_size[0]]
				winner = clones[best_size_index]
				#...
			#Third selection - Random 
			else:
				rand = np.random.randint(len(best_size))
				best_index = best_power[best_size[rand]]
				winner = clones[best_index]
				#...
		return winner

	def run_training_stop_criterion(self):
		#Loop over antibodies
		n_antibodies = len(self.PopAB)
		i = 1
		for AB in self.PopAB:
			#Status message
			print("Processing AntiBody ", i, " of ", n_antibodies) 
			#Power list for stopping criterions
			pow_list = []
			#Store parent index in list
			index = self.PopAB.index(AB)
			best = AB #temporary best
			for j in range(self.S): #Iterate S times
				#Generate pool of clones
				clone_pool = self.generate_clones(best)
				#Loop over clones
				for clone in clone_pool:
					#Hypermutation of clones
					self.hypermutaion(clone)
					#Compute cross-reactivity threshold for clone
					self.cross_reactivity_threshold(clone)
					#Compute affinities, label and power
					self.label_and_power(clone)
				#Find best clone
				best = self.winner_clone(clone_pool)
				#Replace AB with best clone
				self.PopAB[index] = best
				pow_list.append(best.P)
			#Stop iterating when criterion is satisfied or when number iterations exceeds 10*S
			while not(all_equal(pow_list[-self.S:])):
				#Generate pool of clones
				clone_pool = self.generate_clones(best)
				#Loop over clones
				for clone in clone_pool:
					#Hypermutation of clones
					self.hypermutaion(clone)
					#Compute cross-reactivity threshold for clone
					self.cross_reactivity_threshold(clone)
					#Compute affinities, label and power
					self.label_and_power(clone)
				#Find best clone
				best = self.winner_clone(clone_pool)
				#Replace AB with best clone
				self.PopAB[index] = best
				pow_list.append(best.P)	
				if len(pow_list) >= 25:
					break #Ugly but just fuckin works			
			print("Clonal power-up recognized antigens ", pow_list) #debug
			i += 1

	def train(self, target_day):
		'''Train model for given day of week target
		target_day: int in {0,1, ... , 6}

		'''
		#Set model target day
		self.target_day = target_day
		#Load antigen training set according to target day
		self.PopAG = self.load_antigens(target_day)
		#Generate antibodies
		self.PopAB = self.generate_antibodies()
		#Train
		self.run_training_stop_criterion()

	def predict(self, fAG, train = False):
		'''Compute label (forecast) for test AG.
		Parameters.
		fAG: ForecastAntiGen. 
		'''
		if train:
			#Set model target day
			self.target_day = fAG.output_day
			#Load antigen training set according to target day
			self.PopAG = self.load_antigens(self.target_day)
			#Generate antibodies
			self.PopAB = self.generate_antibodies()
			#Train
			self.run_training_stop_criterion()
		#Check if Forecast AG is suitable
		elif fAG.output_day != self.target_day:
			raise ValueError("AG target day does not match model target day")
		#Forecast routine
		weights = [] #weights for forecast
		vectors = [] #AB labels to average
		for AB in self.PopAB: #Trained AB population
			aff = self.affinity(AB, fAG) #compute affinity
			if aff > 0: 
				weight = AB.P * aff
				vectors.append(AB.q)
				weights.append(weight)
		if len(vectors)>0:
			forecast = np.average(a = vectors, axis = 0, weights = weights) #compute weighted average
		else:
			wr.warn("Antigen Not recognized by system!")		
			forecast = None 
		#Set forecast antigen label
		print("F:", forecast)
		fAG.set_label(forecast)
		return forecast

class ParallelAIS():
	'''Class to simultaneoulsy build parallel AIS: one for each weekday'''
	def __init__(self, data, train = True, **kwargs):
		self.data = data
		self.params = kwargs
		self.models = [] #Store Artificial Immune Systems
		#Build & Train separate models
		dayOfWeek={0:'Monday', 1:'Tuesday', 2:'Wednesday', 3:'Thursday', 4:'Friday', 5:'Saturday', 6:'Sunday'}
		for day in range(7):
			ais = AIS(data, **kwargs)
			if train:
				print("Training AIS for ", dayOfWeek[day])
				ais.train(day)
			self.models.append(ais) #Store ais (eventally trained)

	def predict(self, fAG):
		'''Compute label (forecast) for test AG.
		Parameters.
		fAG: ForecastAntiGen. 
		'''
		target_day = fAG.output_day
		self.models[target_day].predict(fAG)



