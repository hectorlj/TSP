#!/usr/bin/python3

from which_pyqt import PYQT_VER
if PYQT_VER == 'PYQT5':
	from PyQt5.QtCore import QLineF, QPointF
elif PYQT_VER == 'PYQT4':
	from PyQt4.QtCore import QLineF, QPointF
else:
	raise Exception('Unsupported Version of PyQt: {}'.format(PYQT_VER))




import time
import numpy as np
from TSPClasses import *
import heapq
import itertools



class TSPSolver:
	def __init__( self, gui_view ):
		self._scenario = None
		self.best_fitness = None
		self.curr_fitness = None
		self.fitness_list = None
	def setupWithScenario( self, scenario ):
		self._scenario = scenario


	''' <summary>
		This is the entry point for the default solver
		which just finds a valid random tour.  Note this could be used to find your
		initial BSSF.
		</summary>
		<returns>results dictionary for GUI that contains three ints: cost of solution, 
		time spent to find solution, number of permutations tried during search, the 
		solution found, and three null values for fields not used for this 
		algorithm</returns> 
	'''
	
	def defaultRandomTour( self, time_allowance=60.0 ):
		results = {}
		cities = self._scenario.getCities()
		ncities = len(cities)
		foundTour = False
		count = 0
		bssf = None
		start_time = time.time()
		while not foundTour and time.time()-start_time < time_allowance:
			# create a random permutation
			perm = np.random.permutation( ncities )
			route = []
			# Now build the route using the random permutation
			for i in range( ncities ):
				route.append( cities[ perm[i] ] )
			bssf = TSPSolution(route)
			count += 1
			if bssf.cost < np.inf:
				# Found a valid route
				foundTour = True
		end_time = time.time()
		results['cost'] = bssf.cost if foundTour else math.inf
		results['time'] = end_time - start_time
		results['count'] = count
		results['soln'] = bssf
		results['max'] = None
		results['total'] = None
		results['pruned'] = None
		return results


	''' <summary>
		This is the entry point for the greedy solver, which you must implement for 
		the group project (but it is probably a good idea to just do it for the branch-and
		bound project as a way to get your feet wet).  Note this could be used to find your
		initial BSSF.
		</summary>
		<returns>results dictionary for GUI that contains three ints: cost of best solution, 
		time spent to find best solution, total number of solutions found, the best
		solution found, and three null values for fields not used for this 
		algorithm</returns> 
	'''

	def greedy( self,time_allowance=60.0 ):
		results = {}
		start_time = time.time()
		cities = self._scenario.getCities()
		numCities = len(cities)
		tourFound = False
		count = 0
		# print('start greedy')
		while not tourFound:
			p = np.random.permutation(numCities)
			route = []
			for i in range(numCities):
				route.append(cities[p[i]])
			bssf = TSPSolution(route)
			count += 1

			if bssf._costOfRoute() < np.inf:
				tourFound = True
		# print('finishing')
		# end_time = time.time()
		results['cost'] = bssf._costOfRoute()
		results['time'] = time.time() - start_time
		results['count'] = count
		results['soln'] = bssf
		
		return results



	
	
	''' <summary>
		This is the entry point for the branch-and-bound algorithm that you will implement
		</summary>
		<returns>results dictionary for GUI that contains three ints: cost of best solution, 
		time spent to find best solution, total number solutions found during search (does
		not include the initial BSSF), the best solution found, and three more ints: 
		max queue size, total number of states created, and number of pruned states.</returns> 
	'''
		
	def branchAndBound( self, time_allowance=60.0 ):
		results = {}
		start_time = time.time()
		cities = self._scenario.getCities()
		numCities = len(cities)
		tourFound = False
		count = 0
		# the default bssf is generated in O(n^2) time, n = number of cities
		# the bssf is an array with space complexity of O(n)
		temp = self.defaultRandomTour(time_allowance=60.0)
		bssf = temp['soln']
		lbBssf = temp['cost']
		# Priority queue operations are done in O(log p) time complexity and O(p) space complexity
		# with p being the number of items in the queue
		pQ = []
		heapq.heappush(pQ, self.buildInitialNode(cities, numCities, 0))
		numSG = 1
		numSP = 0
		maxQS = 1
		numbssf = 0

		# this has a time complexity of O(b^n) where n is the number of cities
		# and b is the number of reachable cities left to visit

		while len(pQ) > 0:
			# print('priority queue')
			# print(pQ)
			# print('size')
			# print(len(pQ))
			# print('')
			if maxQS < len(pQ):
				maxQS  = len(pQ)
			# pop node from queue takes O(logp) time and space
			curr = heapq.heappop(pQ)
			if curr.dct['LB'] < lbBssf:
				# updates BSSF in O(n) time and space
				if len(curr.dct['Route']) == numCities:
					count +=1
					if curr.dct['LB'] < lbBssf:
						lbBssf = curr.dct['LB']
						route = []
						for i in range(len(curr.dct['Route'])):
							route.append(cities[curr.dct['Route'][i]])

						tempbssf = TSPSolution(route)
						if tempbssf.cost < bssf.cost:
							bssf = tempbssf
							numbssf += 1
				else:
					if time.time() - start_time > time_allowance:
						break
					# create a new node for the reachable cities of the current node,
					# this is done for every node left in the priority queue
					# worst case is O(b^n) time and space since reachable states with a
					# higher BSSF are pruned
					for i in range(numCities):
						if i not in curr.dct['Route']:
							newNode = self.buildNode(np.copy(curr.dct['Array']), 
								cities, numCities, curr.dct['Index'], i, list(curr.dct['Route']), curr.dct['LB'])
							numSG += 1
							# selectively adding only promising child nodes to the queue
							if newNode.dct['LB'] < lbBssf:
								heapq.heappush(pQ, newNode)
							else:
								numSP += 1
			else:
				numSP += 1
		numSP += len(pQ)
		# data returned in O(1) time and space
		results['cost'] = lbBssf
		results['time'] = time.time() - start_time
		results['count'] = count
		results['soln'] = bssf
		results['max'] = repr(maxQS)
		results['total'] = repr(numSG)
		results['pruned'] = repr(numSP)
		
		return results



	''' <summary>
		This is the entry point for the algorithm you'll write for your group project.
		</summary>
		<returns>results dictionary for GUI that contains three ints: cost of best solution, 
		time spent to find best solution, total number of solutions found during search, the 
		best solution found.  You may use the other three field however you like.
		algorithm</returns> 
	'''
	def greed_fancy(self, cities, numCities, best_cost, time_allowance=60.0):
		results = {}
		curr_city = random.choice(cities)
		route = [curr_city]
		free_cities = set(cities)
		free_cities.remove(curr_city)
		# print('start greedy')
		while free_cities:
			next_city = min(free_cities, key=lambda x: curr_city.costTo(x))
			free_cities.remove(next_city)
			route.append(next_city)
			curr_city = next_city

		bssf = TSPSolution(route)
		if bssf._costOfRoute() < best_cost:
			self.best_fitness = bssf
		self.fitness_list.append(bssf)
		# print('finishing')
		# end_time = time.time()
		return bssf


	def fancy( self,time_allowance=60.0 ):
		results = {}
		start_time = time.time()
		count = 0
		cities = self._scenario.getCities()
		numCities = len(cities)
		self.best_fitness = np.inf
		self.fitness_list = []
		self.best_fitness = self.greed_fancy(cities, numCities, np.inf)
		self.curr_fitness = self.best_fitness
		self.simAnnealing(cities, numCities, start_time, count)
		curr_bssf = self.best_fitness
		# print(self.fitness_list)
		results['cost'] = curr_bssf._costOfRoute()
		results['time'] = time.time() - start_time
		results['max'] = None
		results['count'] = count
		results['total'] = len(self.fitness_list)
		results['pruned'] = None
		results['soln'] = curr_bssf
		
		return results

	def simAnnealing(self, cities, numCities, start_time, count, time_allowance=60.0):
		alpha = 0.995
		stoppingTemp = 1e-10
		temperature = math.sqrt(numCities)
		
		while temperature >= stoppingTemp and time.time() - start_time < time_allowance:
			temp = list(self.curr_fitness.route)
			l = random.randint(2, numCities - 1)
			i = random.randint(0, numCities - l)
			temp[i: (i + l)] = reversed(temp[i : (i + l)])
			temp_bssf = TSPSolution(temp)
			self.accept(temp_bssf, temperature, count)
			temperature *= alpha
			self.fitness_list.append(temp_bssf)
		

	def accept(self, temp, temperature, count):
		if temp._costOfRoute() < self.curr_fitness._costOfRoute():
			self.curr_fitness = temp
			if self.curr_fitness._costOfRoute() < self.best_fitness._costOfRoute():
				self.best_fitness = temp
				count += 1
		else:
			if random.random() < self.probability_accept(temp, temperature):
				self.curr_fitness = temp

	def probability_accept(self, temp, temperature):
		return math.exp(-abs(temp._costOfRoute() - self.curr_fitness._costOfRoute()) / temperature)

	# runs in O(n^2) and space O(n)
	def rowRedux(self, costArray, numCities):
		# array of smallest in each row

		temp = costArray.min(1)
		# print(costArray)
		# print (temp)
		# print('before')
		# print(costArray)
		# print('temp')
		# print (temp)
		for i in range(numCities):
			for j in range(numCities):
				if np.isinf(temp[i]):
					temp[i] = 0
				if not np.isinf(costArray[i, j]):
					costArray[i, j] -= temp[i]
		# print('after')
		# print(costArray)
		return temp.sum()

	# runs in O(n^2) and space O(n)
	def colRedux(self, costArray, numCities):
		temp = costArray.min(0)
		for j in range(numCities):
			for i in range(numCities):
				if np.isinf(temp[i]):
					temp[j] = 0
				if not np.isinf(costArray[i, j]):
					costArray[i, j] -= temp[j]
		return temp.sum()

	# runs in O(n^2) space and time
	def buildInitialNode(self, cities, numCities, curr):
		costArray = np.zeros(shape=(numCities, numCities))
		for i in range(numCities):
			for j in range(numCities):
				cost = cities[i].costTo(cities[j])
				if costArray[i, j] != np.inf:
					costArray[i, j] = cost
		xMin = costArray.min(axis=1)
		yMin = costArray.min(axis=0)
		LB = self.rowRedux(costArray, numCities)
		LB += self.colRedux(costArray, numCities)
		# print(LB)
		return KeyDict(LB, {"Node": cities[curr], "LB": LB, "Array": costArray, "Route": [curr], "Index": curr})
	# runs in O(n^2) space and time
	def buildNode(self, costArray, cities, numCities, prev, curr, route, LB):
		LB += costArray[prev, curr]
		costArray[prev, curr] = np.inf
		for i in range(numCities):
			costArray[prev, i] = np.inf
		for j in range(numCities):
			costArray[j, curr] = np.inf
		
		LB += self.rowRedux(costArray, numCities)
		LB += self.colRedux(costArray, numCities)
		route.append(curr)
		return KeyDict(LB - 400 * len(route), {"Node": cities[curr], "LB": LB, "Array": costArray, "Route": route, "Index": curr})


class KeyDict(object):
	def __init__(self, key, dct):
		self.key = key
		self.dct = dct

	def __lt__(self, other):
		return self.key < other.key

	def __eq__(self, other):
		return self.key == other.key

	def __repr__(self):
		return '{0.__class__.__name__}(key={0.key}, dct={0.dct})'.format(self)