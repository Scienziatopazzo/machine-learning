import math
import random
import operator

class Perceptron:
	'''
	Implements the Perceptron Learning Algorithm
	fields:
	int dim				Dimensionality of the data
	List weights		Array (dim x 1) of the weights
	List data			Array (N x 1) of tuples (x, y) composed of vectors x and results y=f(x)
	int iterations		Number of iterations of PLA undergone
	'''
	def __init__(self, dim, data):
		self.dim = dim
		self.reset(data)

	def reset(self, data):
		'''
		Reset weights and iterations and feed a data sample
		'''
		self.weights = [0.0] * (self.dim+1)
		for t in data:
			if len(t[0])!=self.dim:
				raise ValueError('Wrong data dimensionality')
			elif t[1]!=1 and t[1]!=-1:
				raise ValueError('Function output is not binary')
		self.data = data
		self.iterations = 0

	def hypothesis(self, x):
		'''
		Takes d-dimensional data vector x and computes h(x)
		using the current weights
		'''
		x_adj = [1.0] + x	#adjusted to include 1 at the start
		weighted_sum = sum(map(operator.mul, self.weights, x_adj))	#dot product of w and x
		if weighted_sum==0.0:
			return 0.0
		else:
			return math.copysign(1.0, weighted_sum)		#sign function

	def classify(self, point):
		'''
		Takes as "point" a tuple (x, y) with x a vector and y=f(x)
		and classifies it, returning True if h(x)=f(x) and False if not
		'''
		h = self.hypothesis(point[0])
		return h == point[1]

	def train(self):
		'''
		Trains the perceptron with the data using the PLA
		'''
		misclass = True
		#iterate until there is no more misclassification
		while(misclass):
			#obtain a set of misclassified points
			misclass_points = []	#array of indexes of misclassified points in data
			for point in data:
				if not self.classify(point):
					misclass_points.append(data.index(point))

			if len(misclass_points)!=0:
				#choose the misclassified point at random
				p = data[random.choice(misclass_points)]
				x_adj = [1.0] + p[0]
				# w <- w + yx	where (x,y) is a misclassified point
				x_sign = [p[1]*xi for xi in x_adj]
				self.weights = [self.weights[i] + x_sign[i] for i in range(len(x_sign))]
				#increment number of iterations
				self.iterations += 1
			else:
				misclass=False

	def f_disagreement(self, new_data):
		'''
		When given a sufficiently big new dataset new_data with the same format of self.data,
		returns the disagreement fraction between the trained function g and the original f
		P[f(x) != g(x)]
		'''
		g_misclass_points = 0	#counter of newdata points misclassified by g
		for point in new_data:
			if not self.classify(point):
				g_misclass_points += 1
		#return the fraction of P
		return g_misclass_points / len(new_data)