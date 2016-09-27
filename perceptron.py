import math

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
		x_adj = np.insert(x, 0, 1.)	#adjusted to include 1 at the start
		weighted_sum = self.weights.dot(x_adj)
		if weighted_sum==0:
			return 0.0
		else:
			return math.copysign(1.0, weighted_sum)

	def classify(self, point):
		'''
		Takes as "point" a tuple (x, y) with x a vector and y=f(x)
		and classifies it, returning True if h(x)=f(x) and False if not
		'''
		h = self.hypothesis(point[0])
		return h == y

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



#Test
data = [([0.1, 0.2], 1),
		([0.2, 0.02], 1),
		([-0.77, -1.0], -1)]
p = Perceptron(2, data)
