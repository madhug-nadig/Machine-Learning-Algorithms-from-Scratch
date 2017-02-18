import numpy as np

class K_Means:
	def __init__(self, k =3, tolerance = 0.0001, max_iterations = 500):
		self.k = k
		self.tolerance = tolerance
		self.max_iterations = max_iterations
	
	def fit(self, data):
		pass

	def pred(self, data):
		pass

X = np.array([[1,2],[1,1],[1.5,1.4],[4,4],[4,6],[5,5],[6,8], [3,1],[2,2],[9,9],[5,5],[1,0.5],[7,7]])

km = K_Means(2)
