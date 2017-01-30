#================================================================================================================
#----------------------------------------------------------------------------------------------------------------
#									SUPPORT VECTOR MACHINE
#----------------------------------------------------------------------------------------------------------------
#================================================================================================================

import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
import pandas
import datetime

#for plotting
plt.style.use('ggplot')

class CustomSVM:
	
	def __init__(self):
		pass
		
		
	#Use the data and find a 'model' ie the values for W and b. Maximize b and minimize b 
	def fit(self, dataset):
		self.dataset = dataset
		
		# Magnitude of W is the key, list of W and b is the value
		options = {}
		
		trans =  [[1,1],[-1,1],[-1,-1],[1 ,-1]]
		data = []
		
		for yi, attrs in self.dataset.items():
			for attr in attrs:
				for f in attr:
					print(f, attr)
					data.append(f)
		print(data)
		self.max_attr = max(data)
		self.min_attr = min(data)
		
		#Clear the data from memory
		data = None
		
		step_size = [self.max_attr * 0.1,self.max_attr * 0.01,self.max_attr * 0.005]
		
		b_range = 5
		b_multiple = 5
		
		latest_optimum = self.max_attr * 10
		
		for step in step_size:
			W = np.array([latest_optimum,latest_optimum])
			opti = False
			
			while not opti:
				for b in np.arange(-1*(self.max_attr* b_range ), self.max_attr * b_range, step * b_multiple):
					for transformation in trans:
						W_t = W * transformation
						found = True
						for yi, xi in self.dataset.items():
							print(b)
							if not (yi* np.dot(W_t , xi)+b ) >= 1:
								found = False
								break
						if found:
							options[np.linalg.norm(W_t)] = [W_t, b]
				if W[0]<0:
					opti = True
					print("Optimized by a step: ", step)
				else:
					W -= step
			
			norms = min([n for n in options])
			self.W = options[norms][0]
			self.b = options[norms][1]
			
			latest_optimum = options[norm][0][0] + step*2

	def predict(self, attrs):
		#sign of the X(i).W + b defines the class
		classification = np.sign(np.dot(np.array(attrs), self.W) + self.b)
		
		return classification
			
def main():

	dataset = { -1 : np.array([[2,3],[4,5],[2,1]]), 1: np.array([[5,6], [8,8], [9,9]]) }
	svm = CustomSVM()

	svm.fit(dataset = dataset)
	pred = svm.predict(attrs = [2,2])
	
if __name__ == "__main__":
	main()
