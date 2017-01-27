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
	

def main():
	svm = CustomSVM()

if __name__ == "__main__":
	main()
