#================================================================================================================
#----------------------------------------------------------------------------------------------------------------
#													LOGISTIC REGRESSION
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

#Using gradient decent here to arrive at the optimal.


class CustomLogisticRegression:
	
	def __init__(self, x, y, tolerence = 0.00001):
		self.tolerance = tolerance
		self.cost = []
		self.alpha = 0.1
		self.lambd = 0.25
		self.iter = 2500
		
		#initialie theta
		self.theta = np.random.rand(x.shape[1],1)
	
	def descent(self, x, y):
		pass

def main():

	#IN CASE THE INPUT IS TO BE TAKEN IN FROM THE COMMAND PROMPT

	#takes in input from the user
	#x = list(map(int, input("Enter x: \n").split()))
	#y = list(map(int, input("Enter y: \n").split()))

	#convert to an numpy array with datatype as 64 bit float.
	#x = np.array(x, dtype = np.float64)
	#y = np.array(y, dtype = np.float64)

	df = pd.read.table('logistic_regression_data.txt', sep = ',', names = ('featureOne', 'featureTwo', 'label'))
	y = np.array(df['label']).T
	df = np.array(df)
	x = [:,:2]
	
	#normalize the data
	
	x_test, y_test, x_train, y_train = train_test_split(x,y, test_size = 0.2, random_state = 0)
	
	
	glm = CustomLogisticRegression(x, y)

	plt.scatter(x, y)
	#plt.scatter(ip, line, color = "red")
	plt.plot(x, reg)
	plt.show()

if __name__ == "__main__":
	main()