#================================================================================================================
#----------------------------------------------------------------------------------------------------------------
#													LOGISTIC REGRESSION
#----------------------------------------------------------------------------------------------------------------
#================================================================================================================

import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
import pandas as pd
import datetime
from sklearn.cross_validation import train_test_split


#for plotting
plt.style.use('ggplot')

#Using gradient decent here to arrive at the optimal.


class CustomLogisticRegression:
	
	def __init__(self, x, y, tolerence = 0.00001):
		self.tolerence = tolerence
		self.cost = []
		self.alpha = 0.1
		self.lambd = 0.25
		self.iter = 2500
		
		#initialie theta
		self.theta = np.random.rand(x.shape[1],1)
	
	#the cost function
	def cost_fn(self, m, x, y):
		pass

	#The sigmoid function
	def sigmoid_function(z):
		return 1.0 / ( 1.0 + math.e**(-1*z) ) #Using 1.0 to make it a floating point type
	
	#Gradient function
	def gradients(self, m, x, y):
		zrd = self.theta
		zrd[0, :] = 0
		h = sigmoid_function(np.dot(x, self.theta))
		return ( 1.0/m ) * np.dot(x.T, ( h - y ) ) + (float(self.lambd)/m) * zrd

	#This is batch
	def descent(self, x, y):
		for i in range(self.iter):
			self.cost.append( cost_fn(x.shape[0], x, y))
			gradientz = gradients(x.shape[0], x , y)
			
			#Change theta based on the "gradientz"
			self.theta[0, :] = gradientz[0, :] - self.alpha * gradientz[0, :]
			self.theta[1, :] = gradientz[1:, :] - self.alpha * gradientz[1:, :]
		
		pred = np.dot(x, self.theta)
		pred[ pred >= 0.5 ] = 1
		pred[ pred < 0.5 ] = 0

def main():

	#IN CASE THE INPUT IS TO BE TAKEN IN FROM THE COMMAND PROMPT

	#takes in input from the user
	#x = list(map(int, input("Enter x: \n").split()))
	#y = list(map(int, input("Enter y: \n").split()))

	#convert to an numpy array with datatype as 64 bit float.
	#x = np.array(x, dtype = np.float64)
	#y = np.array(y, dtype = np.float64)

	df = pd.read_table('.\data\logistic_regression_data.txt', sep = ',', names = ('featureOne', 'featureTwo', 'label'))
	y = np.array(df['label']).T
	df = np.array(df)
	x = df[:,:2]
	
	#normalize the data
	df = (df - df.mean()) / (df.max() - df.min())

	
	x_test, y_test, x_train, y_train = train_test_split(x,y, test_size = 0.1, random_state = 0)
	
	glm = CustomLogisticRegression(x, y)
	
	plt.scatter(x[:,0], y)
	
	plt.show()

if __name__ == "__main__":
	main()