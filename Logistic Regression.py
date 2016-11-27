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


def main():

	glm = CustomLogisticRegression()



	#IN CASE THE INPUT IS TO BE TAKEN IN FROM THE COMMAND PROMPT

	#takes in input from the user
	x = list(map(int, input("Enter x: \n").split()))
	y = list(map(int, input("Enter y: \n").split()))

	#convert to an numpy array with datatype as 64 bit float.
	x = np.array(x, dtype = np.float64)
	y = np.array(y, dtype = np.float64)


	
	plt.scatter(x, y)
	#plt.scatter(ip, line, color = "red")
	plt.plot(x, reg)
	plt.show()

if __name__ == "__main__":
	main()