#================================================================================================================
#----------------------------------------------------------------------------------------------------------------
#											SIMPLE LINEAR REGRESSION
#----------------------------------------------------------------------------------------------------------------
#================================================================================================================

#Simple linear regression is applied to stock data, where the x values are time and y values are the stock closing price.
#This is not an ideal application of simple linear regression, but it suffices to be a good experiment.

import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
import pandas
import datetime

#Quandl for getting stock data
import quandl

#for plotting
plt.style.use('ggplot')

#arithmetic mean
def am(arr):
	tot = 0.0
	for i in arr:
		tot+= i
	return tot/len(arr)

#finding the slope in best fit line
def best_fit(dimOne, dimTwo):
	return  ( (am(dimOne) * am(dimTwo) ) - am(dimOne*dimTwo) ) / ( am(dimOne)**2 - am(dimOne**2) )

#finding the best fit intercept
def y_intercept(m, dimOne ,dimTwo):
	return am( dimTwo ) - ( m* am(dimOne) )

#predict for future values based on model
def predict(ip, slope, intercept):
	ip = np.array(ip)
	predicted = [(slope*param) + intercept for param in ip]
	return predicted
	
#find the squared error
def squared_error(original, model):
	return sum((model - original) **2)

#find co-efficient of determination for R^2
def cod(original, model):
	am_line = [am(original) for y in original]
	sq_error = squared_error(original, model)
	sq_error_am = squared_error(original, am_line)
	return 1 - (sq_error/sq_error_am)

def main():
	stk = quandl.get("WIKI/TSLA")

	stk = stk.reset_index()
	#Add them headers
	stk = stk[['Date','Adj. Open','Adj. High','Adj. Low','Adj. Close', 'Volume']]

	stk['Date'] = pandas.to_datetime(stk['Date'])    
	stk['Date'] = (stk['Date'] - stk['Date'].min())  / np.timedelta64(1,'D')


	#The column that needs to be forcasted using linear regression
	forecast_col = 'Adj. Close'
	stk.fillna(-999999, inplace = True)

	stk['label'] = stk[forecast_col]


	#IN CASE THE INPUT IS TO BE TAKEN IN FROM THE COMMAN PROMPT

	#takes in input from the user
	#x = list(map(int, input("Enter x: \n").split()))
	#y = list(map(int, input("Enter y: \n").split()))

	#convert to an numpy array with datatype as 64 bit float.
	#x = np.array(x, dtype = np.float64)
	#y = np.array(y, dtype = np.float64)

	stk.dropna(inplace = True)

	#x = np.array(stk['Adj. Open'])
	x = np.array(stk['Date'])

	y = np.array(stk['label'])

	slope = best_fit(x, y) #find slope
	intercept = y_intercept(slope, x, y) #find the intercept

	ip = list(map(int, input("Enter x to predict y: \n").split()))

	line = predict(ip, slope, intercept) #predict based on model
	reg = [(slope*param) + intercept for param in x]

	print("Predicted value(s) after linear regression :", line)

	r_sqrd = cod(y, reg)
	print("R^2 Value: " ,r_sqrd)
	
	plt.scatter(x, y)
	plt.scatter(ip, line, color = "red")
	plt.plot(x, reg)
	plt.show()

if __name__ == "__main__":
	main()