import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style

plt.style.use('ggplot')

x = list(map(int, input("Enter x: \n").split()))
y = list(map(int, input("Enter y: \n").split()))

x = np.array(x, dtype = np.float64)
y = np.array(y, dtype = np.float64)


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