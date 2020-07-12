#================================================================================================================
#----------------------------------------------------------------------------------------------------------------
#									SUPPORT VECTOR MACHINE
#----------------------------------------------------------------------------------------------------------------
#================================================================================================================

import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
import pandas as pd
import datetime
import random

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


		all_feature_values = []

		for yi, attrs in self.dataset.items():
			for attr in attrs:
				for f in attr:
					all_feature_values.append(f)

		self.max_attr = max(all_feature_values)
		self.min_attr = min(all_feature_values)
		del all_feature_values

		step_size = [self.max_attr * 0.1,self.max_attr * 0.01,self.max_attr * 0.005]
		latest_optimum = 10 * self.max_attr

		b_range = 5
		b_multiple = 5

		trans =  [[1, 1, 1, 1], [1, 1, 1, -1], [1, 1,-1, 1], [1, 1,-1,-1], [1 ,-1, 1, 1], [1 ,-1, -1, 1], [1 ,-1, 1, -1], [1 ,-1, -1, -1], [-1, 1, 1, 1], [-1, 1, 1, -1], [-1, 1, -1, 1], [-1, 1, -1, -1], [-1,-1, 1, 1], [-1,-1, 1, -1], [-1,-1, -1, 1], [-1,-1, -1, -1]]

		for step in step_size:
			W = np.array([latest_optimum,latest_optimum, latest_optimum, latest_optimum])
			optimization_flag = False

			while not optimization_flag:
				for b in np.arange(-1*(self.max_attr* b_range ), self.max_attr * b_range, step * b_multiple):
					for transformation in trans:
						W_t = W * transformation
						found = True
						for yi, attributes in self.dataset.items():
							for xi in attributes:
								if not (yi * (np.dot(xi, W_t)  +b )) >= 1:
									found = False
									break
						if found:
							options[np.linalg.norm(W_t)] = [W_t, b]
				if W[0]<0:
					optimization_flag = True
				else:
					W = np.array(list(map(lambda w: w-step, W)))

			norms = min([n for n in options])
			self.W = options[norms][0]
			self.b = options[norms][1]

			latest_optimum = options[norms][0][0] + step*2

	def predict(self, attrs):
		#sign of the X(i).W + b defines the class
		dot_product = np.dot(np.array(attrs), self.W)
		classification = np.sign(dot_product + self.b)

		return classification

	def test(self, test_set):
		self.accurate_predictions, self.total_predictions = 0, 0 
		for group in test_set:
			for data in test_set[group]:
				predicted_class = self.predict(data)
				if predicted_class == group:
					self.accurate_predictions += 1
				self.total_predictions += 1
		self.accuracy = 100*(self.accurate_predictions/self.total_predictions)
		print("\nAcurracy :", str(self.accuracy) + "%")

def main():

	df = pd.read_csv(r"./data/iris.csv") #Reading from the data file

	df.replace('Iris-setosa', -1, inplace = True)
	df.replace('Iris-versicolor', 1, inplace = True)

	dataset = df.astype(float).values.tolist()
	#Shuffle the dataset

	random.shuffle(dataset)

	#20% of the available data will be used for testing

	test_size = 0.20

	#The keys of the dict are the classes that the data is classfied into

	training_set = {-1: [], 1:[]}
	test_set = {-1: [], 1:[]}
	training_data = dataset[:-int(test_size * len(dataset))]
	test_data = dataset[-int(test_size * len(dataset)):]

	for record in training_data:
		#Append the list in the dict will all the elements of the record except the class
		training_set[record[-1]].append(record[:-1])
		#Insert data into the test set

	for record in test_data:
		# Append the list in the dict will all the elements of the record except the class
		test_set[record[-1]].append(record[:-1])

	dataset = { -1 : np.array([[2,3],[4,5],[2,1]]), 1: np.array([[5,6], [8,8], [9,9]]) }
	svm = CustomSVM()

	svm.fit(dataset = training_set)
	svm.test(test_set)
if __name__ == "__main__":
	main()
