#================================================================================================================
#----------------------------------------------------------------------------------------------------------------
#									K NEAREST NEIGHBOURS
#----------------------------------------------------------------------------------------------------------------
#================================================================================================================

import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
import multiprocessing as mp
import pandas as pd
import random
from collections import Counter
from sklearn import preprocessing
from itertools import repeat


#for plotting
plt.style.use('ggplot')

class CustomKNN:
	
	def __init__(self):
		self.accurate_predictions = 0
		self.total_predictions = 0
		self.accuracy = 0.0
		
	def distances_for_parallel(self, features, incoming, group):
		return (np.linalg.norm(np.array(features)- np.array(incoming)), group)

	def predict(self, training_data, to_predict, k = 3):
		if len(training_data) >= k:
			print("K cannot be smaller than the total voting groups(ie. number of training data points)")
			return
		
		pool = mp.Pool(processes= 1)

		distributions = []
		for group in training_data:
			euclidean_distances = [ pool.starmap(self.distances_for_parallel, zip(training_data[group], repeat(to_predict), repeat(group)))]
			for iter in euclidean_distances:
				for j in iter:
					distributions.append(j)
				
		
		#for i in range(1, len(training_data)):	
		#	distributions[0].extend(distributions[i])
		

		
		results = [i for i in sorted(distributions)[:k]]
		result = Counter(results).most_common(1)[0][0]
		confidence = Counter(results).most_common(1)[0][1]/k
		print(results)
		return result, confidence
	
	def test(self, test_set, training_set):
		for group in test_set:
			for data in test_set[group]:
				predicted_class,confidence = self.predict(training_set, data, k =3)
				if predicted_class == group:
					self.accurate_predictions += 1
				else:
					print("Wrong classification with confidence " + str(confidence * 100) + " and class " + str(predicted_class))
				self.total_predictions += 1
		self.accuracy = 100*(self.accurate_predictions/self.total_predictions)
		print("\nAcurracy :", str(self.accuracy) + "%")

def mod_data(df):
	df.replace('?', -999999, inplace = True)
	
	df.replace('yes', 4, inplace = True)
	df.replace('no', 2, inplace = True)

	df.replace('notpresent', 4, inplace = True)
	df.replace('present', 2, inplace = True)
	
	df.replace('abnormal', 4, inplace = True)
	df.replace('normal', 2, inplace = True)
	
	df.replace('poor', 4, inplace = True)
	df.replace('good', 2, inplace = True)
	
	df.replace('ckd', 4, inplace = True)
	df.replace('notckd', 2, inplace = True)

def main():
	df = pd.read_csv(r".\data\chronic_kidney_disease.csv")
	mod_data(df)
	dataset = df.astype(float).values.tolist()
	
	#Normalize the data
	x = df.values #returns a numpy array
	min_max_scaler = preprocessing.MinMaxScaler()
	x_scaled = min_max_scaler.fit_transform(x)
	df = pd.DataFrame(x_scaled) #Replace df with normalized values
	
	#Shuffle the dataset
	random.shuffle(dataset)

	#20% of the available data will be used for testing
	test_size = 0.2

	#The keys of the dict are the classes that the data is classfied into
	training_set = {2: [], 4:[]}
	test_set = {2: [], 4:[]}
	
	#Split data into training and test for cross validation
	training_data = dataset[:-int(test_size * len(dataset))]
	test_data = dataset[-int(test_size * len(dataset)):]
	
	#Insert data into the training set
	for record in training_data:
		training_set[record[-1]].append(record[:-1]) # Append the list in the dict will all the elements of the record except the class

	#Insert data into the test set
	for record in training_data:
		test_set[record[-1]].append(record[:-1]) # Append the list in the dict will all the elements of the record except the class

	knn = CustomKNN()
	knn.test(test_set, training_set)

if __name__ == "__main__":
	main()
