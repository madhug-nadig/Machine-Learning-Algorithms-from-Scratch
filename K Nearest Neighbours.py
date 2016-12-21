#================================================================================================================
#----------------------------------------------------------------------------------------------------------------
#									K NEAREST NEIGHBOURS
#----------------------------------------------------------------------------------------------------------------
#================================================================================================================


import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
import pandas as pd
import random

#for plotting
plt.style.use('ggplot')

class CustomKNN:
	
	def __init__(self):
		pass

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

if __name__ == "__main__":
	main()
