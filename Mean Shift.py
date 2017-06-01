#================================================================================================================
#----------------------------------------------------------------------------------------------------------------
#									MEAN SHIFT
#----------------------------------------------------------------------------------------------------------------
#================================================================================================================

import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
import pandas
import datetime
from sklearn import preprocessing, cross_validation

#for plotting
plt.style.use('ggplot')

class CustomMS:
	
	def __init__(self):
		pass
		
			
def main():
	'''
	Pclass Passenger Class (1 = 1st; 2 = 2nd; 3 = 3rd)
	survival Survival (0 = No; 1 = Yes)
	name Name
	sex Sex
	age Age
	sibsp Number of Siblings/Spouses Aboard
	parch Number of Parents/Children Aboard
	ticket Ticket Number
	fare Passenger Fare (British pound)
	cabin Cabin
	embarked Port of Embarkation (C = Cherbourg; Q = Queenstown; S = Southampton)
	boat Lifeboat
	body Body Identification Number
	home.dest Home/Destination
	'''

	df = pd.read_excel('data/titanic.xls')

	original_df = pd.DataFrame.copy(df)
	df.drop(['body','name'], 1, inplace=True)
	df.fillna(0,inplace=True)

	def handle_non_numerical_data(df):
		
		# handling non-numerical data: must convert.
		columns = df.columns.values

		for column in columns:
			text_digit_vals = {}
			def convert_to_int(val):
				return text_digit_vals[val]

			#print(column,df[column].dtype)
			if df[column].dtype != np.int64 and df[column].dtype != np.float64:
				
				column_contents = df[column].values.tolist()
				#finding just the uniques
				unique_elements = set(column_contents)
				# great, found them. 
				x = 0
				for unique in unique_elements:
					if unique not in text_digit_vals:
						# creating dict that contains new
						# id per unique string
						text_digit_vals[unique] = x
						x+=1
				# now we map the new "id" vlaue
				# to replace the string. 
				df[column] = list(map(convert_to_int,df[column]))

		return df

	df = handle_non_numerical_data(df)
	df.drop(['ticket','home.dest'], 1, inplace=True)

	X = np.array(df.drop(['survived'], 1).astype(float))
	X = preprocessing.scale(X)
	y = np.array(df['survived'])
	ms = CustomMS()

	ms.fit(dataset = dataset)
	
	
if __name__ == "__main__":
	main()
