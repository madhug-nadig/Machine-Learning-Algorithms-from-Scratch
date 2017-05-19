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

#for plotting
plt.style.use('ggplot')

class CustomMS:
	
	def __init__(self):
		pass
		
			
def main():

	dataset = { -1 : np.array([[2,3],[4,5],[2,1]]), 1: np.array([[5,6], [8,8], [9,9]]) }
	ms = CustomMS()

	ms.fit(dataset = dataset)
	pred = ms.predict(attrs = [2,2])
	
if __name__ == "__main__":
	main()
