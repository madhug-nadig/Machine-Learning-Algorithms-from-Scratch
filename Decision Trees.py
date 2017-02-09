#================================================================================================================
#----------------------------------------------------------------------------------------------------------------
#									DECISION TREES
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

class CustomDecisionTrees:
	
	def __init__(self):
		pass

class decisionnode:
	def __init__(self,col = -1,value=None,results=None,tb=None,fb=None):
		self.col=col
		self.value=value
		self.results=results
		self.tb=tb
		self.fb=fb
		
	def buildtree(rows,scoref = entropy):
		
		if len(rows)==0: 
			return decisionnode()
		
		current_score = scoref(rows)
		best_gain=0.0
		best_criteria=None
		best_sets=None
		column_count=len(rows[0])-1
		
		for col in range(0,column_count):
			column_values={}
		
		for row in rows:
			column_values[row[col]]=1

		for value in column_values.keys():
			(set1,set2) = divid(rows,col,value)
		
		p = float(len(set1))/len(rows) #p is the size of a child set relative to its parent
		gain = current_score - p * scoref( set1 )-( 1 - p ) * scoref( set2 ) #cf. formula information gain
		
		if gain>best_gain and len(set1)>0 and len(set2)>0: #set must not be empty
			best_gain = gain
			best_criteria = (col,value)
			best_sets = (set1,set2)
		
		
def main():
	def entropy():
		log2 = lambda x:log(x)/log(2)
		results = uniquecounts(rows)
		
		entropy_value = 0.0
		for r in results.keys():
			p = float(results[r])/len(rows)
			entropy_value = entropy_value - p * log2(p)
		return entropy_value
	
	
	def uniquecounts(rows):
		results={}
		for row in rows:
			r=row[len(row) - 1]
			if r not in results: 
				results[r]=0
			results[r]+=1
		return results

	def divid(rows,column,value):

		split_function=None

		if isinstance(value,int) or isinstance(value,float): # check if the value is a number
			split_function=lambda row:row[column]>=value
		else:
			split_function=lambda row:row[column]==value

		# Divide the rows into two sets and return them
		classOne=[row for row in rows if split_function(row)]
		classTwo=[row for row in rows if not split_function(row)]
		return (classOne,classTwo)

	data =[['slashdot','USA','yes',18,'None'],
		['google','France','yes',23,'Premium'],
		['digg','USA','yes',24,'Basic'],
		['kiwitobes','France','yes',23,'Basic'],
		['google','UK','no',21,'Premium'],
		['(direct)','New Zealand','no',12,'None'],
		['(direct)','UK','no',21,'Basic'],
		['google','USA','no',24,'Premium'],
		['slashdot','France','yes',19,'None'],
		['digg','USA','no',18,'None'],
		['google','UK','no',18,'None'],
		['kiwitobes','UK','no',19,'None'],
		['digg','New Zealand','yes',12,'Basic'],
		['slashdot','UK','no',21,'None'],
		['google','UK','yes',18,'Basic'],
		['kiwitobes','France','yes',19,'Basic']]

	
	
	dt = CustomDecisionTrees()

if __name__ == "__main__":
	main()
