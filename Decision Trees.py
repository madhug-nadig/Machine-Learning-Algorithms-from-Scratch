#================================================================================================================
#----------------------------------------------------------------------------------------------------------------
#									DECISION TREES
#----------------------------------------------------------------------------------------------------------------
#================================================================================================================
from math import log
import pandas as pd

def majorityCnt(classList):
	classCount={}
	for vote in classList:
		if vote not in classCount.keys(): classCount[vote] = 0
		classCount[vote] += 1
	sortedClassCount = sorted(classCount.items(), reverse=True)
	return sortedClassCount[0][0]

#for calculting entropy
def calcShannonEnt(dataSet):
	numEntries = len(dataSet)
	labelCounts = {}
	for featVec in dataSet:
		currentLabel = featVec[-1]
		if currentLabel not in labelCounts.keys():
		 labelCounts[currentLabel] = 0
		 labelCounts[currentLabel] += 1
	shannonEnt = 0.0
	for key in labelCounts:
		prob = float(labelCounts[key])/numEntries
		shannonEnt -= prob * log(prob,2)
	return shannonEnt

def createDataSet():
	dataSet = [[1, 1, 'yes'],
	         [1, 1, 'yes'],             #just an example
	         [1, 0, 'no'],
	         [0, 1, 'no'],
	         [0, 1, 'no']]
	labels = ['no surfacing','flippers']
	return dataSet, labels
myDat,labels=createDataSet()
def splitDataSet(dataSet, axis, value):
	retDataSet = []
	for featVec in dataSet:
		if featVec[axis] == value:
			reducedFeatVec = featVec[:axis]
			reducedFeatVec.extend(featVec[axis+1:])
			retDataSet.append(reducedFeatVec)
	return retDataSet
#choosing the best feature to split
def chooseBestFeatureToSplit(dataSet):
	numFeatures = len(dataSet[0]) - 1
	baseEntropy = calcShannonEnt(dataSet)
	bestInfoGain = 0.0; bestFeature = -1
	for i in range(numFeatures):
		featList = [example[i] for example in dataSet]
		uniqueVals = set(featList)
		newEntropy = 0.0
		for value in uniqueVals:
			subDataSet = splitDataSet(dataSet, i, value)
			prob = len(subDataSet)/float(len(dataSet))
			newEntropy += prob * calcShannonEnt(subDataSet)
		infoGain = baseEntropy - newEntropy
		if (infoGain > bestInfoGain):
			bestInfoGain = infoGain
			bestFeature = i
	return bestFeature
print("the best feature to split is",chooseBestFeatureToSplit(myDat))

#function to build tree recursively
def createTree(dataSet,labels):
	classList = [example[-1] for example in dataSet]
	if classList.count(classList[0]) == len(classList):
		return classList[0]
	if len(dataSet[0]) == 1:
		return majorityCnt(classList)
	bestFeat = chooseBestFeatureToSplit(dataSet)
	bestFeatLabel = labels[bestFeat]
	myTree = {bestFeatLabel:{}}
	del(labels[bestFeat])
	featValues = [example[bestFeat] for example in dataSet]
	uniqueVals = set(featValues)
	for value in uniqueVals:
		subLabels = labels[:]
		myTree[bestFeatLabel][value] = createTree(splitDataSet\
		                   (dataSet, bestFeat, value),subLabels)
	return myTree

def main():
	df = pd.read_csv(r".\data\titanic.csv") #Reading from the data file
	# Sex param
    df.replace('male', 0, inplace = True)
    df.replace('female', 1, inplace = True)

    # Embarked param
    df.replace('S', 0, inplace = True)
    df.replace('C', 1, inplace = True)
    df.replace('Q', 2, inplace = True)

    dataset = df.astype(float).values.tolist()
    #Shuffle the dataset
    random.shuffle(dataset) #import random for this
 	
	#20% of the available data will be used for testing

    test_size = 0.2

    #The keys of the dict are the classes that the data is classified into

    training_set = {0: [], 1:[]}
    test_set = {0: [], 1: []}

	print(createTree(myDat,labels))
