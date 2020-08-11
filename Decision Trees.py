# ================================================================================================================
# ----------------------------------------------------------------------------------------------------------------
#									DECISION TREES
# ----------------------------------------------------------------------------------------------------------------
# ================================================================================================================
from math import log
import pandas as pd
import random


class CustomDecisionTree():
    def __init__(self):
        pass

    def majorityCnt(self, classList):
        classCount = {}
        for vote in classList:
            if vote not in classCount.keys():
                classCount[vote] = 0
            classCount[vote] += 1
        sortedClassCount = sorted(classCount.items(), reverse=True)
        return sortedClassCount[0][0]

    # for calculting entropy
    def calcShannonEnt(self, dataSet):
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
            shannonEnt -= prob * log(prob, 2)
        return shannonEnt

    def splitDataSet(self, dataSet, axis, value):
        retDataSet = []
        for featVec in dataSet:
            if featVec[axis] == value:
                reducedFeatVec = featVec[:axis]
                reducedFeatVec.extend(featVec[axis+1:])
                retDataSet.append(reducedFeatVec)
        return retDataSet

    # choosing the best feature to split
    def chooseBestFeatureToSplit(self, dataSet, labels):
        numFeatures = len(dataSet[0]) - 1
        baseEntropy = self.calcShannonEnt(dataSet)
        bestInfoGain = -1
        bestFeature = 0
        for i in range(numFeatures):
            featList = [example[i] for example in dataSet]
            uniqueVals = set(featList)
            newEntropy = 0.0
            for value in uniqueVals:
                subDataSet = self.splitDataSet(dataSet, i, value)
                prob = len(subDataSet)/float(len(dataSet))
                newEntropy += prob * self.calcShannonEnt(subDataSet)
            infoGain = baseEntropy - newEntropy
            print(infoGain, bestInfoGain)
            if (infoGain > bestInfoGain):
                bestInfoGain = infoGain
                bestFeature = i

        print("the best feature to split is", labels[bestFeature])
        return bestFeature

    # function to build tree recursively
    def createTree(self, dataSet, labels):
        classList = [example[-1] for example in dataSet]
        if len(classList) is 0:
            return

        if classList.count(classList[0]) == len(classList):
            return classList[0]
        if len(dataSet[0]) == 1:
            return self.majorityCnt(classList)
        featureVectorList = [row[:len(row)-1] for row in dataSet]
        bestFeat = self.chooseBestFeatureToSplit(featureVectorList, labels)
        bestFeatLabel = labels[bestFeat]
        myTree = {bestFeatLabel: {}}
        del(labels[bestFeat])
        featValues = [example[bestFeat] for example in dataSet]
        uniqueVals = set(featValues)
        for value in uniqueVals:
            subLabels = labels[:]
            myTree[bestFeatLabel][value] = self.createTree(
                self.splitDataSet(dataSet, bestFeat, value), subLabels)
        return myTree


def main():
    df = pd.read_csv("./data/test.csv")  # Reading from the data file
    # Sex param
    df.replace('male', 0, inplace=True)
    df.replace('female', 1, inplace=True)

    # Embarked param
    df.replace('S', 0, inplace=True)
    df.replace('C', 1, inplace=True)
    df.replace('Q', 2, inplace=True)
    df['embarked'] = df['embarked'].fillna(1)
    dataset = df.astype(float).values.tolist()
    labels = ['pclass', 'sex', 'embarked', 'survived']

    # Shuffle the dataset
    random.shuffle(dataset)  # import random for this

    custom_DTree = CustomDecisionTree()
    print(custom_DTree.createTree(dataset, labels))


if __name__ == "__main__":
    main()
