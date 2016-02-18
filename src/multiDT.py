import pandas as pd
import numpy as np

import sklearn.preprocessing as preprocessing
from sklearn import tree
from sklearn.metrics import roc_auc_score


Tuples = pd.read_csv('features.csv')
outputFile = Tuples[Tuples.outcome==-1]['bidder_id'].reset_index() # separate out the test cases as output
Tuples.drop(Tuples[Tuples.outcome >=0][Tuples.n_bids.isnull()].index, inplace=True) # delete rows where bidder make no bids in place 
Tuples = Tuples.sort(['dt_others_median']) #  time between a user's bid and the previous bid in a given auction? (dt_others_median)
Tuples = Tuples.fillna(method='pad') # fill N/A forward (copy value upstream)
Tuples = Tuples.fillna(method='backfill') # fill N/A backward
Tuples.sort_index(inplace=True)
Tuples = Tuples.fillna(0)
Tuples = Tuples.drop('most_common_country', 1)
Tuples = Tuples.drop('bidder_id', 1)


def learn(Tuples, treeNum, max_depth, numSampleEachT, zoRatio, max_features):

	clf = []
	Xtrees = []
	Ytrees = []

	for i in xrange(treeNum):
		clf.append(tree.DecisionTreeClassifier(max_depth=max_depth, max_features=max_features))

	Y = Tuples['outcome'].values # Y is all the outcome values including -1,0,1
	X = 1.0*Tuples.drop('outcome', 1) # X is what Tuples removed outcome values remains
	X = preprocessing.normalize(X.values, axis=0) # normalize each feature

	testTupleBegin = Tuples[Tuples.outcome>=0].shape[0] # number of train cases 

	XTest = X[testTupleBegin:, :] # get all test cases
	YTrain = Y[0:testTupleBegin] # get all labels for training cases
	XTrain = X[0:testTupleBegin, :] # get all tuples for training cases 

	YTrain = pd.DataFrame(YTrain)
	XTrain = pd.DataFrame(XTrain)

	numFolds = 5
	testFold = []
	testLable = []
	trainFold = []
	trainLable = []

	scores0 = []
	scores = []

	for j in xrange(1):
		for i in xrange(numFolds):

			s = i*len(XTrain)/numFolds
			e = s + len(XTrain)/numFolds

			testFold = XTrain.iloc[s:e]
			testLable  = YTrain.iloc[s:e]

			trainFold = XTrain.drop(XTrain.index[s:e])
			trainLable = YTrain.drop(YTrain.index[s:e])

			XValidTree = []
			YValidTree = []

			validPrediction = np.zeros(testFold.shape[0])

			trainFoldIndex = pd.DataFrame(range(len(trainFold)))
			for i in xrange(treeNum):
				trainFoldSubIndex = trainFoldIndex.sample(int(trainFold.shape[0]*numSampleEachT), replace=True)[0].values

				trainFoldSubIndex = balance(trainFoldSubIndex,trainLable,zoRatio)

				clf[i].fit(trainFold.iloc[trainFoldSubIndex],trainLable.iloc[trainFoldSubIndex])
				a = clf[i].predict(testFold)
				validPrediction += a

			validPrediction = 1.0*validPrediction/treeNum

			s = roc_auc_score(testLable, validPrediction)
			scores0.append(s)

		scores.append(np.mean(scores0))

	print "Average Score: "+ str(np.mean(scores))



	testPrediction = np.zeros(XTest.shape[0])

	for i in xrange(treeNum):
		index = pd.DataFrame(range(len(XTrain)))
		subIndex = index.sample(int(XTrain.shape[0]*numSampleEachT), replace=True)[0].values

		subIndex = balance(subIndex,YTrain,zoRatio)

		clf[i].fit(XTrain.loc[subIndex], YTrain.loc[subIndex])
		a = clf[i].predict(XTest)
		testPrediction += a
		
	testPrediction = 1.0*testPrediction/treeNum
	return testPrediction

def balance(trainFoldSubIndex,trainLable,zoRatio):

	num1 = trainLable.iloc[trainFoldSubIndex][trainLable[0]==1].shape[0]
	numDrop = trainFoldSubIndex.shape[0] - zoRatio*num1 - num1

	ret = trainFoldSubIndex

	i = 0
	indexDrop = []
	for index in range(len(trainFoldSubIndex)):
		if trainLable.iloc[trainFoldSubIndex[index]][0] == 0:
			indexDrop.append(index)
			i = i + 1
		if i >= numDrop:
			break

	ret = np.delete(ret, indexDrop)
	return ret 






# Main 


# learn(Tuples, treeNum, max_depth, numSampleEachT, zoRatio, max_features):
testPrediction = learn(Tuples,200,30,0.65,4,45)




outputFile['prediction'] = pd.Series(testPrediction, index=outputFile.index)
outputFile.drop('index', 1)
outputFile.to_csv('multiDT.csv', sep=',', index=False, header=True, columns=['bidder_id', 'prediction'])