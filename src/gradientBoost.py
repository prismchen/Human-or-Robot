import pandas as pd
import numpy as np

import sklearn.preprocessing as preprocessing
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
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

clf = []
clfNum = 5
for i in xrange(clfNum):
	clf.append(GradientBoostingClassifier(n_estimators=3000, learning_rate=0.01, max_depth=None, min_samples_leaf=1))

Y = Tuples['outcome'].values # Yb is all the outcome values including -1,0,1
X = 1.0*Tuples.drop('outcome', 1) # Xb is what Tuples removed outcome values remains 


X = preprocessing.normalize(X.values, axis=0) # normalize each feature
testTupleBegin = Tuples[Tuples.outcome>=0].shape[0] # number of train cases 

XTest = X[testTupleBegin:, :] # get all test cases
YTrain = Y[0:testTupleBegin] # get all labels for training cases
XTrain = X[0:testTupleBegin, :] # get all tuples for training cases 


testPrediction = np.zeros(XTest.shape[0])
for j in range(clfNum):
	clf[j].fit(XTrain, YTrain)
	a = clf[j].predict_proba(XTest)[:,1]
	testPrediction += a
	
testPrediction = 1.0*testPrediction/clfNum
		
outputFile['prediction'] = pd.Series(testPrediction, index=outputFile.index)
outputFile.drop('index', 1)
outputFile.to_csv('GBResult.csv', sep=',', index=False, header=True, columns=['bidder_id', 'prediction'])

