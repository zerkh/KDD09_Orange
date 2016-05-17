import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score, accuracy_score, precision_recall_fscore_support, confusion_matrix
from sklearn.cross_validation import KFold
import time
from utils import get_data

from sklearn.svm import SVC

"""
Train and validate with svm

Params: Attributes, targets
Return: None
"""
def train_with_svm(X, Y):
	model = SVC(max_iter=50)

	kf = KFold(len(X), 10)

	auc_result = []
	pre_result = []
	rec_result = []
	f1_result = []

#Cross-validation
	idx = 0
	st = time.clock()
	for train_idx, test_idx in kf:
		idx += 1
		X_train = X[train_idx]
		X_test = X[test_idx]

		Y_train = Y[train_idx]
		Y_test = Y[test_idx]

		model.fit(X_train, Y_train)
		pred = model.predict(X_test)

		pre_auc = roc_auc_score(Y_test, pred)
		pre_pre, pre_rec, pre_f1, _ = precision_recall_fscore_support(Y_test, pred, average='binary')

		auc_result.append(pre_auc)
		pre_result.append(pre_pre)
		rec_result.append(pre_rec)
		f1_result.append(pre_f1)

	et = time.clock()

#Calculate average of each index
	auc = sum(auc_result) / idx
	precise = sum(pre_result) / idx
	recall = sum(rec_result) / idx
	f1 = sum(f1_result) / idx

	return auc, precise, recall, f1

if __name__ == "__main__":
	print "Train and validate with svm:"
	print "\t\t\tAUC\t\tPrecise\t\tRecall\t\tF1"

	X, app_Y = get_data("../data/orange_aft_clean.csv", attr="appetency")
	auc, pre, rec, f1 = train_with_svm(X, app_Y)
	print "App\t\t%g\t\t%g\t\t%g\t\t%g" %(auc, pre, rec, f1)

	X, churn_Y = get_data("../data/orange_aft_clean.csv", attr="churn")
	auc, pre, rec, f1 = train_with_svm(X, churn_Y)
	print "Churn\t\t%g\t\t%g\t\t%g\t\t%g" %(auc, pre, rec, f1)

	X, up_Y = get_data("../data/orange_aft_clean.csv", attr="upselling")
	auc, pre, rec, f1 = train_with_svm(X, up_Y)
	print "Up\t\t%g\t\t%g\t\t%g\t\t%g" %(auc, pre, rec, f1)