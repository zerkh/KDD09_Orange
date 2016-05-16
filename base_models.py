import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.cross_validation import KFold
import time
from utils import get_data

from sklearn.svm import SVC

"""
Train and validate with svm

Params: Attributes, targets
Return: None
"""
def train_with_svm(X, app_Y, churn_Y, up_Y):
	print "Train and validate with svm:"
	app_model = SVC(max_iter=50)
	churn_model = SVC(max_iter=50)
	up_model = SVC(max_iter=50)

	kf = KFold(len(X), 10)

	auc_app_result = []
	acc_app_result = []
	auc_churn_result = []
	acc_churn_result = []
	auc_up_result = []
	acc_up_result = []

	idx = 0
	st = time.clock()
	for train_idx, test_idx in kf:
		idx += 1
		X_train = X[train_idx]
		X_test = X[test_idx]

		app_Y_train = app_Y[train_idx]
		app_Y_test = app_Y[test_idx]

		churn_Y_train = churn_Y[train_idx]
		churn_Y_test = churn_Y[test_idx]

		up_Y_train = up_Y[train_idx]
		up_Y_test = up_Y[test_idx]

		app_model.fit(X_train, app_Y_train)
		pred = app_model.predict(X_test)
		pre_app_auc = roc_auc_score(app_Y_test, pred)
		auc_app_result.append(pre_app_auc)
		acc_app_result.append(accuracy_score(app_Y_test, pred))

		churn_model.fit(X_train, churn_Y_train)
		pred = churn_model.predict(X_test)
		pre_churn_auc = roc_auc_score(churn_Y_test, pred)
		churn_auc += pre_churn_auc
		auc_churn_result.append(pre_churn_auc)
		acc_churn_result.append(accuracy_score(churn_Y_test, pred))

		up_model.fit(X_train, up_Y_train)
		pred = up_model.predict(X_test)
		pre_up_auc = roc_auc_score(up_Y_test, pred)
		up_auc += pre_up_auc
		auc_up_result.append(pre_up_auc)
		acc_up_result.append(accuracy_score(up_Y_test, pred))
	et = time.clock()

	app_auc = sum(auc_app_result) / idx
	churn_auc = sum(auc_churn_result) / idx
	up_auc = sum(auc_up_result) / idx

	app_pre

	print "\t\t\tAUC\t\tPrecise\t\tRecall\t\tF1"
	for i in xrange(idx):
		print "Fold%d\t\t%g\t\t%g\t\t%g\t\t%g" %(i, auc_app)

if __name__ == "__main__":
	X, app_Y, churn_Y, up_Y = get_data("../data/orange_aft_clean.csv")

	train_with_svm(X, app_Y, churn_Y, up_Y)
