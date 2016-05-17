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
def train_with_svm(X, app_Y, churn_Y, up_Y):
	print "Train and validate with svm:"
	app_model = SVC(max_iter=50)
	churn_model = SVC(max_iter=50)
	up_model = SVC(max_iter=50)

	kf = KFold(len(X), 10)

	auc_app_result = []
	pre_app_result = []
	rec_app_result = []
	f1_app_result = []

	auc_churn_result = []
	pre_churn_result = []
	rec_churn_result = []
	f1_churn_result = []

	auc_up_result = []
	pre_up_result = []
	rec_up_result = []
	f1_up_result = []

#Cross-validation
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

#appetency's log
		app_model.fit(X_train, app_Y_train)
		pred = app_model.predict(X_test)

		pre_app_auc = roc_auc_score(app_Y_test, pred)
		pre_app_pre, pre_app_rec, pre_app_f1, support = precision_recall_fscore_support(app_Y_test, pred, pos_label=-1, average='binary')

		auc_app_result.append(pre_app_auc)
		pre_app_result.append(pre_app_pre)
		rec_app_result.append(pre_app_rec)
		f1_app_result.append(pre_app_f1)

#churn's log
		churn_model.fit(X_train, churn_Y_train)
		pred = churn_model.predict(X_test)

		pre_churn_auc = roc_auc_score(churn_Y_test, pred)
		pre_churn_pre, pre_churn_rec, pre_churn_f1, _ = precision_recall_fscore_support(churn_Y_test, pred, average='binary')

		auc_churn_result.append(pre_churn_auc)
		pre_churn_result.append(pre_churn_pre)
		rec_churn_result.append(pre_churn_rec)
		f1_churn_result.append(pre_churn_f1)

#upselling's log
		up_model.fit(X_train, up_Y_train)
		pred = up_model.predict(X_test)

		pre_up_auc = roc_auc_score(up_Y_test, pred)
		pre_up_pre, pre_up_rec, pre_up_f1,_ = precision_recall_fscore_support(up_Y_test, pred, average='binary')

		auc_up_result.append(pre_up_auc)
		pre_up_result.append(pre_up_pre)
		rec_up_result.append(pre_up_rec)
		f1_up_result.append(pre_up_f1)

	et = time.clock()

	print auc_app_result
	print pre_app_result
	print rec_app_result
#Calculate average of each index
	auc_app = sum(auc_app_result) / idx
	auc_churn = sum(auc_churn_result) / idx
	auc_up = sum(auc_up_result) / idx

	pre_app = sum(pre_app_result) / idx
	pre_churn = sum(pre_churn_result) / idx
	pre_up = sum(pre_up_result) / idx

	rec_app = sum(rec_app_result) / idx
	rec_churn = sum(rec_churn_result) / idx
	rec_up = sum(rec_up_result) / idx

	f1_app = sum(f1_app_result) / idx
	f1_churn = sum(f1_churn_result) / idx
	f1_up = sum(f1_up_result) / idx

	print "\t\t\tAUC\t\tPrecise\t\tRecall\t\tF1"
	print "App\t\t%g\t\t%g\t\t%g\t\t%g" %(auc_app, pre_app, rec_app, f1_app)
	print "Chu\t\t%g\t\t%g\t\t%g\t\t%g" %(auc_churn, pre_churn, rec_churn, f1_churn)
	print "Up\t\t%g\t\t%g\t\t%g\t\t%g" %(auc_up, pre_up, rec_up, f1_up)

if __name__ == "__main__":
	X, app_Y, churn_Y, up_Y = get_data("../data/orange_aft_clean.csv")

	train_with_svm(X, app_Y, churn_Y, up_Y)
