import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.cross_validation import KFold
import time
from utils import get_data
import tensorflow as tf

from nn_model import NN_Model

"""
Train and validate with nn

Params: Attributes, targets
Return: None
"""
def train_with_nn(X, app_Y, churn_Y, up_Y):
	sess = tf.Session()

	print "Train and validate with nn:"
	app_model = NN_Model(sess)
	churn_model = NN_Model(sess)
	up_model = NN_Model(sess)

	kf = KFold(len(X), 10)

	idx = 1
	app_auc = 0.0
	churn_auc = 0.0
	up_auc = 0.0
	for train_idx, test_idx in kf:
		print "Fold %d:" %idx
		st = time.clock()
		idx += 1
		X_train = X[train_idx]
		X_test = X[test_idx]

		app_Y_train = app_Y[train_idx]
		app_Y_test = app_Y[test_idx]

		churn_Y_train = app_Y[train_idx]
		churn_Y_test = app_Y[test_idx]

		up_Y_train = app_Y[train_idx]
		up_Y_test = app_Y[test_idx]

		app_model.fit(sess, X_train, app_Y_train)
		pred = app_model.predict(sess, X_test)
		pre_app_auc = roc_auc_score(app_Y_test, pred)
		app_auc += pre_app_auc
		print "auc of appetency in this fold: %g" %(pre_app_auc)
		print "accuracy of appetency in this fold: %g" %(accuracy_score(app_Y_test, pred))

		churn_model.fit(sess, X_train, churn_Y_train)
		pred = churn_model.predict(sess, X_test)
		pre_churn_auc = roc_auc_score(churn_Y_test, pred)
		churn_auc += pre_churn_auc
		print "auc of churn in this fold: %g" %(pre_churn_auc)
		print "accuracy of churn in this fold: %g" %(accuracy_score(churn_Y_test, pred))

		up_model.fit(sess, X_train, up_Y_train)
		pred = up_model.predict(sess, X_test)
		pre_up_auc = roc_auc_score(up_Y_test, pred)
		up_auc += pre_up_auc
		print "auc of upselling in this fold: %g" %(pre_up_auc)
		print "accuracy of upselling in this fold: %g" %(accuracy_score(up_Y_test, pred))

		et = time.clock()
		print "Use %d s in fold %d" %(et-st, idx)

	app_auc /= 10
	churn_auc /= 10
	up_auc /= 10
	print "Average auc score:"
	print "appetency: %g" %app_auc
	print "churn: %g" %churn_auc
	print "upselling: %g" %up_auc

if __name__ == "__main__":
	X, app_Y, churn_Y, up_Y = get_data("../data/orange_aft_clean.csv")

	train_with_nn(X, app_Y, churn_Y, up_Y)
