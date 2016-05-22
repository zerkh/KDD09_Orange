import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score, accuracy_score, precision_recall_fscore_support, confusion_matrix
from sklearn.cross_validation import KFold
import time
import sys
from utils import get_data
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.manifold import Isomap

from sklearn.naive_bayes import GaussianNB

"""
Train and validate with given model

Params: Attributes, targets
Return: None
"""
def train_and_validation(X, Y, model):
	kf = KFold(len(X), 10)

#	im = Isomap(n_neighbors=10, n_components=100)
#	X = im.fit_transform(X)

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

def fine_tuning_step(n_neighbors):
	model = NearestNeighbors(n_neighbors=n_neighbors)
	X, app_Y = get_data("../data/orange_aft_clean.csv", attr="appetency", is_balance=True)
	app_auc, pre, rec, f1 = train_and_validation(X, app_Y, model)

	model = NearestNeighbors(n_neighbors=n_neighbors)
	X, churn_Y = get_data("../data/orange_aft_clean.csv", attr="churn", is_balance=True)
	churn_auc, pre, rec, f1 = train_and_validation(X, churn_Y, model)
	

	model = NearestNeighbors(n_neighbors=n_neighbors)
	X, up_Y = get_data("../data/orange_aft_clean.csv", attr="upselling", is_balance=True)
	up_auc, pre, rec, f1 = train_and_validation(X, up_Y, model)

	return app_auc, churn_auc, up_auc

def fine_tune(log_file):
	of = open(log_file, "w")

	max_app_auc = 0.0
	max_churn_auc = 0.0
	max_up_auc = 0.0

	app_param = []
	churn_param = []
	up_param = []

	for n_neighbors in range(5, 20):
		learning_rate = learning_rate_int/10.0
		app_auc, churn_auc, up_auc = fine_tuning_step(n_neighbors)

		if app_auc > max_app_auc:
			max_app_auc = app_auc
			app_param = [n_estimators, learning_rate]
		if churn_auc > max_churn_auc:
			max_churn_auc = churn_auc
			churn_param = [n_estimators, learning_rate]
		if up_auc > max_up_auc:
			max_up_auc = up_auc
			up_param = [n_estimators, learning_rate]

	result = {"app":(max_app_auc,app_param), "churn":(max_churn_auc, churn_param), "up":(max_churn_auc, churn_param)}

	of.write("\t\t\tauc\t\tn_neighbors\n")
	of.write("auc\t\t%g\t\t%d\n" %(max_app_auc, app_param[0]))
	of.write("churn\t\t%g\t\t%d\n" %(max_churn_auc, churn_param[0]))
	of.write("up\t\t%g\t\t%d\n" %(max_up_auc, up_param[0]))

	of.close()

def test(log_file):
	of = open(log_file, "w")

	of.write("\t\t\tAUC\t\tPrecise\t\tRecall\t\tF1\n")

	model = GaussianNB()
	X, app_Y = get_data("../data/orange_aft_clean.csv", attr="appetency", is_balance=True)
	auc, pre, rec, f1 = train_and_validation(X, app_Y, model)
	of.write("App\t\t%g\t\t%g\t\t%g\t\t%g\n" %(auc, pre, rec, f1))

	model = GaussianNB()
	X, churn_Y = get_data("../data/orange_aft_clean.csv", attr="churn", is_balance=True)
	auc, pre, rec, f1 = train_and_validation(X, churn_Y, model)
	of.write("Churn\t\t%g\t\t%g\t\t%g\t\t%g\n" %(auc, pre, rec, f1))

	model = GaussianNB()
	X, up_Y = get_data("../data/orange_aft_clean.csv", attr="upselling", is_balance=True)
	auc, pre, rec, f1 = train_and_validation(X, up_Y, model)
	of.write("Up\t\t%g\t\t%g\t\t%g\t\t%g\n" %(auc, pre, rec, f1))

if __name__ == "__main__":
	if len(sys.argv) < 2:
		print "Usage: python base_models.py log_file"
		exit -1

	log_file = sys.argv[1]
	test(log_file)

#	fine_tune(log_file)