import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score, accuracy_score, precision_recall_fscore_support, confusion_matrix
from sklearn.cross_validation import KFold
import time
import sys
from utils import get_data, normal_y_softmax
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.manifold import Isomap
import tensorflow as tf

from nn_vote_model import NN_Voting
from nn_model import NN_Model
from sklearn.svm import SVC

"""
Train and validate with given model

Params: Attributes, targets
Return: None
"""
def train_and_validation(sess, X, Y, model):
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

		Y_train = normal_y_softmax(Y_train)
		model.fit(sess, X_train, Y_train)
		pred = model.predict(sess, X_test)

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

def fine_tuning_step(n_estimators, learning_rate):
	sess = tf.Session(config=tf.ConfigProto(intra_op_parallelism_threads=5, inter_op_parallelism_threads=5))
	
	model = NN_Voting(sess, n_estimators=n_estimators, learning_rate=learning_rate)
	X, app_Y = get_data("../data/orange_aft_clean.csv", attr="appetency", is_balance=True)
	app_auc, pre, rec, f1 = train_and_validation(sess, X, app_Y, model)

	model = NN_Voting(sess, n_estimators=n_estimators, learning_rate=learning_rate)
	X, churn_Y = get_data("../data/orange_aft_clean.csv", attr="churn", is_balance=True)
	churn_auc, pre, rec, f1 = train_and_validation(sess, X, churn_Y, model)
	
	model = NN_Voting(sess, n_estimators=n_estimators, learning_rate=learning_rate)
	X, up_Y = get_data("../data/orange_aft_clean.csv", attr="upselling", is_balance=True)
	up_auc, pre, rec, f1 = train_and_validation(sess, X, up_Y, model)

	return app_auc, churn_auc, up_auc

def fine_tune(log_file):
	of = open(log_file, "w")

	max_app_auc = 0.0
	max_churn_auc = 0.0
	max_up_auc = 0.0

	app_param = []
	churn_param = []
	up_param = []

	for n_estimators in range(20, 30):
		for learning_rate_int in range(1,10):
			learning_rate = learning_rate_int/10.0
			app_auc, churn_auc, up_auc = fine_tuning_step(n_estimators, learning_rate)

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

	of.write("\t\t\tauc\t\tn_estimators\tlearning_rate\n")
	of.write("auc\t\t%g\t\t%d\t\t%g\n" %(max_app_auc, app_param[0], app_param[1]))
	of.write("churn\t\t%g\t\t%d\t\t%g\n" %(max_churn_auc, churn_param[0], churn_param[1]))
	of.write("up\t\t%g\t\t%d\t\t%g\n" %(max_up_auc, up_param[0], up_param[1]))

	of.close()

def test(log_file):
	of = open(log_file, "w")
	sess = tf.Session(config=tf.ConfigProto(intra_op_parallelism_threads=5, inter_op_parallelism_threads=5))

#	model = AdaBoostClassifier(base_estimator=SVC(10, probability=True), n_estimators=20, learning_rate=1)
#	model = AdaBoostClassifier(n_estimators=20, learning_rate=1)

	of.write("\t\t\tAUC\t\tPrecise\t\tRecall\t\tF1\n")

	model = NN_Voting(sess, n_estimators=5, learning_rate=0.001)
	X, app_Y = get_data("../data/orange_aft_clean.csv", attr="appetency", is_balance=True)
	auc, pre, rec, f1 = train_and_validation(sess, X, app_Y, model)
	of.write("App\t\t%g\t\t%g\t\t%g\t\t%g\n" %(auc, pre, rec, f1))

	model = NN_Voting(sess, n_estimators=5, learning_rate=0.001)
	X, churn_Y = get_data("../data/orange_aft_clean.csv", attr="churn", is_balance=True)
	auc, pre, rec, f1 = train_and_validation(sess, X, churn_Y, model)
	of.write("Churn\t\t%g\t\t%g\t\t%g\t\t%g\n" %(auc, pre, rec, f1))

	model = NN_Voting(sess, n_estimators=5, learning_rate=0.001)
	X, up_Y = get_data("../data/orange_aft_clean.csv", attr="upselling", is_balance=True)
	auc, pre, rec, f1 = train_and_validation(sess, X, up_Y, model)
	of.write("Up\t\t%g\t\t%g\t\t%g\t\t%g\n" %(auc, pre, rec, f1))

if __name__ == "__main__":
	if len(sys.argv) < 2:
		print "Usage: python base_models.py log_file"
		exit -1

	log_file = sys.argv[1]
	test(log_file)

#	fine_tune(log_file)
