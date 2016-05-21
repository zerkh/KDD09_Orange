import tensorflow as tf
import numpy as np

import random
from nn_model import NN_Model
__author__ = "kh"

class NN_Voting:
	def __init__(self, sess, n_estimators, learning_rate):
		self.base_estimators = []
		for i in xrange(n_estimators):
			layer_size = random.randint(3,10)
			hidden_size = random.randint(200,1000)
			with tf.name_scope("model%d" %i):
				nn_model = NN_Model(sess, layer_size=layer_size,\
					hidden_size=hidden_size, learning_rate = learning_rate)
			self.base_estimators.append(nn_model)


	def fit(self, sess, X, Y, batch_size=128, max_iter=10, verbose=False):
		for i in xrange(len(self.base_estimators)):
			self.base_estimators[i].fit(sess, X, Y, batch_size, max_iter, verbose)

	def predict(self, sess, X):
		all_results = []
		final_result = []
		for i in xrange(len(self.base_estimators)):
			all_results.append(self.base_estimators[i].predict(sess, X))

		for i in xrange(len(X)):
			pos_num = 0
			neg_num = 0
			for j in xrange(len(self.base_estimators)):
				if all_results[j][i] = 1:
					pos_num += 1
				else:
					neg_num += 1
			if pos_num > neg_num:
				final_result.append(1)
			else:
				final_result.append(0)
		return final_result