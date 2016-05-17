import tensorflow as tf
import numpy as np

__author__ = "kh"

class NN_Model:
	def __init__(self, sess, layer_size=5, hidden_size=200, dropout_prob=0.7, learning_rate=0.01):
		self.layer_size = layer_size
		self.hidden_size = hidden_size
		self.dropout_prob = dropout_prob

		x = tf.placeholder(tf.float32, [None, 420])
		y = tf.placeholder(tf.float32, [None, ])

		self.x = x
		self.y = y

		input_size = 420
		output_size = 1

		layer = x
		if layer_size != 1:
			output_size = hidden_size

		for l in xrange(layer_size):
			if l == layer_size-1:
				layer = tf.nn.dropout(layer, dropout_prob)
				output_size = 1
			with tf.name_scope("Layer" + str(l)):
				weight = tf.Variable(tf.truncated_normal([input_size, output_size]), name="weight%d" %(l))
				b = tf.Variable(tf.zeros([output_size]), name="b%d" %(l))

				layer = tf.tanh(tf.matmul(layer,weight)+b)
			input_size = hidden_size

		output = tf.tanh(layer)
		loss = tf.reduce_mean(tf.square(output-y))

		self.loss = loss
		self.output = output

		optimizer = tf.train.GradientDescentOptimizer(learning_rate)
		self.train_step = optimizer.minimize(loss)

		sess.run(tf.initialize_all_variables())

	def fit(self, sess, X, Y, batch_size=128, max_iter=10, verbose=False):
		batch_len = len(X) // batch_size + 1

		for iter in xrange(max_iter):
			loss_val = 0.0
			if verbose:
				print "Iter %d: " %(iter)
			for b in xrange(batch_len):
				if b == batch_len-1:
					X_ = X[b*batch_size:, :]
					Y_ = Y[b*batch_size:]
				else:
					X_ = X[b*batch_size:(b+1)*batch_size, :]
					Y_ = Y[b*batch_size:(b+1)*batch_size]

				_, t_loss_val = sess.run([self.train_step, self.loss], feed_dict={self.x:X_, self.y:Y_})
				loss_val += t_loss_val
			loss_val /= batch_len

			if verbose:
				print "loss: %g" %(loss_val)

	def predict(self, sess, X):
		outputs = []
		for b in xrange(len(X)):
			X_ =X[b].reshape(1, len(X[b]))
			output_val = sess.run(self.output, feed_dict={self.x:X_})
			output_val = output_val[0,0]
			if output_val > 0:
				output_val = 1
			else:
				output_val = -1
			outputs.append(output_val)
		return outputs
