import tensorflow as tf
import numpy as np

__author__ = "kh"

class NN_Model:
	def __init__(self, sess, layer_size=5, hidden_size=200, dropout_prob=0.7, learning_rate=0.01):
		self.layer_size = layer_size
		self.hidden_size = hidden_size
		self.dropout_prob = dropout_prob

		x = tf.placeholder(tf.float32, [None, 420])
		y = tf.placeholder(tf.float32, [None, 2])

		self.x = x
		self.y = y

		input_size = 420
		output_size = 2

		l2_loss = None
		layer = tf.nn.dropout(x, dropout_prob)
		if layer_size != 1:
			output_size = hidden_size

		for l in xrange(layer_size):
			if l == layer_size-1:
				#layer = tf.nn.dropout(layer, dropout_prob)
				output_size = 2
			with tf.name_scope("Layer" + str(l)):
				weight = tf.Variable(tf.truncated_normal([input_size, output_size]), name="weight%d" %(l), trainable=True)
				b = tf.Variable(tf.zeros([output_size]), name="b%d" %(l), trainable=True)

				if l2_loss == None:
					l2_loss = tf.nn.l2_loss(weight)
				else:
					l2_loss += tf.nn.l2_loss(weight)

				if l == layer_size-1:
					layer = tf.matmul(layer,weight)+b
				else:
					layer = tf.sigmoid(tf.matmul(layer,weight)+b)
			input_size = hidden_size

		output = layer
#		loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(output,y)+l2_loss)
		loss = tf.reduce_mean(tf.sqrt(tf.square(output-y)) + l2_loss)
		output = tf.nn.softmax(layer)

		self.loss = loss
		self.output = output

		optimizer = tf.train.AdamOptimizer(learning_rate)
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
			output_val = output_val[0]

			if output_val[0] > output_val[1]:
				output_val = 0
			else:
				output_val = 1

			outputs.append(output_val)
		return outputs
