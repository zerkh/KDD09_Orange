import pandas as pd
import numpy as np

"""
Get matrix of data from csv

Params: filename: str
		attr: str
		is_balance: boolean
Return: Attributes, target of appetency, churn and upselling
"""
def get_data(filename, attr=None, is_balance=True):
	df = pd.read_csv(filename)

	X = df.loc[:, "Var1":"Var230"]
	Y = df.loc[:, attr]

	if not is_balance:
		return X.values, Y.values
	else:
		pos_labels = Y[Y==1].index.values
		neg_labels = Y[Y==-1].index.values

		pos_size = len(pos_labels)
		neg_size = len(neg_labels)

		if pos_size < neg_size:
			neg_samples = np.random.choice(neg_labels, pos_size, replace=False)
			samples = np.concatenate([pos_labels, neg_samples])
		else:
			pos_samples = np.random.choice(pos_labels, neg_size, replace=False)
			samples = np.concatenate([pos_samples, neg_labels])

		shuffle_samples = np.random.shuffle(samples)
		X = X[shuffle_samples]
		Y = Y[shuffle_samples]

		return X.values, Y.values