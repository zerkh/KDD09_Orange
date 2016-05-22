import numpy as np
import pandas as pd

def variance_threshold(df, threshold):
	X = df.values
	var = np.var(X, axis=0)

	fs = var<threshold

	columns = df.columns

	for i in xrange(len(columns)):
		if fs[i] == True:
			df = df.drop(columns[i], 1)

	return df


if __name__ == "__main__":
	df = pd.read_csv("../data/orange_aft_clean.csv")

	features = df.loc[:, "Var1":"Var230"]

	fs_df = variance_threshold(features, 1e-4)

	tar_df = df.loc[:, "appetency":]
	df = pd.concat([fs_df, tar_df], axis=1)

	df.to_csv("orange_aft_fs.csv", index=False)
