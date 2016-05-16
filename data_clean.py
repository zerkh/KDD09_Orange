import pandas as pd
import numpy as np
import time

__author__ = "kh"

#process numerical attribute
def clean_num(df):
	attrs = df.columns
	for attr in attrs:
		df_attr = df[attr]
		min_val = df_attr.min()
		max_val = df_attr.max()
		df_attr = (df_attr - min_val) / (max_val-min_val)
		mean_val = df_attr.mean()
		df_nadix = df_attr.isnull()
		df_idc = pd.DataFrame(np.zeros(df_nadix.size, dtype="int64"), columns=[attr+"_idc"])
		df_idc[df_nadix] = 1

		if df_attr.count() == 0:
			df_attr = df_attr.fillna(0.0)
		df_attr = df_attr.fillna(mean_val)
		df[attr] = df_attr
		df = pd.concat([df, df_idc], axis=1)
	return df

#process categorical attribute
def clean_cate(df):
	attrs = df.columns
	for attr in attrs:
		df_attr = df[attr]
		df_attr = df_attr.fillna("NaN")
		cate = df_attr.unique()
		m_cate = {}
		idx = 0
		for c in cate:
			m_cate[c] = idx
			idx += 1
		df_attr = df_attr.map(m_cate)
		df[attr] = df_attr

		with open("cat_log/" + attr + ".log", "w") as fout:
			for key in m_cate:
				fout.write("%d %s\n" %(m_cate[key], key))

	return df

if __name__ == "__main__":
	df = pd.read_csv("orange_with_labels.csv")

	print "Process numerical attribute...."
	st = time.clock()
	num_df = clean_num(df.loc[:,"Var1":"Var190"])
	et = time.clock()
	print "Complete! Use %d s" %(et-st)

	print "Process categorical attribute...."
	st = time.clock()
	cat_df = clean_cate(df.loc[:,"Var191":"Var230"])
	print "Complete! Use %d s" %(et-st)

	tar_df = df.loc[:, "appetency":]
	df = pd.concat([num_df, cat_df, tar_df], axis=1)

	df.to_csv("orange_aft_clean.csv", index=False)
