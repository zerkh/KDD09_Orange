import pandas as pd
import numpy as np

"""
Get matrix of data from csv

Params: filename
Return: Attributes, target of appetency, churn and upselling
"""
def get_data(filename):
	df = pd.read_csv(filename)

	X = df.loc[:, "Var1":"Var230"].values
	app_Y = df.loc[:, "appetency"].values
	churn_Y = df.loc[:, "churn"].values
	up_Y = df.loc[:, "upselling"].values

	return X, app_Y, churn_Y, up_Y
