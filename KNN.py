# KNN - Evaluation method 1 (iris dataset)
# input test sample (sl, sw, pl, pw)
# output - class prediction
# number of neighbors, K = 1, 3, 5, 7, 9 ...

import pandas as pd
from math import *

inputarray = []

# K = number of nearest neighbors for the test pattern
K1 = input('lower limit of range of k : ')
# input the test data
inputarray.append(input('first attribute : '))
inputarray.append(input('second attribute : '))
inputarray.append(input('third attribute : '))
inputarray.append(input('fourth attribute : '))
# load the data form csv file
location = r"dataset/iris.csv"
df_iris = pd.read_csv(location)
print("Data Frame Shape {}".format(df_iris.shape))
# print(df_iris.head(5))
# add a column called 'distance'
df_iris['distance'] = 0
# for each row in the dataframe, calculate the distance
for index, row in df_iris.iterrows():
    eucDist = sqrt(((float(inputarray[0]) - float(row['sepal_length'])) ** 2 +
                    (float(inputarray[1]) - float(row['sepal_width'])) ** 2 +
                    (float(inputarray[2]) - float(row['petal_length'])) ** 2 +
                    (float(inputarray[3]) - float(row['petal_width'])) ** 2))

    df_iris.loc[index, 'distance'] = eucDist

df_iris.sort_values('distance', axis=0, ascending=True, inplace=True)
# select the first K rows, into a new df
k = int(K1)
df_iris_k = df_iris.iloc[0:k].copy()
print("OUTPUT -- Nearest Neighbors")
print(df_iris_k['class'])
