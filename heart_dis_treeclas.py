import sklearn.preprocessing as skl
import sklearn.tree as skltree
import sklearn.model_selection as skmodev
import sklearn.metrics as metric
import numpy as np
import pandas as pd
np.set_printoptions(threshold=np.inf)

pandadat=pd.read_csv("C:/Users/ZBOOK 15 G5/Desktop/heartdis/framingham.csv", usecols=range(16))
pandadat=pandadat.fillna(pandadat.mean())
rownum=pandadat.shape[0]
inp=pandadat.iloc[0:int(rownum*0.6),range(15)].to_numpy()

outputs1=pandadat["TenYearCHD"].to_numpy()
outputs=outputs1[0:int(rownum*0.6)]

test_inp=pandadat.iloc[int(rownum*0.6):,range(15)].to_numpy()
test_outputs=outputs1[int(rownum*0.6):]

min_maxer=skl.MinMaxScaler()

scaled_inp=min_maxer.fit_transform(inp)
scaled_test_inp=min_maxer.fit_transform(test_inp)

tree=skltree.DecisionTreeClassifier()
tree.fit(scaled_inp, outputs)

predicts=tree.predict(scaled_test_inp)
cvscor=(skmodev.cross_val_score(tree, scaled_test_inp, test_outputs)).mean()
print(metric.accuracy_score(test_outputs, predicts))
print(cvscor)