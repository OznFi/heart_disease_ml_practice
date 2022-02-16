import sklearn.preprocessing as skl
import sklearn.ensemble as sklensemb
import sklearn.model_selection as skmodev
import matplotlib.pyplot as plt
import sklearn.metrics as metric
from plot_learning_curve import plot_learning_curve
import numpy as np
import pandas as pd

pandadat=pd.read_csv("C:/Users/ZBOOK 15 G5/Desktop/heartdis/framingham.csv", usecols=range(16))
pandadat=pandadat.fillna(pandadat.mean())
rownum=pandadat.shape[0]
parameters={'n_estimators':[50,200]}
all_inputs=pandadat.iloc[:,range(15)].to_numpy()
all_outputs=outputs1=pandadat["TenYearCHD"].to_numpy()
inp=pandadat.iloc[0:int(rownum*0.6),range(15)].to_numpy()

outputs1=pandadat["TenYearCHD"].to_numpy()
outputs=outputs1[0:int(rownum*0.6)]

test_inp=pandadat.iloc[int(rownum*0.6):,range(15)].to_numpy()
test_outputs=outputs1[int(rownum*0.6):]

min_maxer=skl.MinMaxScaler()

scaled_inp=min_maxer.fit_transform(inp)
scaled_test_inp=min_maxer.fit_transform(test_inp)
scaled_all_inp=min_maxer.fit_transform(all_inputs)

forester=sklensemb.RandomForestClassifier()
grid=skmodev.GridSearchCV(forester, parameters)
grid.fit(scaled_inp, outputs)
#print(grid.get_params())

cvscor=skmodev.cross_val_score(grid, scaled_test_inp, test_outputs).mean()
pred=grid.predict_proba(scaled_test_inp)
propred=(pred[:,1]>=0.5).astype('int')
scor=metric.accuracy_score(test_outputs, propred)
f1=metric.f1_score(test_outputs, propred)
plot_learning_curve(grid, 'Learning Curves', scaled_all_inp, all_outputs)
print(scor)
print(metric.recall_score(test_outputs, propred))
print(cvscor)
print(f1)