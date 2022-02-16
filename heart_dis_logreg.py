import sklearn.preprocessing as skl
import sklearn.linear_model as sklin
import sklearn.model_selection as skmodev
import matplotlib.pyplot as plt
import sklearn.metrics as metric
import numpy as np
import pandas as pd
from plot_learning_curve import plot_learning_curve
np.set_printoptions(threshold=np.inf)
#inp=numlib.genfromtxt(open("C:/Users/ZBOOK 15 G5/Desktop/heartdis/framingham.csv", "r"), delimiter=",",skip_header=1, usecols=range(16))

pandadat=pd.read_csv("C:/Users/ZBOOK 15 G5/Desktop/heartdis/framingham.csv", usecols=range(16))
pandadat=pandadat.fillna(pandadat.mean())
parameters={'C':[1,10]}
rownum=pandadat.shape[0]
all_inputs=pandadat.iloc[:,range(15)].to_numpy()
all_outputs=outputs1=pandadat["TenYearCHD"].to_numpy()
inp=pandadat.iloc[0:int(rownum*0.6),range(15)].to_numpy()

outputs1=pandadat["TenYearCHD"].to_numpy()
outputs=outputs1[0:int(rownum*0.6)]

test_inp=pandadat.iloc[int(rownum*0.6):,range(15)].to_numpy()
test_output=outputs1[int(rownum*0.6):]

min_maxer=skl.MinMaxScaler()

scaled_inp=min_maxer.fit_transform(inp)
scaled_test_inp=min_maxer.fit_transform(test_inp)
scaled_all_inp=min_maxer.fit_transform(all_inputs)

logistic_reg=sklin.LogisticRegression()
grid=skmodev.GridSearchCV(logistic_reg, parameters)
grid.fit(scaled_inp,outputs)
print(grid.get_params())
#scor=skmodev.cross_validate(logistic_reg, scaled_inp, outputs, return_estimator=True)
#print(scor)
plot_learning_curve(grid, 'Learning Curves', scaled_all_inp, all_outputs)
pred=grid.predict(scaled_test_inp)
print(metric.accuracy_score(test_output, pred))
print(metric.f1_score(test_output, pred))
print(metric.recall_score(test_output, pred))
print(metric.precision_score(test_output, pred))
print(skmodev.cross_val_score(grid, scaled_inp, outputs).mean())
#pred_scor=skmodev.cross_val_predict(logistic_reg,test_inp)
