# coding: utf-8

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold 
from sklearn.model_selection import cross_val_score
from xgboost import XGBClassifier

df = pd.read_csv('./train.csv')
y = df.label
x = df.drop('label',axis = 1)
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.3,random_state=33)
clf = XGBClassifier(nthread = 4, max_depth = 10, learning_rate=0.1, n_estimators = 20, objective='multi:softmax', num_class = 10, seed=20)
clf = clf.fit(x_train,y_train)

kf = KFold(n_splits=5, shuffle=True)
scores = cross_val_score(clf, x, y, scoring='f1_macro', cv = kf)  

df_test = pd.read_csv('./test.csv')
pre_y = clf.predict(df_test)

ids = list(range(1,len(pre_y)+1))
save_df = pd.DataFrame({
    'ImageId' : ids,
    'Label' : pre_y
})
save_df.to_csv('./sub_4.csv',index=False)


# In[7]:




