
# coding: utf-8

# In[34]:


"""
2018_03_07

version_0.0.1

"""

# import package

# -----------manipulate data package-----------
import pandas as pd
import numpy as np

# -----------connect to mysql package-----------
# import pymysql
# import sqlalchemy
# from sqlalchemy import create_engine

# -----------machine learning package-----------
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.model_selection import learning_curve
from xgboost import XGBClassifier,DMatrix
from sklearn.externals import joblib # save model
# -----------ploting package-----------
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline # plot in jupyter')
# set random seed to replicated results
import random
random.seed(9001) 


# In[ ]:


# # connect database setting

# #寫入資料庫
# user = 'user'
# pwd = 'password'
# port = 3306

# engine_set_up = 'mysql+pymysql://{}:{}@127.0.0.1:{}/dbname?charset=utf8'
# in_conn = create_engine(engine_set_up.format(user,pwd,port))
# read_db_data = pd.read_sql('select * from table',con=in_conn)
# read_db_data.to_sql('table',in_conn,if_exists='',index=False) #寫入資料庫

# read_csv or excel
# csv_data = pd.read_csv('.csv')
# xlsx_data = pd.read_excel('.xlsx')


# In[ ]:


## ------delect columns------
# del data['col']

## ------rearrange columns-------
## End to front
# cols = df.columns.tolist()
# cols = cols[-1:] + cols[:-1]
# dfc = df[cols].copy() # make a copy of rearranged df


# In[ ]:


## ------plotting-------

# plt.figure(figsize=(10,7)) # make plot larger
# plt.plot(x=,y=,color='blue',marker = 'o',markersize=5,label='training accuracy')
# plt.grid() # make a grid
# plt.xlabel('Number of training samples')
# plt.ylabel('Accuracy')
# plt.legend(loc='lower right')
# plt.show() # with or w/o %matplotlib inline


# In[ ]:


## ------Normalization-------
# std_scler = StandardScaler()
# used_col = ['col1','col2','col3']
# X_std = std_scler.fit_transform(data[used_col])
# test_std = std_scler.transform(test_data[used_col])
# joblib.dump(std_scler,'std_scale.pkl') # dump normalized model


# In[ ]:


## ------train test split-------

# X = data.iloc[:-1,:]
# Y = data.iloc[-1:,:]

# xx,tx,yy,ty = train_test_split(X[used_col],Y,test_size=0.2,random_state=1)


# In[ ]:


## ------classification-------
## random forest
# rf = RandomForestClassifier(random_state=6,class_weight={0: 0.07, 1: 0.93}) # random forest with imbalanced data
# rf.fit(xx,yy)

# print('train_score',round(rf.score(xx,yy),4))
# print('test_score',round(rf.score(tx,ty),4))

# np.unique(rf.predict(xx)) # check how many label to be predicted

# auc = roc_auc_score(yy, rf.predict(xx))
# print(auc) #  check area under roc curve, as close to score is good


# In[ ]:


## Extreme gradient boosting

# xgb = XGBClassifier(random_state=6,learning_rate=0.017) # feeling better lr start with 0.017

# xgb.fit(X_train_std,y)

# print('train_score',round(xgb.score(X_train_std,y),4))
# print('test_score',round(xgb.score(X_test_std,ty),4))
# train_y_pred = xgb.predict(X_train_std)
# print(np.unique(train_y_pred))# check how many label to be predicted
# auc = roc_auc_score(y, train_y_pred)
# print(auc) #  check area under roc curve, as close to score is good


# In[ ]:


# # -------use SBS algorithm to find the best feature combination -------
# from SBS import * # writting by myself but copy by book:python ml 
# from sklearn.neighbors import KNeighborsClassifier

# knn = KNeighborsClassifier(n_neighbors=2)
# sbs = SBS(rf,k_features=1)
# sbs.fit(X_train_std,y)

# k_feat = [len(k) for k in sbs.subsets_]
# plt.grid(b=True)
# plt.plot(k_feat, sbs.scores_, marker='o')


# peak_on_the_plot = 22
# feature_list = list(sbs.subsets_[peak_on_the_plot])
# print(len(feature_list)) 

# data.iloc[:,:-1].columns[feature_list] # assume last column is label


# In[ ]:


# # -------saved needed columns in txt for future model use-------
# need_col = data.iloc[:,:-1].columns[feature_list] # assume last column is label
# np.savetxt('need_col.txt', need_col.values,fmt='%s,', delimiter=',') 
# # sort_importance feature
# feature_import_sort = data.iloc[:,:-1].columns[feature_list][rf.feature_importances_.argsort()[::-1]]
# np.savetxt('need_col_feature_import_sort.txt', feature_import_sort.values,fmt='%s,', delimiter=',') 

