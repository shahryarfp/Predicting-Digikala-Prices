#!/usr/bin/env python
# coding: utf-8

# # Shahryar Namdari

# ## Access to Competition
# https://quera.org/problemset/138168/

# In[76]:


import numpy as np
import math
import pandas as pd
import matplotlib.pyplot as plt
import json
from tqdm import notebook
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn import neighbors
import ast
from collections import OrderedDict
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import mean_squared_error
from sklearn import neighbors


# ## Reading Data

# In[2]:


train = pd.read_csv('./data/train.csv')
train.dropna(axis=0, inplace=True)
final_test = pd.read_csv('./data/test.csv')
print("train data length:", len(train))
print("final_test data length:", len(final_test))


# ## Displaying Data

# In[3]:


train


# In[4]:


final_test


# ## Sample Preprocess

# ### Before:

# In[5]:


sample = train.values[1,:]
sample


# ### After:

# In[6]:


sample = sample[1]
sample = sample.replace('\\n','')
sample = sample.replace('\\r','')
sample = sample.replace('\\u200c','')
sample = sample.replace('\\\\/','')
sample


# ### Sample Convert to Dictionary:

# In[7]:


sample = train.values[22,:][1]
res = ast.literal_eval(sample)
res.keys()


# ## Preprocess

# In[8]:


list_train_dict = []
for i in notebook.tqdm(range(len(train))):
    sample = train.values[i,1]
    sample = sample.replace('\\n','')
    sample = sample.replace('\\r','')
    sample = sample.replace('\\u200c','')
    sample = sample.replace('\\\\/','')
    list_train_dict.append(ast.literal_eval(sample))


# In[9]:


list_test_dict = []
for i in notebook.tqdm(range(len(final_test))):
    sample = final_test.values[i,1]
    sample = sample.replace('\\n','')
    sample = sample.replace('\\r','')
    sample = sample.replace('\\u200c','')
    sample = sample.replace('\\\\/','')
    list_test_dict.append(ast.literal_eval(sample))


# In[10]:


def count_keys_num(list_t_dict):
    keys_count_dict = OrderedDict()
    for i in range(len(list_t_dict)):
        temp_keys = list(list_t_dict[i].keys())
        for key in temp_keys:
            if key not in keys_count_dict.keys():
                keys_count_dict[key] = 1
            else:
                keys_count_dict[key] += 1
    # sorting
    keys_count_dict = dict(OrderedDict(sorted(keys_count_dict.items(), key=lambda t: t[1])))
    keys_count_dict = OrderedDict(reversed(list(keys_count_dict.items())))
    keys_count_list = []
    for key in keys_count_dict.keys():
        keys_count_list.append([key, keys_count_dict[key]])
    return keys_count_list

def count_keys_type(list_t_dict, feature):
    count_type = 0
    types = []
    for i in range(len(list_t_dict)):
        type_ = list_t_dict[i][feature]
        if type_ not in types:
            types.append(type_)
            count_type += 1
    print(feature,":", count_type)
    return types


# In[11]:


keys_count_list_train = count_keys_num(list_train_dict)
keys_count_list_test = count_keys_num(list_test_dict)


# In[107]:


# a = count_keys_type(list_train_dict, 'دسته بندی')
for i in range(20):
    print(keys_count_list_train[i])


# ## Regression Models

# In[12]:


def find_feature(feature, list_dict):
    x = []
    for i in range(len(list_dict)):
        if feature in list_dict[i].keys():
            x.append(list_dict[i][feature])
    return x

def MAPE(y_test, pred):
    y_test_temp = []
    pred_temp = []
    l1 = np.array(y_test)
    l2 = np.array(pred)
    for i in range(len(y_test)):
        if l1[i] != 0:
            y_test_temp.append(l1[i])
            pred_temp.append(l2[i])
    y_test_temp, pred_temp = np.array(y_test_temp), np.array(pred_temp)
    mape = np.mean(np.abs((y_test_temp - pred_temp) / y_test_temp))
    return mape


# ### Train

# In[99]:


feature_1 = "برند"
feature_2 = "دسته بندی"

y = train["price"]

#normalizing
print('max y: ', max(y))
print('min y: ', min(y)) # ==> 0
train_max_price = max(y)
y = y/max(y)

x1 = find_feature(feature_1, list_train_dict)
x2 = find_feature(feature_2, list_train_dict)
x = pd.DataFrame(list(zip(x1, x2)), columns =['برند', 'دسته بندی'])
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, shuffle=True)

ohe = OneHotEncoder(handle_unknown = 'ignore')
ohe.fit(x_train)
x_train_ohe = ohe.transform(x_train).toarray()

# model = LinearRegression()
# model = DecisionTreeRegressor()
# model = RandomForestRegressor(n_estimators = 50, random_state = 0, bootstrap=True, max_samples=0.1)
model = neighbors.KNeighborsRegressor(n_neighbors = 5)
model.fit(x_train_ohe, y_train)


# ### Test

# In[100]:


x_test_ohe = ohe.transform(x_test).toarray()
y_pred = model.predict(x_test_ohe)


# ### Evaluation

# In[101]:


print('MAPE for rows in test_data with nonzero value ==> ', MAPE(y_test, y_pred))
print('MSE: ==> ', mean_squared_error(y_test,y_pred))
print('Model score ==> ', model.score(x_test_ohe, y_test))


# ### Calculate price column for test.csv

# In[102]:


x1 = find_feature(feature_1, list_test_dict)
x2 = find_feature(feature_2, list_test_dict)
x_final_test = pd.DataFrame(list(zip(x1, x2)), columns =['برند', 'دسته بندی'])
x_final_test_ohe = ohe.transform(x_final_test).toarray()
y_final_test = model.predict(x_final_test_ohe) * train_max_price


# ### Save results in output.csv

# In[103]:


titles = ['id', 'price']
rows = [titles]
for i in range(len(final_test)):
    rows.append([str(final_test['id'][i]), str(y_final_test[i])])

np.savetxt("output.csv",
           rows,
           delimiter =",",
           fmt ='% s')

