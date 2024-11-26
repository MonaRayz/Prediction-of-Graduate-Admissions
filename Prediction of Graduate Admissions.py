#!/usr/bin/env python
# coding: utf-8

# In[7]:


import pandas as pd

#Lading the dataset as pandas dataframe
df = pd.read_csv('Admission_predict.csv')
print(df.head(10))


# In[8]:


#standardizing the column names of the pandas dataframe
df.columns = df.columns.str.strip().str.replace(' ', '_').str.lower()
print(df.head(5))


# In[4]:


#creating the feature and target labels 
X = df.loc[:,'gre_score':'research']
y = df['chance_of_admit']>=0.8


# In[17]:


#fitting and predicting chance of admission using Decision Tree Classifier

from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix




x_train, x_test, y_train, y_test= train_test_split(X,y,random_state=0, test_size=0.2)
dt = DecisionTreeClassifier(max_depth=2, ccp_alpha=0.01,criterion='gini')


#training the model 

dt.fit(x_train, y_train)


#predicting using the model 

y_pred = dt.predict(x_test)

#scoring the model 
print(dt.score(x_test,y_test))
print(accuracy_score(y_test, y_pred))



# In[27]:


#graphical representation of the decison tree
import matplotlib.pyplot as plt
from sklearn import tree


tree.plot_tree(dt, feature_names = ['gre_score','toefl_score','university_rating','sop','lor','cgpa','research'],  
               max_depth=3, class_names = ['unlikely admit', 'likely admit'],
               label='root', filled=True)
print(tree.export_text(dt, feature_names = X.columns.tolist()))


# In[31]:


#fitting and predicting chance of admission using Decision Tree Regressor
from sklearn.tree import DecisionTreeRegressor
x_train, x_test, y_train, y_test = train_test_split(X,y, random_state=0, test_size=0.2)
dtr = DecisionTreeRegressor(max_depth=3, ccp_alpha=0.001)
dtr.fit(x_train, y_train)
y_pred = dtr.predict(x_test)
print(dtr.score(x_test, y_test))


# In[34]:


plt.figure(figsize=(10,10))
tree.plot_tree(dtr, feature_names = ['gre_score','toefl_score','university_rating','sop','lor','cgpa','research'],  
                filled=True);


# In[ ]:




