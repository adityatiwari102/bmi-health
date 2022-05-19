#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler, StandardScaler, LabelEncoder

from sklearn.metrics import accuracy_score,classification_report, confusion_matrix

import warnings
warnings.filterwarnings('ignore')
import pickle


# In[2]:


df = pd.read_excel('dataGYM.xlsx')
df.head()


# In[3]:


df['Prediction'].value_counts()


# In[4]:


sns.countplot(x= df['Class'], palette='viridis')
plt.xticks(rotation= 45)


# In[5]:


df.Class.unique()


# In[6]:


df.info()


# In[7]:


df.Class.value_counts()


# In[8]:


df['Class'] = df['Class'].replace('EXtremely obese', 'Extremely obese')


# In[9]:


#encoding Prediction for feed data into the model
label_encoder = LabelEncoder()
df['Prediction'] = label_encoder.fit_transform(df['Prediction'])


# In[10]:


X = df.iloc[:, :4]
y = df.iloc[:, 5:]


# In[11]:


X.head()


# In[12]:


y.value_counts(ascending=True)


# 0 : Extremely Obese
# 1 : Healthy
# 2 : Obese
# 3 : Overweight
# 4 : Underweight

# In[13]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[14]:


rfc = RandomForestClassifier()


# In[15]:


rfc.fit(X_train, y_train)


# In[16]:


y_pred = rfc.predict(X_test)


# In[17]:


result = confusion_matrix(y_test, y_pred)

result1 = classification_report(y_test, y_pred)

result2 = accuracy_score(y_test, y_pred)

print('Confusion Matrix:',result)
print('Classification report:',result1)
print('Accuracy Score : ', result2)


# In[18]:



importances = rfc.feature_importances_
#
# Sort the feature importance in descending order
#
sorted_indices = np.argsort(importances)[::-1]


# In[19]:


plt.title('Feature Importance')
plt.bar(range(X_train.shape[1]), importances[sorted_indices], align='center')
plt.xticks(range(X_train.shape[1]), X_train.columns[sorted_indices], rotation=90)
plt.tight_layout()
plt.show()


# In[20]:


X = X.drop(['BMI'], axis = 1)
X.head()


# In[21]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[22]:


rfc = RandomForestClassifier(n_estimators=20, criterion='entropy')


# In[23]:


rfc.fit(X_train, y_train)


# In[24]:


y_pred = rfc.predict(X_test)


# In[25]:


result = confusion_matrix(y_test, y_pred)

result1 = classification_report(y_test, y_pred)

result2 = accuracy_score(y_test, y_pred)

print('Confusion Matrix:',result)
print('Classification report:',result1)
print('Accuracy Score : ', result2)


# In[26]:


#support vector classifier to improve the accuracy
svc = SVC()

svc.fit(X_train, y_train)


# In[27]:


y_pred = svc.predict(X_test)


# In[28]:


result = confusion_matrix(y_test, y_pred)

result1 = classification_report(y_test, y_pred)

result2 = accuracy_score(y_test, y_pred)

print('Confusion Matrix:',result)
print('Classification report:',result1)
print('Accuracy Score : ', result2)


# In[29]:


#scaling the X for using Standard Scaler

sc = StandardScaler()
sc.fit(X_train)
X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)

cols = ['Age', 'Height', 'weight']


X_train_std = pd.DataFrame(X_train_std, columns=cols)
X_test_std = pd.DataFrame(X_test_std, columns=cols)


# In[30]:


# defining parameter range 
param_grid = {'C': [0.1, 1, 10, 50, 75, 100],   
              'gamma': [1, 0.1, 0.01, 0.001, 0.0001], 
              'gamma':['scale', 'auto'],
              'kernel': ['linear', 'poly', 'rbf', 'sigmoid']}  
   
grid = GridSearchCV(SVC(), param_grid, refit = True, verbose = 3,n_jobs=-1) 
   
# fitting the model for grid search 
grid.fit(X_train_std, y_train) 
 
# print best parameter after tuning 
print(grid.best_params_) 
grid_predictions = grid.predict(X_test_std) 
   
# print classification report 
print(classification_report(y_test, grid_predictions))


# In[31]:


svc = SVC(C=110, gamma= 'auto', kernel='rbf', probability=True)


# In[32]:


svc.fit(X_train_std, y_train)


# In[33]:


y_pred = svc.predict(X_test_std)


# In[34]:


result = confusion_matrix(y_test, y_pred)

result1 = classification_report(y_test, y_pred)

result2 = accuracy_score(y_test, y_pred)

print('Confusion Matrix:',result)
print('Classification report:',result1)
print('Accuracy Score : ', result2)


# In[35]:


pickle.dump(svc, open('bmi_health_model.pkl', 'wb'))
model = pickle.load(open('bmi_health_model.pkl', 'rb'))


# In[36]:


print(svc.predict([[34, 5.7, 65]]))


# In[ ]:




