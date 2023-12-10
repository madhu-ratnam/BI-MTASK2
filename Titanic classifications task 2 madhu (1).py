#!/usr/bin/env python
# coding: utf-8

# ## BHARAT INTERNSHIP
# # 
# #   ## NAME- SEELAM MADHU RATNAM
# #   
# #   ## TASK 2-TITANIC CLASSIFICATION
# #   - In this we predicts if a passenger will survive on the titanic or not
# #   
# 

# # **1. Importing Data and Packages**

# In[1]:


import os
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier,ExtraTreesClassifier
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import RFE
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, classification_report, precision_score, recall_score 
from sklearn.metrics import confusion_matrix, precision_recall_curve, roc_curve, auc, log_loss


# In[2]:


train_data = pd.read_csv('/kaggle/input/titanic/train.csv')
test_data = pd.read_csv('/kaggle/input/titanic/test.csv')

        


# In[3]:


train_data.head()
print(train_data.shape[0])


# In[4]:


test_data.head()


# # **2. Data Exploring and Manipulation**

# 

# In[5]:


train_data.isnull().sum()


# 
# # 2.1.1 Age Missing Values

# In[6]:


age_plot = train_data['Age'].hist(bins=15, density=True, stacked=True, color='teal', alpha=0.6)
train_data['Age'].plot(kind='density', color='teal')
age_plot.set(xlabel='Age')
plt.xlim(-10,85)
plt.show()


# We will put median of the age's of missing age inputs!!

# In[7]:


print('The median of "Age" is %.2f' %(train_data["Age"].median(skipna=True)))


# # 2.1.2 Cabin Missing Values

# In[8]:


print('Percent of missing "Cabin" records is %.2f%%' %((train_data['Cabin'].isnull().sum()/train_data.shape[0])*100))


# # 2.1.3 Embarked Missing Values

# In[9]:


# percent of missing "Embarked" 
print('Percent of missing "Embarked" records is %.2f%%' %((train_data['Embarked'].isnull().sum()/train_data.shape[0])*100))
print('Boarded passengers grouped by port of embarkation (C = Cherbourg, Q = Queenstown, S = Southampton):')
print(train_data['Embarked'].value_counts())
sns.countplot(x='Embarked', data=train_data, palette='Set2')
plt.show()


# # 2.2 Data Manipulation

# In[10]:


train_data_manip = train_data.copy()
train_data_manip["Age"].fillna(train_data["Age"].median(skipna=True), inplace=True)
train_data_manip["Embarked"].fillna(train_data['Embarked'].value_counts().idxmax(), inplace=True)
train_data_manip.drop('Cabin', axis=1, inplace=True)
train_data_manip.isnull().sum()


# Visualizing adjusted and raw data.

# In[11]:


plt.figure(figsize=(15,8))
ax = train_data["Age"].hist(bins=15, density=True, stacked=True, color='teal', alpha=0.6)
train_data["Age"].plot(kind='density', color='teal')
ax = train_data_manip["Age"].hist(bins=15, density=True, stacked=True, color='orange', alpha=0.5)
train_data_manip["Age"].plot(kind='density', color='orange')
ax.legend(['Raw Age', 'Adjusted Age'])
ax.set(xlabel='Age')
plt.xlim(-10,85)
plt.show()


# # 2.3. Additional Variables

# In[12]:


train_data_manip['TravelAlone']=np.where((train_data_manip["SibSp"]+train_data_manip["Parch"])>0, 0, 1)
train_data_manip.drop('SibSp', axis=1, inplace=True)
train_data_manip.drop('Parch', axis=1, inplace=True)


# In[13]:


training=pd.get_dummies(train_data_manip, columns=["Pclass","Embarked","Sex"])
training.drop('Sex_female', axis=1, inplace=True)
training.drop('PassengerId', axis=1, inplace=True)
training.drop('Name', axis=1, inplace=True)
training.drop('Ticket', axis=1, inplace=True)

final_train = training
final_train.head()


# We will aplly same changes to the test data. 

# In[14]:


test_data.isnull().sum()
test_data_manip = test_data.copy()
test_data_manip["Age"].fillna(train_data["Age"].median(skipna=True), inplace=True)
test_data_manip["Fare"].fillna(train_data["Fare"].median(skipna=True), inplace=True)
test_data_manip.drop('Cabin', axis=1, inplace=True)

test_data_manip['TravelAlone']=np.where((test_data_manip["SibSp"]+test_data_manip["Parch"])>0, 0, 1)

test_data_manip.drop('SibSp', axis=1, inplace=True)
test_data_manip.drop('Parch', axis=1, inplace=True)

testing = pd.get_dummies(test_data_manip, columns=["Pclass","Embarked","Sex"])
testing.drop('Sex_female', axis=1, inplace=True)
testing.drop('PassengerId', axis=1, inplace=True)
testing.drop('Name', axis=1, inplace=True)
testing.drop('Ticket', axis=1, inplace=True)

final_test = testing
final_test.head()


# In[15]:


final_train['IsMinor']=np.where(final_train['Age']<=16, 1, 0)

final_test['IsMinor']=np.where(final_test['Age']<=16, 1, 0)


# # 3. Logistic Regression Model and Prediction 

# 

# # 3.1 Feature Selection

# In[16]:


from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import RFE

cols = ["Age","Fare","TravelAlone","Pclass_1","Pclass_2","Embarked_C","Embarked_S","Sex_male","IsMinor"] 
X = final_train[cols]
y = final_train['Survived']
# Build a logreg and compute the feature importances
model = LogisticRegression()
# create the RFE model and select 8 attributes
rfe = RFE(model)
rfe = rfe.fit(X, y)
# summarize the selection of the attributes
print('Selected features: %s' % list(X.columns[rfe.support_]))


# In[17]:


from sklearn.feature_selection import RFECV
# Create the RFE object and compute a cross-validated score.
# The "accuracy" scoring is proportional to the number of correct classifications
rfecv = RFECV(estimator=LogisticRegression(), step=1, cv=10, scoring='accuracy')
rfecv.fit(X, y)

print("Optimal number of features: %d" % rfecv.n_features_)
print('Selected features: %s' % list(X.columns[rfecv.support_]))

# Plot number of features VS. cross-validation scores
plt.figure(figsize=(10,6))
plt.xlabel("Number of features selected")
plt.ylabel("Cross validation score (nb of correct classifications)")
plt.plot(range(1, len(rfecv.grid_scores_) + 1), rfecv.grid_scores_)
plt.show()


# In[18]:


Selected_features = ['Age', 'TravelAlone', 'Pclass_1', 'Pclass_2', 'Embarked_C', 
                     'Embarked_S', 'Sex_male', 'IsMinor']
X = final_train[Selected_features]

plt.subplots(figsize=(8, 5))
sns.heatmap(X.corr(), annot=True, cmap="RdYlGn")
plt.show()


# # 3.2 Model Training Based on train/test Split 

# In[19]:


X = final_train[Selected_features]
y = final_train['Survived']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2)

logreg = LogisticRegression()
logreg.fit(X_train, y_train)
y_pred = logreg.predict(X_test)
y_pred_proba = logreg.predict_proba(X_test)[:, 1]
[fpr, tpr, thr] = roc_curve(y_test, y_pred_proba)
print('Train/Test split results:')
print(logreg.__class__.__name__+" accuracy is %2.3f" % accuracy_score(y_test, y_pred))
print(logreg.__class__.__name__+" log_loss is %2.3f" % log_loss(y_test, y_pred_proba))
print(logreg.__class__.__name__+" auc is %2.3f" % auc(fpr, tpr))


# # **4. Model Evaluation and Analysis**
# 
# # 4.1 Receiver Operating Characteristics (ROC Curve)

# In[20]:


#Import roc_curve, auc
from sklearn.metrics import roc_curve, auc

y_train_score = logreg.decision_function(X_train)

# Calculate the fpr, tpr, and thresholds for the training set
train_fpr, train_tpr, thresholds = roc_curve(y_train, y_train_score)

#Calculate the probability scores of each point in the test set
y_test_score = logreg.decision_function(X_test)

#Calculate the fpr, tpr, and thresholds for the test set
test_fpr, test_tpr, test_thresholds = roc_curve(y_test, y_test_score)


#Plotting ROC curve
idx = np.min(np.where(tpr > 0.95))
plt.figure()
plt.plot(fpr, tpr, color='coral', label='ROC curve (area = %0.3f)' % auc(fpr, tpr))
plt.plot([0, 1], [0, 1], 'k--')
plt.plot([0,fpr[idx]], [tpr[idx],tpr[idx]], 'k--', color='blue')
plt.plot([fpr[idx],fpr[idx]], [0,tpr[idx]], 'k--', color='blue')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate (1 - specificity)', fontsize=14)
plt.ylabel('True Positive Rate (recall)', fontsize=14)
plt.title('Receiver operating characteristic (ROC) curve')
plt.legend(loc="lower right")
plt.show()


# # **4.2 K-fold Cross-validation**

# In[21]:


from sklearn.model_selection import cross_validate

scoring = {'accuracy': 'accuracy', 'log_loss': 'neg_log_loss', 'auc': 'roc_auc'}

modelCV = LogisticRegression()

results = cross_validate(modelCV, X, y, cv=10, scoring=list(scoring.values()), 
                         return_train_score=False)

print('K-fold cross-validation results:')
for sc in range(len(scoring)):
    print(modelCV.__class__.__name__+" average %s: %.3f (+/-%.3f)" % (list(scoring.keys())[sc], -results['test_%s' % list(scoring.values())[sc]].mean()
                               if list(scoring.values())[sc]=='neg_log_loss' 
                               else results['test_%s' % list(scoring.values())[sc]].mean(), 
                               results['test_%s' % list(scoring.values())[sc]].std()))


# # **5.Submission**

# In[22]:


final_test['Survived'] = logreg.predict(final_test[Selected_features])
final_test['PassengerId'] = test_data['PassengerId']

submission = final_test[['PassengerId','Survived']]

submission.to_csv("submission.csv", index=False)

submission.tail()

