import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn import metrics
from sklearn.metrics import confusion_matrix
import pickle

# loading the dataset to a Pandas DataFrame
health_dataset = pd.read_csv('/home/chaitanya/HackToFuture/Maternal Health Risk Data Set.csv')

# number of rows & columns in the dataset
health_dataset.shape

# first 5 rows of the dataset
health_dataset.head()

# checking for missing values
health_dataset.isnull().sum()

# statistical measures of the dataset
health_dataset.describe()

# number of values for each RiskLevel
sns.catplot(x='RiskLevel', data = health_dataset, kind = 'count')

# SystolicBP vs Risklevel
plot = plt.figure(figsize=(5,5))
sns.barplot(x='RiskLevel', y = 'SystolicBP', data = health_dataset)

# DiastolicBP vs RiskLevel
plot = plt.figure(figsize=(5,5))
sns.barplot(x='RiskLevel', y = 'DiastolicBP', data = health_dataset)

# HeartRate vs RiskLevel
plot = plt.figure(figsize=(5,5))
sns.barplot(x='RiskLevel', y = 'HeartRate', data = health_dataset)

# Age vs RiskLevel
plot = plt.figure(figsize=(5,5))
sns.barplot(x='RiskLevel', y = 'Age', data = health_dataset)

correlation = health_dataset.corr()

# constructing a heatmap to understand the correlation between the columns
plt.figure(figsize=(10,10))
sns.heatmap(correlation, cbar=True, square=True, fmt = '.1f', annot = True, annot_kws={'size':8}, cmap = 'Blues')

# separate the data and Label
X = health_dataset.drop('RiskLevel',axis=1)

print(X)

Y = health_dataset['RiskLevel']

print(Y)

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=3)

print(Y.shape, Y_train.shape, Y_test.shape)

model = RandomForestClassifier()

model.fit(X_train, Y_train)

# accuracy on test data
Y_test_prediction = model.predict(X_test)
test_data_accuracy = accuracy_score(Y_test_prediction, Y_test)

print('Accuracy : ', test_data_accuracy)

input_data = (23,160,80,7.01,98,70)

# changing the input data to a numpy array
input_data_as_numpy_array = np.asarray(input_data)

# reshape the data as we are predicting the label for only one instance
input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

prediction = model.predict(input_data_reshaped)
print(prediction)

#accuracy score
metrics.accuracy_score(Y_test, Y_test_prediction)

#recall score
recall=metrics.recall_score(Y_test, Y_test_prediction,average='macro')
print(recall)

#precision score
prec=metrics.precision_score(Y_test, Y_test_prediction,average='macro')
print(prec)

#f1score
f1score = 2*((recall*prec)/(recall+prec))#verify mathematically
print(f1score)
f1score1 = metrics.f1_score(Y_test, Y_test_prediction,average='macro')#verify functionally
print(f1score1)

con = confusion_matrix(Y_test , model.predict(X_test))
sns.heatmap(con, annot=True)
plt.tight_layout()
plt.show()
p,q,r,s ,t , u , v ,x,y= confusion_matrix(Y_test ,model.predict(X_test)).ravel()
           
tp = p
          
fn = q+r
fp =s+v
tn = t+u+x+y
tp = t
test_score = (tp+tn)/(tp+tn+fp+fn)
test_score

#ROC curve (receiver operating characteristic curve)



TPR = tp/(tp+fn)  #true postive rate
FPR = fp/(fp+tn)   #false positive rate
#AUC (area under the roc curve)
print(TPR)

print(FPR)

RF_prob = model.predict_proba(X_test)
RF_prob

from sklearn.metrics import roc_auc_score

# auc scores
auc_score1 = roc_auc_score(Y_test, RF_prob,multi_class='ovr')


print(auc_score1)

with open('model.pkl', 'wb') as file:
    pickle.dump(model, file)

