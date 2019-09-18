import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset=pd.read_csv("diabetes.csv")

dataset.head(3)

#checking there is null 
dataset.isnull().values.any()
dataset.isnull().sum()

#visualizing
dataset.plot(kind="box",subplots=True,layout=(3,3))
plt.show()

#finding the 0 in columns
len(dataset[dataset['SkinThickness'] == 0]==1)
len(dataset[dataset['Glucose'] == 0]==1)
len(dataset[dataset['BloodPressure'] == 0]==1)
len(dataset[dataset['Age'] == 0]==1)
len(dataset[dataset['Insulin'] == 0]==1)
len(dataset[dataset['BMI'] == 0]==1)


y=dataset["Outcome"]
x=dataset.drop("Outcome",axis=1)

from sklearn.cross_validation import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)


x2=x_train["Pregnancies"].values
x3=x_test["Pregnancies"].values
#converting them to arrays
x2=x2.reshape(614,1)
x3=x3.reshape(154,1)

#imputer to replace 0
from sklearn.preprocessing import Imputer
fill_values = Imputer(missing_values=0, strategy="mean", axis=0)
x_train = fill_values.fit_transform(x_train)
x_test = fill_values.fit_transform(x_test)

#replacing the pregnancy columns with original one
x_train=np.concatenate((x_train,x2),axis=1)
x_test=np.concatenate((x_test,x3),axis=1)
x_train=np.delete(x_train, 0, axis=1) 
x_test=np.delete(x_test, 0, axis=1) 

#1st classifier
from sklearn.ensemble import RandomForestClassifier

clf=RandomForestClassifier(n_estimators=10,random_state=0)
clf.fit(x_train,y_train)


y_pred=clf.predict(x_test)
from sklearn.metrics import confusion_matrix
c=confusion_matrix(y_pred,y_test)
from sklearn import metrics
metrics.accuracy_score(y_pred,y_test)

#2 classifier
from sklearn.svm import SVC
clf3=SVC(kernel="linear",random_state=0)
clf3.fit(x_train,y_train)

y_pred3=clf.predict(x_test)
from sklearn.metrics import confusion_matrix
c=confusion_matrix(y_pred3,y_test)
from sklearn import metrics
metrics.accuracy_score(y_pred3,y_test)

#grid search
from sklearn.model_selection import GridSearchCV
import xgboost
classifier=xgboost.XGBClassifier()

params={
        "criterion":["gini","entropy"],
        "max_depth":[2,3,4,5],
        
        "min_samples_leaf":[1,2,3,45] ,
        "min_weight_fraction_leaf":[0,0.01,0.03,0.02,0.04,0.2,0.3,0.4],
        "max_features":["auto"],
        "max_leaf_nodes":[2,3,4]
        
        }

clf2=GridSearchCV(estimator=clf,param_grid=params,scoring="accuracy",n_jobs=-1,cv=10)
clf2.fit(x_train,y_train)
best_accuracy=clf2.best_score_
best_params=clf2.best_params_

y_pred2=clf2.predict(x_test)


from sklearn.metrics import confusion_matrix
c2=confusion_matrix(y_pred2,y_test)
metrics.accuracy_score(y_pred2,y_test)













