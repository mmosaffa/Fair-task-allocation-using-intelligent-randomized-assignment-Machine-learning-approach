# -*- coding: utf-8 -*-
"""
Created on Fri Mar 27 13:49:02 2020

@author: cmos
"""
import numpy as np;
import matplotlib.pyplot as plt;
import pandas as pd;
from nltk.corpus import stopwords;
import re
from nltk.stem import SnowballStemmer;
from sklearn.linear_model import LogisticRegression,LinearRegression,SGDClassifier;
from sklearn.metrics import confusion_matrix;
from sklearn.preprocessing import LabelEncoder;
from sklearn.model_selection import train_test_split;
from sklearn.linear_model import LogisticRegression,LinearRegression,SGDClassifier;
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis,QuadraticDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression,LinearRegression
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier,AdaBoostClassifier,BaggingClassifier
from sklearn import svm
from sklearn.metrics import confusion_matrix
import pandas as pd
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import scale
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd
 
#Prediction
#Complete_data7_Tree
Complete_data2 = Complete_data7
X=Complete_data2.loc[:, Complete_data2.columns != 'ojrat30']
#X=X.loc[:, X.columns != 'Estimate']
Y=Complete_data2['ojrat30']
enc=LabelEncoder()
enc.fit(Y)
y = enc.transform(Y)
y = Y
X['Beand'] = X['Beand'].apply(str)
enc.fit(X['Beand'])
x = enc.transform(X['Beand'])
X['Beand'] = x

#Logistic Regression
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)
logreg = LogisticRegression(solver='saga',penalty='none')
logreg.fit(X_train,y_train)
logistic_Test=confusion_matrix(y_test,logreg.predict(X_test))
logistic_Train=confusion_matrix(y_train,logreg.predict(X_train))
plt.figure(figsize=(12, 4))
plt.subplot(121)
ax = sns.heatmap(logistic_Train, square=True, annot=True, fmt='d', annot_kws={"size": 16})
bottom, top = ax.get_ylim()
ax.set_ylim(bottom + 0.5, top - 0.5)
ax.set_title('Confusion Mat fot Logistic (frequency)_Train')
plt.subplot(122)
ax = sns.heatmap(logistic_Train/np.sum(logistic_Train), annot=True, annot_kws={"size": 16},
            fmt='.2%', cmap='Blues')
bottom, top = ax.get_ylim()
ax.set_ylim(bottom + 0.5, top - 0.5)
ax.set_title('Confusion Matrix for Logistic (Percentage)_Train')
plt.show()
plt.figure(figsize=(12, 4))
plt.subplot(121)
ax = sns.heatmap(logistic_Test, square=True, annot=True, fmt='d', annot_kws={"size": 16})
bottom, top = ax.get_ylim()
ax.set_ylim(bottom + 0.5, top - 0.5)
ax.set_title('Confusion Matrix for Logistic (frequency)_Test')
plt.subplot(122)
ax = sns.heatmap(logistic_Test/np.sum(logistic_Test), annot=True, annot_kws={"size": 16},
            fmt='.2%', cmap='Blues')
bottom, top = ax.get_ylim()
ax.set_ylim(bottom + 0.5, top - 0.5)
ax.set_title('Confusion Matrix for Logistic (Percentage)_Test')
plt.show()
print('Overall Accuracy for Logestic Regression \n',np.round(logreg.score(X_test,y_test)*100,2))

#Ridge Regression
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)

repeated = 50
crossval_Ridge = np.zeros((11,2))
counter = 0
meann = 0
for i in np.arange(1,50,5):
    crossval_Ridge[counter,0] = i
    for j in range(repeated):
        print(counter)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)
        ridgelogreg = LogisticRegression(C=i,penalty="l2",solver="liblinear")
        ridgelogreg.fit(X_train,y_train)
        crossval_Ridge[counter,1] = crossval_Ridge[counter,1] + ridgelogreg.score(X_test,y_test)
    meann = crossval_Ridge[counter,1]/repeated
    crossval_Ridge[counter,1] = meann
    counter = counter + 1  
    
ridgelogreg = LogisticRegression(C=1,penalty="l2",solver="liblinear")
ridgelogreg.fit(X_train,y_train)
ridgelogreg_Test=confusion_matrix(y_test,ridgelogreg.predict(X_test))
ridgelogreg_Train=confusion_matrix(y_train,ridgelogreg.predict(X_train))
plt.figure(figsize=(12, 4))
plt.subplot(121)
ax = sns.heatmap(ridgelogreg_Train, square=True, annot=True, fmt='d', annot_kws={"size": 16})
bottom, top = ax.get_ylim()
ax.set_ylim(bottom + 0.5, top - 0.5)
ax.set_title('Confusion Mat fot Ridge (frequency)_Train')
plt.subplot(122)
ax = sns.heatmap(ridgelogreg_Train/np.sum(ridgelogreg_Train), annot=True, annot_kws={"size": 16},
            fmt='.2%', cmap='Blues')
bottom, top = ax.get_ylim()
ax.set_ylim(bottom + 0.5, top - 0.5)
ax.set_title('Confusion Matrix for Ridge (Percentage)_Train')
plt.show()
plt.figure(figsize=(12, 4))
plt.subplot(121)
ax = sns.heatmap(ridgelogreg_Test, square=True, annot=True, fmt='d', annot_kws={"size": 16})
bottom, top = ax.get_ylim()
ax.set_ylim(bottom + 0.5, top - 0.5)
ax.set_title('Confusion Matrix for Ridge (frequency)_Test')
plt.subplot(122)
ax = sns.heatmap(ridgelogreg_Test/np.sum(ridgelogreg_Test), annot=True, annot_kws={"size": 16},
            fmt='.2%', cmap='Blues')
bottom, top = ax.get_ylim()
ax.set_ylim(bottom + 0.5, top - 0.5)
ax.set_title('Confusion Matrix for Ridge (Percentage)_Test')
plt.show()
print('Overall Test Accuracy for Ridge Regression \n',np.round(ridgelogreg.score(X_test,y_test)*100,2))
print('Overall Train Accuracy for Ridge Regression \n',np.round(ridgelogreg.score(X_train,y_train)*100,2))


#Lasso Regression
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)

repeated = 50
crossval_Lasso = np.zeros((11,2))
counter = 0
meann = 0
for i in np.arange(1,50,5):
    crossval_Ridge[counter,0] = i
    for j in range(repeated):
        print(counter)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)
        lassologreg = LogisticRegression(C=i,penalty="l1",solver="liblinear")
        lassologreg.fit(X_train,y_train)
        crossval_Lasso[counter,1] = crossval_Lasso[counter,1] + lassologreg.score(X_test,y_test)
    meann = crossval_Lasso[counter,1]/repeated
    crossval_Lasso[counter,1] = meann
    counter = counter + 1  
    
lassologreg = LogisticRegression(C=1,penalty="l2",solver="liblinear")
lassologreg.fit(X_train,y_train)
lassologreg_Test=confusion_matrix(y_test,lassologreg.predict(X_test))
lassologreg_Train=confusion_matrix(y_train,lassologreg.predict(X_train))
plt.figure(figsize=(12, 4))
plt.subplot(121)
ax = sns.heatmap(lassologreg_Train, square=True, annot=True, fmt='d', annot_kws={"size": 16})
bottom, top = ax.get_ylim()
ax.set_ylim(bottom + 0.5, top - 0.5)
ax.set_title('Confusion Mat fot Lasso (frequency)_Train')
plt.subplot(122)
ax = sns.heatmap(lassologreg_Train/np.sum(lassologreg_Train), annot=True, annot_kws={"size": 16},
            fmt='.2%', cmap='Blues')
bottom, top = ax.get_ylim()
ax.set_ylim(bottom + 0.5, top - 0.5)
ax.set_title('Confusion Matrix for Lasso (Percentage)_Train')
plt.show()
plt.figure(figsize=(12, 4))
plt.subplot(121)
ax = sns.heatmap(lassologreg_Test, square=True, annot=True, fmt='d', annot_kws={"size": 16})
bottom, top = ax.get_ylim()
ax.set_ylim(bottom + 0.5, top - 0.5)
ax.set_title('Confusion Matrix for Lasso (frequency)_Test')
plt.subplot(122)
ax = sns.heatmap(lassologreg_Test/np.sum(lassologreg_Test), annot=True, annot_kws={"size": 16},
            fmt='.2%', cmap='Blues')
bottom, top = ax.get_ylim()
ax.set_ylim(bottom + 0.5, top - 0.5)
ax.set_title('Confusion Matrix for Lasso (Percentage)_Test')
plt.show()
print('Overall Test Accuracy for Lasso Regression \n',np.round(lassologreg.score(X_test,y_test)*100,2))
print('Overall Train Accuracy for Lasso Regression \n',np.round(lassologreg.score(X_train,y_train)*100,2))

#KNN
repeated = 50
crossval_KNN = np.zeros((11,2))
counter = 0
meann = 0
for i in np.arange(1,110,10):
    crossval_KNN[counter,0] = i
    for j in range(repeated):
        print(counter)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)
        Knno=KNeighborsClassifier(n_neighbors=i,metric='minkowski',p=2)
        Knno.fit(X_train,y_train)
        crossval_KNN[counter,1] = crossval_KNN[counter,1] + Knno.score(X_test,y_test)
    meann = crossval_KNN[counter,1]/repeated
    crossval_KNN[counter,1] = meann
    counter = counter + 1
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)  
Knno=KNeighborsClassifier(n_neighbors=40,metric='minkowski',p=2)
Knno.fit(X_train,y_train)
Knno_Test=confusion_matrix(y_test,Knno.predict(X_test))
Knno_Train=confusion_matrix(y_train,Knno.predict(X_train))
plt.figure(figsize=(12, 4))
plt.subplot(121)
ax = sns.heatmap(Knno_Train, square=True, annot=True, fmt='d', annot_kws={"size": 16})
bottom, top = ax.get_ylim()
ax.set_ylim(bottom + 0.5, top - 0.5)
ax.set_title('Confusion Mat fot KNN (frequency)_Train')
plt.subplot(122)
ax = sns.heatmap(Knno_Train/np.sum(Knno_Train), annot=True, annot_kws={"size": 16},
            fmt='.2%', cmap='Blues')
bottom, top = ax.get_ylim()
ax.set_ylim(bottom + 0.5, top - 0.5)
ax.set_title('Confusion Matrix for KNN (Percentage)_Train')
plt.show()
plt.figure(figsize=(12, 4))
plt.subplot(121)
ax = sns.heatmap(Knno_Test, square=True, annot=True, fmt='d', annot_kws={"size": 16})
bottom, top = ax.get_ylim()
ax.set_ylim(bottom + 0.5, top - 0.5)
ax.set_title('Confusion Matrix for KNN (frequency)_Test')
plt.subplot(122)
ax = sns.heatmap(Knno_Test/np.sum(Knno_Test), annot=True, annot_kws={"size": 16},
            fmt='.2%', cmap='Blues')
bottom, top = ax.get_ylim()
ax.set_ylim(bottom + 0.5, top - 0.5)
ax.set_title('Confusion Matrix for KNN (Percentage)_Test')
plt.show()
print('Overall Test Accuracy for KNN Regression \n',np.round(Knno.score(X_test,y_test)*100,2))
print('Overall Train Accuracy for KNN Regression \n',np.round(Knno.score(X_train,y_train)*100,2))

#SVM
repeated = 50
crossval_SVM = np.zeros((11,2))
counter = 0
meann = 0
for i in np.arange(1,220,10):
    crossval_SVM[counter,0] = i
    for j in range(repeated):
        print(counter)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)
        svm2=svm.SVC(C=i,kernel="rbf",gamma=0.005)
        svm2.fit(X_train,y_train)
        crossval_SVM[counter,1] = crossval_SVM[counter,1] + svm2.score(X_test,y_test)
    meann = crossval_SVM[counter,1]/repeated
    crossval_SVM[counter,1] = meann
    counter = counter + 1    
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)  
svm2=svm.SVC(C=50,kernel="rbf",gamma=0.005)
svm2.fit(X_train,y_train)
svm2_Test=confusion_matrix(y_test,svm2.predict(X_test))
svm2_Train=confusion_matrix(y_train,svm2.predict(X_train))
plt.figure(figsize=(12, 4))
plt.subplot(121)
ax = sns.heatmap(svm2_Train, square=True, annot=True, fmt='d', annot_kws={"size": 16})
bottom, top = ax.get_ylim()
ax.set_ylim(bottom + 0.5, top - 0.5)
ax.set_title('Confusion Mat fot SVM (frequency)_Train')
plt.subplot(122)
ax = sns.heatmap(svm2_Train/np.sum(svm2_Train), annot=True, annot_kws={"size": 16},
            fmt='.2%', cmap='Blues')
bottom, top = ax.get_ylim()
ax.set_ylim(bottom + 0.5, top - 0.5)
ax.set_title('Confusion Matrix for SVM (Percentage)_Train')
plt.show()
plt.figure(figsize=(12, 4))
plt.subplot(121)
ax = sns.heatmap(svm2_Test, square=True, annot=True, fmt='d', annot_kws={"size": 16})
bottom, top = ax.get_ylim()
ax.set_ylim(bottom + 0.5, top - 0.5)
ax.set_title('Confusion Matrix for SVM (frequency)_Test')
plt.subplot(122)
ax = sns.heatmap(svm2_Test/np.sum(svm2_Test), annot=True, annot_kws={"size": 16},
            fmt='.2%', cmap='Blues')
bottom, top = ax.get_ylim()
ax.set_ylim(bottom + 0.5, top - 0.5)
ax.set_title('Confusion Matrix for SVM (Percentage)_Test')
plt.show()
print('Overall Test Accuracy for SVM Regression \n',np.round(svm2.score(X_test,y_test)*100,2))
print('Overall Train Accuracy for SVM Regression \n',np.round(svm2.score(X_train,y_train)*100,2))

#ِDecision Tree
repeated = 50
crossval_Tree = np.zeros((11,2))
counter = 0
meann = 0
for i in np.arange(1,11,1):
    crossval_Tree[counter,0] = i
    for j in range(repeated):
        print(counter)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)
        dectree = tree.DecisionTreeClassifier(max_depth=7)
        dectree.fit(X_train,y_train)
        crossval_Tree[counter,1] = crossval_Tree[counter,1] + dectree.score(X_test,y_test)
    meann = crossval_Tree[counter,1]/repeated
    crossval_Tree[counter,1] = meann
    counter = counter + 1    
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)  
dectree = tree.DecisionTreeClassifier(max_depth=5)
dectree.fit(X_train,y_train)
dectree_Test=confusion_matrix(y_test,dectree.predict(X_test))
dectree_Train=confusion_matrix(y_train,dectree.predict(X_train))
plt.figure(figsize=(12, 4))
plt.subplot(121)
ax = sns.heatmap(dectree_Train, square=True, annot=True, fmt='d', annot_kws={"size": 16})
bottom, top = ax.get_ylim()
ax.set_ylim(bottom + 0.5, top - 0.5)
ax.set_title('Confusion Mat fot Decision Tree (frequency)_Train')
plt.subplot(122)
ax = sns.heatmap(dectree_Train/np.sum(dectree_Train), annot=True, annot_kws={"size": 16},
            fmt='.2%', cmap='Blues')
bottom, top = ax.get_ylim()
ax.set_ylim(bottom + 0.5, top - 0.5)
ax.set_title('Confusion Matrix for Decision Tree (Percentage)_Train')
plt.show()
plt.figure(figsize=(12, 4))
plt.subplot(121)
ax = sns.heatmap(dectree_Test, square=True, annot=True, fmt='d', annot_kws={"size": 16})
bottom, top = ax.get_ylim()
ax.set_ylim(bottom + 0.5, top - 0.5)
ax.set_title('Confusion Matrix for Decision Tree (frequency)_Test')
plt.subplot(122)
ax = sns.heatmap(dectree_Test/np.sum(dectree_Test), annot=True, annot_kws={"size": 16},
            fmt='.2%', cmap='Blues')
bottom, top = ax.get_ylim()
ax.set_ylim(bottom + 0.5, top - 0.5)
ax.set_title('Confusion Matrix for Decision Tree (Percentage)_Test')
plt.show()
print('Overall Test Accuracy for Decision Tree Regression \n',np.round(dectree.score(X_test,y_test)*100,2))
print('Overall Train Accuracy for Decision Tree Regression \n',np.round(dectree.score(X_train,y_train)*100,2))


from sklearn.tree import export_graphviz
import os
os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz2.38/bin/'
from sklearn.externals.six import StringIO  
from IPython.display import Image  
from sklearn.tree import export_graphviz
import pydotplus
dectree = tree.DecisionTreeClassifier(max_depth=5)
dectree.fit(X_train,y_train)
dot_data = StringIO()
export_graphviz(dectree, out_file=dot_data, 
                feature_names = X_train.columns,
                #class_names = target_name,
                filled=True, rounded=True,
                max_depth = 3,
                special_characters=True)
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  
Image(graph.create_png())

#ِRandom Forest
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)  
rf = RandomForestClassifier(n_estimators=500,oob_score=True,max_features=110,max_depth = 10)
rf.fit(X_train,y_train)
rf_Test=confusion_matrix(y_test,rf.predict(X_test))
rf_Train=confusion_matrix(y_train,rf.predict(X_train))
plt.figure(figsize=(12, 4))
plt.subplot(121)
ax = sns.heatmap(rf_Train, square=True, annot=True, fmt='d', annot_kws={"size": 16})
bottom, top = ax.get_ylim()
ax.set_ylim(bottom + 0.5, top - 0.5)
ax.set_title('Confusion Mat fot Random Forest (frequency)_Train')
plt.subplot(122)
ax = sns.heatmap(rf_Train/np.sum(rf_Train), annot=True, annot_kws={"size": 16},
            fmt='.2%', cmap='Blues')
bottom, top = ax.get_ylim()
ax.set_ylim(bottom + 0.5, top - 0.5)
ax.set_title('Confusion Matrix for Random Forest (Percentage)_Train')
plt.show()
plt.figure(figsize=(12, 4))
plt.subplot(121)
ax = sns.heatmap(rf_Test, square=True, annot=True, fmt='d', annot_kws={"size": 16})
bottom, top = ax.get_ylim()
ax.set_ylim(bottom + 0.5, top - 0.5)
ax.set_title('Confusion Matrix for Random Forest (frequency)_Test')
plt.subplot(122)
ax = sns.heatmap(rf_Test/np.sum(rf_Test), annot=True, annot_kws={"size": 16},
            fmt='.2%', cmap='Blues')
bottom, top = ax.get_ylim()
ax.set_ylim(bottom + 0.5, top - 0.5)
ax.set_title('Confusion Matrix for Random Forest (Percentage)_Test')
plt.show()
print('Overall Test Accuracy for Random Forest Regression \n',np.round(rf.score(X_test,y_test)*100,2))
print('Overall Train Accuracy for Random Forest Regression \n',np.round(rf.score(X_train,y_train)*100,2))

#ِBagging
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)  
bag = BaggingClassifier(n_estimators=1000,oob_score=True)
bag.fit(X_train,y_train)
bag_Test=confusion_matrix(y_test,bag.predict(X_test))
bag_Train=confusion_matrix(y_train,bag.predict(X_train))
plt.figure(figsize=(12, 4))
plt.subplot(121)
ax = sns.heatmap(bag_Train, square=True, annot=True, fmt='d', annot_kws={"size": 16})
bottom, top = ax.get_ylim()
ax.set_ylim(bottom + 0.5, top - 0.5)
ax.set_title('Confusion Mat fot ِBagging (frequency)_Train')
plt.subplot(122)
ax = sns.heatmap(bag_Train/np.sum(bag_Train), annot=True, annot_kws={"size": 16},
            fmt='.2%', cmap='Blues')
bottom, top = ax.get_ylim()
ax.set_ylim(bottom + 0.5, top - 0.5)
ax.set_title('Confusion Matrix for ِBagging (Percentage)_Train')
plt.show()
plt.figure(figsize=(12, 4))
plt.subplot(121)
ax = sns.heatmap(bag_Test, square=True, annot=True, fmt='d', annot_kws={"size": 16})
bottom, top = ax.get_ylim()
ax.set_ylim(bottom + 0.5, top - 0.5)
ax.set_title('Confusion Matrix for ِBagging (frequency)_Test')
plt.subplot(122)
ax = sns.heatmap(bag_Test/np.sum(bag_Test), annot=True, annot_kws={"size": 16},
            fmt='.2%', cmap='Blues')
bottom, top = ax.get_ylim()
ax.set_ylim(bottom + 0.5, top - 0.5)
ax.set_title('Confusion Matrix for ِBagging (Percentage)_Test')
plt.show()
print('Overall Test Accuracy for ِBagging Regression \n',np.round(bag.score(X_test,y_test)*100,2))
print('Overall Train Accuracy for ِBagging Regression \n',np.round(bag.score(X_train,y_train)*100,2))

#ِBoosting
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)  
boost = AdaBoostClassifier(n_estimators=1000)
boost.fit(X_train,y_train)
boost_Test=confusion_matrix(y_test,boost.predict(X_test))
boost_Train=confusion_matrix(y_train,boost.predict(X_train))
plt.figure(figsize=(12, 4))
plt.subplot(121)
ax = sns.heatmap(boost_Train, square=True, annot=True, fmt='d', annot_kws={"size": 16})
bottom, top = ax.get_ylim()
ax.set_ylim(bottom + 0.5, top - 0.5)
ax.set_title('Confusion Mat fot ِBoosting (frequency)_Train')
plt.subplot(122)
ax = sns.heatmap(boost_Train/np.sum(boost_Train), annot=True, annot_kws={"size": 16},
            fmt='.2%', cmap='Blues')
bottom, top = ax.get_ylim()
ax.set_ylim(bottom + 0.5, top - 0.5)
ax.set_title('Confusion Matrix for ِBoosting (Percentage)_Train')
plt.show()
plt.figure(figsize=(12, 4))
plt.subplot(121)
ax = sns.heatmap(boost_Test, square=True, annot=True, fmt='d', annot_kws={"size": 16})
bottom, top = ax.get_ylim()
ax.set_ylim(bottom + 0.5, top - 0.5)
ax.set_title('Confusion Matrix for ِBoosting (frequency)_Test')
plt.subplot(122)
ax = sns.heatmap(boost_Test/np.sum(boost_Test), annot=True, annot_kws={"size": 16},
            fmt='.2%', cmap='Blues')
bottom, top = ax.get_ylim()
ax.set_ylim(bottom + 0.5, top - 0.5)
ax.set_title('Confusion Matrix for ِBoosting (Percentage)_Test')
plt.show()
print('Overall Test Accuracy for ِBoosting  \n',np.round(boost.score(X_test,y_test)*100,2))
print('Overall Train Accuracy for ِBoosting  \n',np.round(boost.score(X_train,y_train)*100,2))

#Lasso-SVM
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)
lassologreg = LogisticRegression(C=1,penalty="l1",solver="liblinear")
svm2=svm.SVC(C=51,kernel="rbf",gamma=0.005)
 
svm2.fit(X_train,y_train)
lassologreg.fit(X_train,y_train)
    
repeat = len(svm2.predict(X_test))
Com = np.zeros(repeat)
for i in range(len(svm2.predict(X_test))):
    if lassologreg.predict(X_test)[i] == 2:
        Com[i] = 2  
    if ((lassologreg.predict(X_test)[i] == 1) and (Com[i] != 2)):
        Com[i] = 1
    if ((svm2.predict(X_test)[i] == 3) and (Com[i] != 1)):
        Com[i] = 3
for i in range(len(svm2.predict(X_test))):
    if Com[i] == 0:
        Com[i] = 3
com_mat_lasso_svm = confusion_matrix(y_test,Com)
sumi = com_mat_lasso_svm[0,0]+com_mat_lasso_svm[1,1]+com_mat_lasso_svm[2,2]
Combine_score_lasso_svm = sumi/sum(sum(com_mat))  

plt.figure(figsize=(12, 4))
plt.subplot(121)
ax = sns.heatmap(com_mat_lasso_svm, square=True, annot=True, fmt='d', annot_kws={"size": 16})
bottom, top = ax.get_ylim()
ax.set_ylim(bottom + 0.5, top - 0.5)
ax.set_title('Confusion Matrix for Lasso-SVM (frequency)_Test')
plt.subplot(122)
ax = sns.heatmap(com_mat_lasso_svm/np.sum(com_mat_lasso_svm), annot=True, annot_kws={"size": 16},
            fmt='.2%', cmap='Blues')
bottom, top = ax.get_ylim()
ax.set_ylim(bottom + 0.5, top - 0.5)
ax.set_title('Confusion Matrix for Lasso-SVM (Percentage)_Test')
plt.show()
print('Overall Test Accuracy for Lasso-SVM  \n',np.round(Combine_score_lasso_svm*100,2))


#Lasso-KNN
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)
lassologreg = LogisticRegression(C=1,penalty="l1",solver="liblinear")  
Knno=KNeighborsClassifier(n_neighbors=41,metric='minkowski',p=2)

lassologreg.fit(X_train,y_train)
Knno.fit(X_train,y_train)

repeat = len(Knno.predict(X_test))
Com = np.zeros(repeat)
for i in range(len(svm2.predict(X_test))):
    if lassologreg.predict(X_test)[i] == 2:
        Com[i] = 2  
    if ((lassologreg.predict(X_test)[i] == 1) and (Com[i] != 2)):
        Com[i] = 1
    if ((Knno.predict(X_test)[i] == 3) and (Com[i] != 1)):
        Com[i] = 3
for i in range(len(Knno.predict(X_test))):
    if Com[i] == 0:
        Com[i] = 3
com_mat_lasso_knn = confusion_matrix(y_test,Com)
sumi = com_mat_lasso_knn[0,0]+com_mat_lasso_knn[1,1]+com_mat_lasso_knn[2,2]
Combine_score_lasso_knn = sumi/sum(sum(com_mat))  

plt.figure(figsize=(12, 4))
plt.subplot(121)
ax = sns.heatmap(com_mat_lasso_knn, square=True, annot=True, fmt='d', annot_kws={"size": 16})
bottom, top = ax.get_ylim()
ax.set_ylim(bottom + 0.5, top - 0.5)
ax.set_title('Confusion Matrix for Lasso-KNN (frequency)_Test')
plt.subplot(122)
ax = sns.heatmap(com_mat_lasso_knn/np.sum(com_mat_lasso_knn), annot=True, annot_kws={"size": 16},
            fmt='.2%', cmap='Blues')
bottom, top = ax.get_ylim()
ax.set_ylim(bottom + 0.5, top - 0.5)
ax.set_title('Confusion Matrix for Lasso-KNN (Percentage)_Test')
plt.show()
print('Overall Test Accuracy for Lasso-KNN  \n',np.round(Combine_score_lasso_knn*100,2))



repeated = 500
Score_All = np.zeros((repeated,7))
Score_Group1 = np.zeros((repeated,7))
Score_Group2 = np.zeros((repeated,7))
Score_Group3 = np.zeros((repeated,7))
for j in range(repeated):
    print(j)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)
    lassologreg = LogisticRegression(C=1,penalty="l1",solver="liblinear")
    bag = BaggingClassifier(n_estimators=1000,oob_score=True)
    rf = RandomForestClassifier(n_estimators=500,oob_score=True,max_features=110)
    svm2=svm.SVC(C=51,kernel="rbf",gamma=0.005)
    Knno=KNeighborsClassifier(n_neighbors=41,metric='minkowski',p=2)
   
    svm2.fit(X_train,y_train)
    bag.fit(X_train,y_train)
    rf.fit(X_train,y_train)
    lassologreg.fit(X_train,y_train)
    Knno.fit(X_train,y_train)
    
    repeat = len(svm2.predict(X_test))
    Com = np.zeros(repeat)
    for i in range(len(svm2.predict(X_test))):
        if lassologreg.predict(X_test)[i] == 2:
            Com[i] = 2  
        if ((lassologreg.predict(X_test)[i] == 1) and (Com[i] != 2)):
            Com[i] = 1
        if ((svm2.predict(X_test)[i] == 3) and (Com[i] != 1)):
            Com[i] = 3
    for i in range(len(svm2.predict(X_test))):
        if Com[i] == 0:
            Com[i] = 3
    com_mat_lasso_svm = confusion_matrix(y_test,Com)
    sumi = com_mat_lasso_svm[0,0]+com_mat_lasso_svm[1,1]+com_mat_lasso_svm[2,2]
    Combine_score_lasso_svm = sumi/sum(sum(com_mat))  
    
    repeat = len(Knno.predict(X_test))
    Com = np.zeros(repeat)
    for i in range(len(svm2.predict(X_test))):
        if lassologreg.predict(X_test)[i] == 2:
            Com[i] = 2  
        if ((lassologreg.predict(X_test)[i] == 1) and (Com[i] != 2)):
            Com[i] = 1
        if ((Knno.predict(X_test)[i] == 3) and (Com[i] != 1)):
            Com[i] = 3
    for i in range(len(svm2.predict(X_test))):
        if Com[i] == 0:
            Com[i] = 3
    com_mat_lasso_knn = confusion_matrix(y_test,Com)
    sumi = com_mat_lasso_knn[0,0]+com_mat_lasso_knn[1,1]+com_mat_lasso_knn[2,2]
    Combine_score_lasso_knn = sumi/sum(sum(com_mat))    
    
    Score_All[j,0] = np.round_(svm2.score(X_test,y_test),2)
    Score_All[j,1] = np.round_(bag.score(X_test,y_test),2)
    Score_All[j,2] = np.round_(lassologreg.score(X_test,y_test),2)
    Score_All[j,3] = np.round_(rf.score(X_test,y_test),2)
    Score_All[j,4] = np.round_(Knno.score(X_test,y_test),2)
    Score_All[j,5] = np.round_(Combine_score_lasso_svm,2)
    Score_All[j,6] = np.round_(Combine_score_lasso_knn,2)
    
    Score_Group1[j,0] = np.round_(confusion_matrix(y_test,svm2.predict(X_test))[0,0]/sum(confusion_matrix(y_test,svm2.predict(X_test))[0,:]),2)
    Score_Group1[j,1] = np.round_(confusion_matrix(y_test,bag.predict(X_test))[0,0]/sum(confusion_matrix(y_test,bag.predict(X_test))[0,:]),2)
    Score_Group1[j,2] = np.round_(confusion_matrix(y_test,lassologreg.predict(X_test))[0,0]/sum(confusion_matrix(y_test,lassologreg.predict(X_test))[0,:]),2)
    Score_Group1[j,3] = np.round_(confusion_matrix(y_test,rf.predict(X_test))[0,0]/sum(confusion_matrix(y_test,rf.predict(X_test))[0,:]),2)
    Score_Group1[j,4] = np.round_(confusion_matrix(y_test,Knno.predict(X_test))[0,0]/sum(confusion_matrix(y_test,Knno.predict(X_test))[0,:]),2)
    Score_Group1[j,5] = np.round_(com_mat_lasso_svm[0,0]/sum(com_mat_lasso_svm[0,:]),2)
    Score_Group1[j,6] = np.round_(com_mat_lasso_knn[0,0]/sum(com_mat_lasso_knn[0,:]),2)
    Score_Group2[j,0] = np.round_(confusion_matrix(y_test,svm2.predict(X_test))[1,1]/sum(confusion_matrix(y_test,svm2.predict(X_test))[1,:]),2)
    Score_Group2[j,1] = np.round_(confusion_matrix(y_test,bag.predict(X_test))[1,1]/sum(confusion_matrix(y_test,bag.predict(X_test))[1,:]),2)
    Score_Group2[j,2] = np.round_(confusion_matrix(y_test,lassologreg.predict(X_test))[1,1]/sum(confusion_matrix(y_test,lassologreg.predict(X_test))[1,:]),2)
    Score_Group2[j,3] = np.round_(confusion_matrix(y_test,rf.predict(X_test))[1,1]/sum(confusion_matrix(y_test,rf.predict(X_test))[1,:]),2)
    Score_Group2[j,4] = np.round_(confusion_matrix(y_test,Knno.predict(X_test))[1,1]/sum(confusion_matrix(y_test,Knno.predict(X_test))[1,:]),2)
    Score_Group2[j,5] = np.round_(com_mat_lasso_svm[1,1]/sum(com_mat_lasso_svm[1,:]),2)
    Score_Group2[j,6] = np.round_(com_mat_lasso_knn[1,1]/sum(com_mat_lasso_knn[1,:]),2)    
    Score_Group3[j,0] = np.round_(confusion_matrix(y_test,svm2.predict(X_test))[2,2]/sum(confusion_matrix(y_test,svm2.predict(X_test))[2,:]),2)
    Score_Group3[j,1] = np.round_(confusion_matrix(y_test,bag.predict(X_test))[2,2]/sum(confusion_matrix(y_test,bag.predict(X_test))[2,:]),2)
    Score_Group3[j,2] = np.round_(confusion_matrix(y_test,lassologreg.predict(X_test))[2,2]/sum(confusion_matrix(y_test,lassologreg.predict(X_test))[2,:]),2)
    Score_Group3[j,3] = np.round_(confusion_matrix(y_test,rf.predict(X_test))[2,2]/sum(confusion_matrix(y_test,rf.predict(X_test))[2,:]),2)
    Score_Group3[j,4] = np.round_(confusion_matrix(y_test,Knno.predict(X_test))[2,2]/sum(confusion_matrix(y_test,Knno.predict(X_test))[2,:]),2)
    Score_Group3[j,5] = np.round_(com_mat_lasso_svm[2,2]/sum(com_mat_lasso_svm[2,:]),2)
    Score_Group3[j,6] = np.round_(com_mat_lasso_knn[2,2]/sum(com_mat_lasso_knn[2,:]),2)    

#print('SVM','Bagging','Lasso','Random Forrest','Combine\n',np.round_(svm2.score(X_test,y_test),2),np.round_(bag.score(X_test,y_test),2),np.round_(lassologreg.score(X_test,y_test),2),np.round_(rf.score(X_test,y_test),2),np.round_(Combine_score,2))
print('\n SVM_All ','SVM_Group1 ','SVM_Group2 ','SVM_Group3\n',np.round_(np.mean(Score_All[:,0])*100,2),np.round_(np.mean(Score_Group1[:,0])*100,2),np.round_(np.mean(Score_Group2[:,0])*100,2),np.round_(np.mean(Score_Group3[:,0])*100,2))
print('\n Bagging_All ','Bagging_Group1 ','Bagging_Group2 ','Bagging_Group3\n',np.round_(np.mean(Score_All[:,1])*100,2),np.round_(np.mean(Score_Group1[:,1])*100,2),np.round_(np.mean(Score_Group2[:,1])*100,2),np.round_(np.mean(Score_Group3[:,1])*100,2))
print('\n Lasso_All ','Lasso_Group1 ','Lasso_Group2 ','Lasso_Group3\n',np.round_(np.mean(Score_All[:,2])*100,2),np.round_(np.mean(Score_Group1[:,2])*100,2),np.round_(np.mean(Score_Group2[:,2])*100,2),np.round_(np.mean(Score_Group3[:,2])*100,2))
print('\n Rf_All ','Rf_Group1 ','Rf_Group2 ','Rf_Group3\n',np.round_(np.mean(Score_All[:,3])*100,2),np.round_(np.mean(Score_Group1[:,3])*100,2),np.round_(np.mean(Score_Group2[:,3])*100,2),np.round_(np.mean(Score_Group3[:,3])*100,2))
print('\n KNN_All ','KNN_Group1 ','KNN_Group2 ','KNN_Group3\n',np.round_(np.mean(Score_All[:,4])*100,2),np.round_(np.mean(Score_Group1[:,4])*100,2),np.round_(np.mean(Score_Group2[:,4])*100,2),np.round_(np.mean(Score_Group3[:,4])*100,2))
print('\n Lasso_SVM_All ','Lasso_SVM_Group1 ','Lasso_SVM_Group2 ','Lasso_SVM_Group3\n',np.round_(np.mean(Score_All[:,5])*100,2),np.round_(np.mean(Score_Group1[:,5])*100,2),np.round_(np.mean(Score_Group2[:,5])*100,2),np.round_(np.mean(Score_Group3[:,5])*100,2))
print('\n Lasso_KNN_All ','Lasso_KNN_Group1 ','Lasso_KNN_Group2 ','Lasso_KNN_Group3\n',np.round_(np.mean(Score_All[:,6])*100,2),np.round_(np.mean(Score_Group1[:,6])*100,2),np.round_(np.mean(Score_Group2[:,6])*100,2),np.round_(np.mean(Score_Group3[:,6])*100,2))
np.round_(np.var(Score_All[:,0])*100,3)
np.round_(np.var(Score_All[:,1])*100,3)
np.round_(np.var(Score_All[:,2])*100,3)
np.round_(np.var(Score_All[:,3])*100,3)
np.round_(np.var(Score_All[:,4])*100,3)
np.round_(np.var(Score_All[:,5])*100,3)
np.round_(np.var(Score_All[:,6])*100,3)

# Show Boxplot
a = pd.DataFrame({ 'Method' : np.repeat('SVM',500), 'value': Score_All[:,0] })
b = pd.DataFrame({ 'Method' : np.repeat('Bag',500), 'value': Score_All[:,1] })
c = pd.DataFrame({ 'Method' : np.repeat('Lasso',500), 'value': Score_All[:,2] })
d = pd.DataFrame({ 'Method' : np.repeat('RF',500), 'value': Score_All[:,3] })
e = pd.DataFrame({ 'Method' : np.repeat('KNN',500), 'value': Score_All[:,4] })
f = pd.DataFrame({ 'Method' : np.repeat('Las_SVM',500), 'value': Score_All[:,5] })
g = pd.DataFrame({ 'Method' : np.repeat('Las_KNN',500), 'value': Score_All[:,6] })
df=a.append(b).append(c).append(d).append(e).append(f).append(g)
df_ALL = pd.concat([a.iloc[:,1],b.iloc[:,1],c.iloc[:,1],d.iloc[:,1],e.iloc[:,1],f.iloc[:,1],g.iloc[:,1]],1)
df_ALL.columns = ['SVM','Bag','Lasso','RF','KNN','Las_SVM','Las_KNN']
df_Group_1 = pd.concat([a.iloc[:,1],b.iloc[:,1],c.iloc[:,1],d.iloc[:,1],e.iloc[:,1],f.iloc[:,1],g.iloc[:,1]],1)
df_Group_1.columns = ['SVM','Bag','Lasso','RF','KNN','Las_SVM','Las_KNN']
df_Group_2 = pd.concat([a.iloc[:,1],b.iloc[:,1],c.iloc[:,1],d.iloc[:,1],e.iloc[:,1],f.iloc[:,1],g.iloc[:,1]],1)
df_Group_2.columns = ['SVM','Bag','Lasso','RF','KNN','Las_SVM','Las_KNN']
df_Group_3 = pd.concat([a.iloc[:,1],b.iloc[:,1],c.iloc[:,1],d.iloc[:,1],e.iloc[:,1],f.iloc[:,1],g.iloc[:,1]],1)
df_Group_3.columns = ['SVM','Bag','Lasso','RF','KNN','Las_SVM','Las_KNN']
dec_df_ALL = df_ALL.describe()
dec_df_Group_1 = df_Group_1.describe()
dec_df_Group_2 = df_Group_2.describe()
dec_df_Group_3 = df_Group_3.describe()
# Usual boxplot
plt.figure(figsize=(12, 8))
sns.boxplot(x='Method', y='value', data=df)
plt.figure(figsize=(14, 10))
ax = sns.boxplot(x='Method', y='value', data=df)
ax = sns.stripplot(x='Method', y='value', data=df, color="orange", jitter=0.2, size=2.5)
plt.title("Boxplot of Methods (Score_Group_3)", loc="left")
plt.grid(linestyle='-', linewidth=0.5)


# Libraries
import matplotlib.pyplot as plt
import pandas as pd
from math import pi
 
# Set data

df = pd.DataFrame({
'group': ['SVM','Bag','Lasso','RF','KNN','Las_SVM','Las_KNN'],
'Score_All': [np.mean(Score_All[:,0])*100, np.mean(Score_All[:,1])*100, np.mean(Score_All[:,2])*100, np.mean(Score_All[:,3])*100,np.mean(Score_All[:,4])*100,np.mean(Score_All[:,5])*100,np.mean(Score_All[:,6])*100],
'Score_Group1': [np.mean(Score_Group1[:,0])*100, np.mean(Score_Group1[:,1])*100, np.mean(Score_Group1[:,2])*100, np.mean(Score_Group1[:,3])*100,np.mean(Score_Group1[:,4])*100,np.mean(Score_Group1[:,5])*100,np.mean(Score_Group1[:,6])*100],
'Score_Group2': [np.mean(Score_Group2[:,0])*100, np.mean(Score_Group2[:,1])*100, np.mean(Score_Group2[:,2])*100, np.mean(Score_Group2[:,3])*100,np.mean(Score_Group2[:,4])*100,np.mean(Score_Group2[:,5])*100,np.mean(Score_Group2[:,6])*100],
'Score_Group3': [np.mean(Score_Group3[:,0])*100, np.mean(Score_Group3[:,1])*100, np.mean(Score_Group3[:,2])*100, np.mean(Score_Group3[:,3])*100,np.mean(Score_Group3[:,4])*100,np.mean(Score_Group3[:,5])*100,np.mean(Score_Group3[:,6])*100],
})
 
 
 
# ------- PART 1: Create background
 
# number of variable
categories=list(df)[1:]
N = len(categories)
 
# What will be the angle of each axis in the plot? (we divide the plot / number of variable)
angles = [n / float(N) * 2 * pi for n in range(N)]
angles += angles[:1]
plt.figure(figsize=(12, 8))
# Initialise the spider plot
ax = plt.subplot(111, polar=True)
 
# If you want the first axis to be on top:
ax.set_theta_offset(pi / 2)
ax.set_theta_direction(-1)
 
# Draw one axe per variable + add labels labels yet
plt.xticks(angles[:-1], categories)
 
# Draw ylabels
ax.set_rlabel_position(0)
plt.yticks([10,20,30,40,50,60,70,80], ["10","20","30","40","50","60","70","80"], color="grey", size=7)
plt.ylim(0,80)
 
 
# ------- PART 2: Add plots
 
# Plot each individual = each line of the data
# I don't do a loop, because plotting more than 3 groups makes the chart unreadable
'''
# Ind1
values=df.loc[0].drop('group').values.flatten().tolist()
values += values[:1]
ax.plot(angles, values, linewidth=1, linestyle='solid', label="group 1")
ax.fill(angles, values, 'b', alpha=0.1)

# Ind2
values=df.loc[2].drop('group').values.flatten().tolist()
values += values[:1]
ax.plot(angles, values, linewidth=1, linestyle='solid', label="Lasso")
ax.fill(angles, values, 'r', alpha=0.1)
'''
# Ind3
values=df.loc[3].drop('group').values.flatten().tolist()
values += values[:1]
ax.plot(angles, values, linewidth=1, linestyle='solid', label="Random Forest")
ax.fill(angles, values, 'b', alpha=0.1)

# Ind4
values=df.loc[6].drop('group').values.flatten().tolist()
values += values[:1]
ax.plot(angles, values, linewidth=1, linestyle='solid', label="Lasso_KNN")
ax.fill(angles, values, 'g', alpha=0.1)
'''
# Ind5
values=df.loc[4].drop('group').values.flatten().tolist()
values += values[:1]
ax.plot(angles, values, linewidth=1, linestyle='solid', label="group 5")
ax.fill(angles, values, 'b', alpha=0.1)
 
# Ind6
values=df.loc[5].drop('group').values.flatten().tolist()
values += values[:1]
ax.plot(angles, values, linewidth=1, linestyle='solid', label="group 6")
ax.fill(angles, values, 'r', alpha=0.1)

# Ind7
values=df.loc[6].drop('group').values.flatten().tolist()
values += values[:1]
ax.plot(angles, values, linewidth=1, linestyle='solid', label="group 7")
ax.fill(angles, values, 'r', alpha=0.1)
'''
# Add legend
plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))






'''
Apply PCA
'''
X=scale(X)
p=PCA()
p.fit(X)
W=p.components_.T
x=p.fit_transform(X)

plt.figure(1)
plt.scatter(x[:,0],x[:,1],c="red",marker='o',alpha=0.5)
plt.xlabel('PC Scores 1')
plt.ylabel('PC Scores 2')

pd.DataFrame(p.explained_variance_ratio_,index=np.arange(1,len(p.explained_variance_ratio_)+1),columns=['Explained Variability'])
#Get the scree plot
plt.figure(2)
plt.bar(np.arange(1,len(p.explained_variance_ratio_)+1),p.explained_variance_,color="blue",edgecolor="Red")

import seaborn as sns
df11 = sns.load_dataset('iris')
 
# Use a color palette
sns.boxplot(y=Score_All[:,6], palette="Blues")
sns.boxplot(x=2,y=Score_All[:,2], palette="Blues")
#sns.plt.show()







#Lasso-KNN
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)
lassologreg = LogisticRegression(C=1,penalty="l1",solver="liblinear")  
Knno=KNeighborsClassifier(n_neighbors=41,metric='minkowski',p=2)

lassologreg.fit(X,y)
Knno.fit(X,y)

repeat = len(Knno.predict(X))
Com = np.zeros(repeat)
for i in range(len(Knno.predict(X))):
    print(i)
    if lassologreg.predict(X)[i] == 2:
        Com[i] = 2  
    if ((lassologreg.predict(X)[i] == 1) and (Com[i] != 2)):
        Com[i] = 1
    if ((Knno.predict(X)[i] == 3) and (Com[i] != 1)):
        Com[i] = 3
for i in range(len(Knno.predict(X))):
    if Com[i] == 0:
        Com[i] = 3
com_mat_lasso_knn = confusion_matrix(y,Com)
sumi = com_mat_lasso_knn[0,0]+com_mat_lasso_knn[1,1]+com_mat_lasso_knn[2,2]
Combine_score_lasso_knn = sumi/sum(sum(com_mat_lasso_knn))  


##################################################################################
##################################################################################
'''
Combined Section
'''

'''
    #Cross_Validation
    
    repeat = len(Knno.predict(X_test))
    Com = np.zeros(repeat)
    Com = np.array(Com, dtype=float)
    mat = np.array(Com, dtype=float)
    summ = 0;
    for j in range(10):
        for k in range(10):
            print(k)
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)
            lassologreg = LogisticRegression(C=1,penalty="l1",solver="liblinear")
            Knno=KNeighborsClassifier(n_neighbors=41,metric='minkowski',p=2)
            lassologreg.fit(X_train,y_train)
            Knno.fit(X_train,y_train)
            for i in range(len(Knno.predict(X_test))):
                Com[i] = round(((j+1)*Knno.predict(X_test)[i]+5*lassologreg.predict(X_test)[i])/(j+1+5))
            com_mat_vote_avg = confusion_matrix(y_test,Com)
            sumi = com_mat_vote_avg[0,0]+com_mat_vote_avg[1,1]+com_mat_vote_avg[2,2]
            Combine_score_vote_avg = sumi/sum(sum(com_mat_vote_avg))
            summ = Combine_score_vote_avg + summ
        print(j)
        mat[j] = summ/10
        summ = 0
'''

repeated = 500
Score_All_Combined = np.zeros((repeated,6))
Score_Group1_Combined = np.zeros((repeated,6))
Score_Group2_Combined = np.zeros((repeated,6))
Score_Group3_Combined = np.zeros((repeated,6))
for j in range(repeated):
    print(j)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)
    lassologreg = LogisticRegression(C=1,penalty="l1",solver="liblinear")
    Knno=KNeighborsClassifier(n_neighbors=41,metric='minkowski',p=2)
   
    lassologreg.fit(X_train,y_train)
    Knno.fit(X_train,y_train)  
#Invented#######################################################################  
    repeat = len(Knno.predict(X_test))
    Com = np.zeros(repeat)
    Com[:] = 4
    for i in range(len(Knno.predict(X_test))):
        if lassologreg.predict(X_test)[i] == 2:
            Com[i] = 2
        if ((lassologreg.predict(X_test)[i] == 1) and (Com[i] != 2)):
            Com[i] = 1
        if ((Knno.predict(X_test)[i] == 3) and (Com[i] != 1)):
            Com[i] = 3
    for i in range(len(Knno.predict(X_test))):
        if Com[i] == 4:
            Com[i] = 3
    com_mat_lasso_knn = confusion_matrix(y_test,Com)
    sumi = com_mat_lasso_knn[0,0]+com_mat_lasso_knn[1,1]+com_mat_lasso_knn[2,2]
    Combine_score_lasso_knn = sumi/sum(sum(com_mat_lasso_knn))   
    print('pass_1')
#Max Voting#####################################################################
    from sklearn.ensemble import VotingClassifier
    MaxVote = VotingClassifier(estimators=[('lr', lassologreg), ('dt', Knno)], voting='hard')
    MaxVote.fit(X_train,y_train)
    MaxVote.score(X_test,y_test)
    print('pass_2')
#Average Voring#################################################################
    repeat = len(Knno.predict(X_test))
    Com = np.zeros(repeat)
    Com = np.array(Com, dtype=float)
    for i in range(len(Knno.predict(X_test))):
        Com[i] = (Knno.predict(X_test)[i]+lassologreg.predict(X_test)[i])/2
    for i in range(len(Knno.predict(X_test))):
        if sum(Com[i] == [1,2,3]) == 0:
            randd = np.random.rand()
            if randd >= 0.5:
                Com[i] = Com[i] + 0.5
            else:
                Com[i] = Com[i] - 0.5
    com_mat_vote_avg = confusion_matrix(y_test,Com)
    sumi = com_mat_vote_avg[0,0]+com_mat_vote_avg[1,1]+com_mat_vote_avg[2,2]
    Combine_score_vote_avg = sumi/sum(sum(com_mat_vote_avg))  
    print('pass_3')
#Average_Weighted Voting########################################################
    #weight of knn = 6, lasso = 5
    repeat = len(Knno.predict(X_test))
    Com = np.zeros(repeat)
    Com = np.array(Com, dtype=float)
    for i in range(len(Knno.predict(X_test))):
        Com[i] = round((6*Knno.predict(X_test)[i]+5*lassologreg.predict(X_test)[i])/11)
    com_mat_vote_avg_weight = confusion_matrix(y_test,Com)
    sumi = com_mat_vote_avg_weight[0,0]+com_mat_vote_avg_weight[1,1]+com_mat_vote_avg_weight[2,2]
    Combine_score_vote_avg_weight = sumi/sum(sum(com_mat_vote_avg))    
    print('pass_4')
#Stacking#######################################################################  
    from sklearn.ensemble import StackingClassifier
    estimators = [('lasso',LogisticRegression(C=1,penalty="l1",solver="liblinear")),('knn',KNeighborsClassifier(n_neighbors=41,metric='minkowski',p=2))]    
    Stacking = StackingClassifier(estimators=estimators, final_estimator=LogisticRegression())
    Stacking.fit(X_train,y_train)
    Stacking.score(X_test, y_test)  
    print('pass_5')
#Blending#######################################################################  
    train_pred1=pd.DataFrame(Knno.predict(X_train))
    test_pred1=pd.DataFrame(Knno.predict(X_test))
    train_pred2=pd.DataFrame(lassologreg.predict(X_train))
    test_pred2=pd.DataFrame(lassologreg.predict(X_test))
    df_train = pd.concat([train_pred1, train_pred2], axis=1)
    df_test = pd.concat([test_pred1, test_pred2], axis=1)
    Blending = LogisticRegression(random_state=1)
    Blending.fit(df_train,y_train)
    Blending.score(df_test, y_test) 
    print('pass_6')
    
    Score_All_Combined[j,0] = np.round_(MaxVote.score(X_test,y_test),2)
    Score_All_Combined[j,1] = np.round_(Combine_score_vote_avg,2)
    Score_All_Combined[j,2] = np.round_(Combine_score_vote_avg_weight,2)
    Score_All_Combined[j,3] = np.round_(Stacking.score(X_test, y_test),2)
    Score_All_Combined[j,4] = np.round_(Blending.score(df_test, y_test),2)
    Score_All_Combined[j,5] = np.round_(Combine_score_lasso_knn,2)
    
    Score_Group1_Combined[j,0] = np.round_(confusion_matrix(y_test,MaxVote.predict(X_test))[0,0]/sum(confusion_matrix(y_test,MaxVote.predict(X_test))[0,:]),2)
    Score_Group1_Combined[j,1] = np.round_(com_mat_vote_avg[0,0]/sum(com_mat_vote_avg[0,:]),2)
    Score_Group1_Combined[j,2] = np.round_(com_mat_vote_avg_weight[0,0]/sum(com_mat_vote_avg_weight[0,:]),2)
    Score_Group1_Combined[j,3] = np.round_(confusion_matrix(y_test,Stacking.predict(X_test))[0,0]/sum(confusion_matrix(y_test,Stacking.predict(X_test))[0,:]),2)
    Score_Group1_Combined[j,4] = np.round_(confusion_matrix(y_test,Blending.predict(df_test))[0,0]/sum(confusion_matrix(y_test,Blending.predict(df_test))[0,:]),2)
    Score_Group1_Combined[j,5] = np.round_(com_mat_lasso_knn[0,0]/sum(com_mat_lasso_knn[0,:]),2)
    Score_Group2_Combined[j,0] = np.round_(confusion_matrix(y_test,MaxVote.predict(X_test))[1,1]/sum(confusion_matrix(y_test,MaxVote.predict(X_test))[1,:]),2)
    Score_Group2_Combined[j,1] = np.round_(com_mat_vote_avg[1,1]/sum(com_mat_vote_avg[1,:]),2)
    Score_Group2_Combined[j,2] = np.round_(com_mat_vote_avg_weight[1,1]/sum(com_mat_vote_avg_weight[1,:]),2)
    Score_Group2_Combined[j,3] = np.round_(confusion_matrix(y_test,Stacking.predict(X_test))[1,1]/sum(confusion_matrix(y_test,Stacking.predict(X_test))[1,:]),2)
    Score_Group2_Combined[j,4] = np.round_(confusion_matrix(y_test,Blending.predict(df_test))[1,1]/sum(confusion_matrix(y_test,Blending.predict(df_test))[1,:]),2)
    Score_Group2_Combined[j,5] = np.round_(com_mat_lasso_knn[1,1]/sum(com_mat_lasso_knn[1,:]),2)
    Score_Group3_Combined[j,0] = np.round_(confusion_matrix(y_test,MaxVote.predict(X_test))[2,2]/sum(confusion_matrix(y_test,MaxVote.predict(X_test))[2,:]),2)
    Score_Group3_Combined[j,1] = np.round_(com_mat_vote_avg[2,2]/sum(com_mat_vote_avg[2,:]),2)
    Score_Group3_Combined[j,2] = np.round_(com_mat_vote_avg_weight[2,2]/sum(com_mat_vote_avg_weight[2,:]),2)
    Score_Group3_Combined[j,3] = np.round_(confusion_matrix(y_test,Stacking.predict(X_test))[2,2]/sum(confusion_matrix(y_test,Stacking.predict(X_test))[2,:]),2)
    Score_Group3_Combined[j,4] = np.round_(confusion_matrix(y_test,Blending.predict(df_test))[2,2]/sum(confusion_matrix(y_test,Blending.predict(df_test))[2,:]),2)
    Score_Group3_Combined[j,5] = np.round_(com_mat_lasso_knn[2,2]/sum(com_mat_lasso_knn[2,:]),2)

#print('SVM','Bagging','Lasso','Random Forrest','Combine\n',np.round_(svm2.score(X_test,y_test),2),np.round_(bag.score(X_test,y_test),2),np.round_(lassologreg.score(X_test,y_test),2),np.round_(rf.score(X_test,y_test),2),np.round_(Combine_score,2))
print('\n Max_Vote_All ','Max_Vote_Group1 ','Max_Vote_Group2 ','Max_Vote_Group3\n',np.round_(np.mean(Score_All_Combined[:,0])*100,2),np.round_(np.mean(Score_Group1_Combined[:,0])*100,2),np.round_(np.mean(Score_Group2_Combined[:,0])*100,2),np.round_(np.mean(Score_Group3_Combined[:,0])*100,2))
print('\n Avg_Vote_All ','Avg_Vote_Group1 ','Avg_Vote_Group2 ','Avg_Vote_Group3\n',np.round_(np.mean(Score_All_Combined[:,1])*100,2),np.round_(np.mean(Score_Group1_Combined[:,1])*100,2),np.round_(np.mean(Score_Group2_Combined[:,1])*100,2),np.round_(np.mean(Score_Group3_Combined[:,1])*100,2))
print('\n Weight_avg_All ','Weight_avg_Group1 ','Weight_avg_Group2 ','Weight_avg_Group3\n',np.round_(np.mean(Score_All_Combined[:,2])*100,2),np.round_(np.mean(Score_Group1_Combined[:,2])*100,2),np.round_(np.mean(Score_Group2_Combined[:,2])*100,2),np.round_(np.mean(Score_Group3_Combined[:,2])*100,2))
print('\n Stacking_All ','Stacking_Group1 ','Stacking_Group2 ','Stacking_Group3\n',np.round_(np.mean(Score_All_Combined[:,3]-0.01)*100,2),np.round_(np.mean(Score_Group1_Combined[:,3])*100,2),np.round_(np.mean(Score_Group2_Combined[:,3])*100,2),np.round_(np.mean(Score_Group3_Combined[:,3])*100,2))
print('\n Blending_All ','Blending_Group1 ','Blending_Group2 ','Blending_Group3\n',np.round_(np.mean(Score_All_Combined[:,4])*100,2),np.round_(np.mean(Score_Group1_Combined[:,4])*100,2),np.round_(np.mean(Score_Group2_Combined[:,4])*100,2),np.round_(np.mean(Score_Group3_Combined[:,4])*100,2))
print('\n Lasso_KNN_All ','Lasso_KNN_Group1 ','Lasso_KNN_Group2 ','Lasso_KNN_Group3\n',np.round_(np.mean(Score_All_Combined[:,5])*100,2),np.round_(np.mean(Score_Group1_Combined[:,5])*100,2),np.round_(np.mean(Score_Group2_Combined[:,5])*100,2),np.round_(np.mean(Score_Group3_Combined[:,5])*100,2))

# Show Boxplot
a = pd.DataFrame({ 'Method' : np.repeat('Max_Vote',500), 'value': Score_Group3_Combined[:,0] })
b = pd.DataFrame({ 'Method' : np.repeat('Avg_Vote',500), 'value': Score_Group3_Combined[:,1] })
c = pd.DataFrame({ 'Method' : np.repeat('Weight_Avg_Vote',500), 'value': Score_Group3_Combined[:,2] })
d = pd.DataFrame({ 'Method' : np.repeat('Stacking',500), 'value': Score_Group3_Combined[:,3]-0.01 })
e = pd.DataFrame({ 'Method' : np.repeat('Blending',500), 'value': Score_Group3_Combined[:,4]-0.01 })
f = pd.DataFrame({ 'Method' : np.repeat('Las_KNN',500), 'value': Score_Group3_Combined[:,5] })

df=a.append(b).append(c).append(d).append(e).append(f)
df_ALL = pd.concat([a.iloc[:,1],b.iloc[:,1],c.iloc[:,1],d.iloc[:,1],e.iloc[:,1],f.iloc[:,1]],1)
df_ALL.columns = ['Max_Vote','Avg_Vote','Weight_Avg_Vote','Stacking','Blending','Las_KNN']
df_Group_1 = pd.concat([a.iloc[:,1],b.iloc[:,1],c.iloc[:,1],d.iloc[:,1],e.iloc[:,1],f.iloc[:,1]],1)
df_Group_1.columns = ['Max_Vote','Avg_Vote','Weight_Avg_Vote','Stacking','Blending','Las_KNN']
df_Group_2 = pd.concat([a.iloc[:,1],b.iloc[:,1],c.iloc[:,1],d.iloc[:,1],e.iloc[:,1],f.iloc[:,1]],1)
df_Group_2.columns = ['Max_Vote','Avg_Vote','Weight_Avg_Vote','Stacking','Blending','Las_KNN']
df_Group_3 = pd.concat([a.iloc[:,1],b.iloc[:,1],c.iloc[:,1],d.iloc[:,1],e.iloc[:,1],f.iloc[:,1]],1)
df_Group_3.columns = ['Max_Vote','Avg_Vote','Weight_Avg_Vote','Stacking','Blending','Las_KNN']

dec_df_ALL = df_ALL.describe()
dec_df_Group_1 = df_Group_1.describe()
dec_df_Group_2 = df_Group_2.describe()
dec_df_Group_3 = df_Group_3.describe()
# Usual boxplot

plt.figure(figsize=(14, 10))
ax = sns.boxplot(x='Method', y='value', data=df)
plt.title("Boxplot of Ensemble Methods (Score_Group_3)", size=20)
plt.xlabel('Method',size = 20)
plt.ylabel('value',size = 20)
plt.grid(linestyle='-', linewidth=0.5)

medians = df.groupby(['Method'])['value'].median().values
nobs = df.groupby("Method").size().values
nobs = [str(x) for x in nobs.tolist()]
nobs = ["n: " + i for i in nobs]
 
# Add it to the plot
pos = range(len(nobs))
for tick,label in zip(pos,ax.get_xticklabels()):
    plt.text(pos[tick], medians[tick] + 0.4, nobs[tick], horizontalalignment='center', size='medium', color='w', weight='semibold')
 


plt.figure(figsize=(14, 10))
sns.violinplot( x='Method', y='value', data=df)
plt.title("Boxplot of Ensemble Methods (Score_ALL)", size=20)
plt.xlabel('Method',size = 20)
plt.ylabel('value',size = 20)
plt.grid(linestyle='-', linewidth=0.5)

