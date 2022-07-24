# -*- coding: utf-8 -*-
"""
Created on Sat Mar 30 17:45:04 2013

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
import seaborn as sns


comperation = np.zeros((12,4))


comperation = np.zeros((12,4))

#Random with predictors
df = pd.read_excel('C:/Users/cmos/Desktop/New folder (2)/Predictor.xlsx',encoding='latin-1')
df = df.iloc[:,2:]
jobs = df.shape[0]
persons = 5
rnadd = 0
sumi = 0
condition1 = 0
persondata_month_Pre = np.zeros((1000,persons*12))
persongetdata = np.zeros((1,persons))
meandata_month_Pre = np.zeros((100,12))
counterjob = 0
passingjob = 0
counter = 0
alter = df.iloc[0,1]
for t in range(100):
    print(t)
    while counterjob<jobs:
        #print(counterjob)
        passingjob = passingjob + counter
        counter = 0
        for i in range(passingjob,jobs):
            if alter == df.iloc[i,1]:
                counter = counter + 1
            else:
                persongetdata[0,:] = 0
                alter = df.iloc[i,1]
                break
        for i in range(passingjob,passingjob+counter):
            condition1 = 0
            if sum(sum(persongetdata)) == persons:
                persongetdata[0,:] = 0
            if df.iloc[i,3] == 3:
                while condition1 == 0:
                    rnadd = np.random.randint(0,persons)
                    if persongetdata[0,rnadd] == 0:
                        for j in range(len(persondata_month_Pre)):
                            if persondata_month_Pre[j,(int(df.iloc[i,0])-1)*5 + rnadd] == 0:
                                persondata_month_Pre[j,(int(df.iloc[i,0])-1)*5 + rnadd] = df.iloc[i,2]
                                #persondata[i,rnadd+5] = d
                                sumi = sumi + df.iloc[i,2]
                                counterjob = counterjob + 1
                                break
                        persongetdata[0,rnadd] = 1
                        condition1 = 1
        for i in range(passingjob,passingjob+counter):
            condition1 = 0
            if sum(sum(persongetdata)) == persons:
                persongetdata[0,:] = 0
            if df.iloc[i,3] == 2:
                while condition1 == 0:
                    rnadd = np.random.randint(0,persons)
                    if persongetdata[0,rnadd] == 0:
                        for j in range(len(persondata_month_Pre)):
                            if persondata_month_Pre[j,(int(df.iloc[i,0])-1)*5 + rnadd] == 0:
                                persondata_month_Pre[j,(int(df.iloc[i,0])-1)*5 + rnadd] = df.iloc[i,2]
                                #persondata[i,rnadd+5] = d
                                sumi = sumi + df.iloc[i,2]
                                counterjob = counterjob + 1
                                break
                        persongetdata[0,rnadd] = 1
                        condition1 = 1
                        break
        for i in range(passingjob,passingjob+counter):
            condition1 = 0
            if sum(sum(persongetdata)) == persons:
                persongetdata[0,:] = 0
            if df.iloc[i,3] == 1:
                while condition1 == 0:
                    rnadd = np.random.randint(0,persons)
                    if persongetdata[0,rnadd] == 0:
                        for j in range(len(persondata_month_Pre)):
                            if persondata_month_Pre[j,(int(df.iloc[i,0])-1)*5 + rnadd] == 0:
                                persondata_month_Pre[j,(int(df.iloc[i,0])-1)*5 + rnadd] = df.iloc[i,2]
                                #persondata[i,rnadd+5] = d
                                sumi = sumi + df.iloc[i,2]
                                counterjob = counterjob + 1
                                break
                        persongetdata[0,rnadd] = 1
                        condition1 = 1
                              
    meandata_month_Pre[t,0] = abs(sum(persondata_month_Pre[:,0])-sum(sum(persondata_month_Pre[:,:5]))/5)+abs(sum(persondata_month_Pre[:,1])-sum(sum(persondata_month_Pre[:,:5]))/5)+abs(sum(persondata_month_Pre[:,2])-sum(sum(persondata_month_Pre[:,:5]))/5)+abs(sum(persondata_month_Pre[:,3])-sum(sum(persondata_month_Pre[:,:5]))/5)+abs(sum(persondata_month_Pre[:,4])-sum(sum(persondata_month_Pre[:,:5]))/5)                     
    meandata_month_Pre[t,1] = abs(sum(persondata_month_Pre[:,5])-sum(sum(persondata_month_Pre[:,5:10]))/5)+abs(sum(persondata_month_Pre[:,6])-sum(sum(persondata_month_Pre[:,5:10]))/5)+abs(sum(persondata_month_Pre[:,7])-sum(sum(persondata_month_Pre[:,5:10]))/5)+abs(sum(persondata_month_Pre[:,8])-sum(sum(persondata_month_Pre[:,5:10]))/5)+abs(sum(persondata_month_Pre[:,9])-sum(sum(persondata_month_Pre[:,5:10]))/5)                     
    meandata_month_Pre[t,2] = abs(sum(persondata_month_Pre[:,10])-sum(sum(persondata_month_Pre[:,10:15]))/5)+abs(sum(persondata_month_Pre[:,11])-sum(sum(persondata_month_Pre[:,10:15]))/5)+abs(sum(persondata_month_Pre[:,12])-sum(sum(persondata_month_Pre[:,10:15]))/5)+abs(sum(persondata_month_Pre[:,13])-sum(sum(persondata_month_Pre[:,10:15]))/5)+abs(sum(persondata_month_Pre[:,14])-sum(sum(persondata_month_Pre[:,10:15]))/5)                     
    meandata_month_Pre[t,3] = abs(sum(persondata_month_Pre[:,15])-sum(sum(persondata_month_Pre[:,15:20]))/5)+abs(sum(persondata_month_Pre[:,16])-sum(sum(persondata_month_Pre[:,15:20]))/5)+abs(sum(persondata_month_Pre[:,17])-sum(sum(persondata_month_Pre[:,15:20]))/5)+abs(sum(persondata_month_Pre[:,18])-sum(sum(persondata_month_Pre[:,15:20]))/5)+abs(sum(persondata_month_Pre[:,19])-sum(sum(persondata_month_Pre[:,15:20]))/5)                     
    meandata_month_Pre[t,4] = abs(sum(persondata_month_Pre[:,20])-sum(sum(persondata_month_Pre[:,20:25]))/5)+abs(sum(persondata_month_Pre[:,21])-sum(sum(persondata_month_Pre[:,20:25]))/5)+abs(sum(persondata_month_Pre[:,22])-sum(sum(persondata_month_Pre[:,20:25]))/5)+abs(sum(persondata_month_Pre[:,23])-sum(sum(persondata_month_Pre[:,20:25]))/5)+abs(sum(persondata_month_Pre[:,24])-sum(sum(persondata_month_Pre[:,20:25]))/5)                     
    meandata_month_Pre[t,5] = abs(sum(persondata_month_Pre[:,25])-sum(sum(persondata_month_Pre[:,25:30]))/5)+abs(sum(persondata_month_Pre[:,26])-sum(sum(persondata_month_Pre[:,25:30]))/5)+abs(sum(persondata_month_Pre[:,27])-sum(sum(persondata_month_Pre[:,25:30]))/5)+abs(sum(persondata_month_Pre[:,28])-sum(sum(persondata_month_Pre[:,25:30]))/5)+abs(sum(persondata_month_Pre[:,29])-sum(sum(persondata_month_Pre[:,25:30]))/5)                     
    meandata_month_Pre[t,6] = abs(sum(persondata_month_Pre[:,30])-sum(sum(persondata_month_Pre[:,30:35]))/5)+abs(sum(persondata_month_Pre[:,31])-sum(sum(persondata_month_Pre[:,30:35]))/5)+abs(sum(persondata_month_Pre[:,32])-sum(sum(persondata_month_Pre[:,30:35]))/5)+abs(sum(persondata_month_Pre[:,33])-sum(sum(persondata_month_Pre[:,30:35]))/5)+abs(sum(persondata_month_Pre[:,34])-sum(sum(persondata_month_Pre[:,30:35]))/5)                     
    meandata_month_Pre[t,7] = abs(sum(persondata_month_Pre[:,35])-sum(sum(persondata_month_Pre[:,35:40]))/5)+abs(sum(persondata_month_Pre[:,36])-sum(sum(persondata_month_Pre[:,35:40]))/5)+abs(sum(persondata_month_Pre[:,37])-sum(sum(persondata_month_Pre[:,35:40]))/5)+abs(sum(persondata_month_Pre[:,38])-sum(sum(persondata_month_Pre[:,35:40]))/5)+abs(sum(persondata_month_Pre[:,39])-sum(sum(persondata_month_Pre[:,35:40]))/5)                     
    meandata_month_Pre[t,8] = abs(sum(persondata_month_Pre[:,40])-sum(sum(persondata_month_Pre[:,40:45]))/5)+abs(sum(persondata_month_Pre[:,41])-sum(sum(persondata_month_Pre[:,40:45]))/5)+abs(sum(persondata_month_Pre[:,42])-sum(sum(persondata_month_Pre[:,40:45]))/5)+abs(sum(persondata_month_Pre[:,43])-sum(sum(persondata_month_Pre[:,40:45]))/5)+abs(sum(persondata_month_Pre[:,44])-sum(sum(persondata_month_Pre[:,40:45]))/5)                     
    meandata_month_Pre[t,9] = abs(sum(persondata_month_Pre[:,45])-sum(sum(persondata_month_Pre[:,45:50]))/5)+abs(sum(persondata_month_Pre[:,46])-sum(sum(persondata_month_Pre[:,45:50]))/5)+abs(sum(persondata_month_Pre[:,47])-sum(sum(persondata_month_Pre[:,45:50]))/5)+abs(sum(persondata_month_Pre[:,48])-sum(sum(persondata_month_Pre[:,45:50]))/5)+abs(sum(persondata_month_Pre[:,49])-sum(sum(persondata_month_Pre[:,45:50]))/5)                     
    meandata_month_Pre[t,10] = abs(sum(persondata_month_Pre[:,50])-sum(sum(persondata_month_Pre[:,50:55]))/5)+abs(sum(persondata_month_Pre[:,51])-sum(sum(persondata_month_Pre[:,50:55]))/5)+abs(sum(persondata_month_Pre[:,52])-sum(sum(persondata_month_Pre[:,50:55]))/5)+abs(sum(persondata_month_Pre[:,53])-sum(sum(persondata_month_Pre[:,50:55]))/5)+abs(sum(persondata_month_Pre[:,54])-sum(sum(persondata_month_Pre[:,50:55]))/5)                     
    meandata_month_Pre[t,11] = abs(sum(persondata_month_Pre[:,55])-sum(sum(persondata_month_Pre[:,55:60]))/5)+abs(sum(persondata_month_Pre[:,56])-sum(sum(persondata_month_Pre[:,55:60]))/5)+abs(sum(persondata_month_Pre[:,57])-sum(sum(persondata_month_Pre[:,55:60]))/5)+abs(sum(persondata_month_Pre[:,58])-sum(sum(persondata_month_Pre[:,55:60]))/5)+abs(sum(persondata_month_Pre[:,59])-sum(sum(persondata_month_Pre[:,55:60]))/5)                     
    persondata_month_Pre = np.zeros((1000,persons*12))
    sumi = 0
    counterjob = 0
    passingjob = 0
    counter = 0
    alter = df.iloc[0,1]
    persongetdata[0,:] = 0
    
 
#Random with predictors
df = pd.read_excel('C:/Users/cmos/Desktop/New folder (2)/Predictor2.xlsx',encoding='latin-1')
df = df.iloc[:,2:]
jobs = df.shape[0]
persons = 5
rnadd = 0
sumi = 0
condition1 = 0
persondata_month_Pre = np.zeros((1000,persons*12))
persongetdata = np.zeros((1,persons))
meandata_month_Pre_C = np.zeros((100,12))
counterjob = 0
passingjob = 0
counter = 0
alter = df.iloc[0,1]
for t in range(100):
    print(t)
    while counterjob<jobs:
        #print(counterjob)
        passingjob = passingjob + counter
        counter = 0
        for i in range(passingjob,jobs):
            if alter == df.iloc[i,1]:
                counter = counter + 1
            else:
                persongetdata[0,:] = 0
                alter = df.iloc[i,1]
                break
        for i in range(passingjob,passingjob+counter):
            condition1 = 0
            if sum(sum(persongetdata)) == persons:
                persongetdata[0,:] = 0
            if df.iloc[i,3] == 3:
                while condition1 == 0:
                    rnadd = np.random.randint(0,persons)
                    if persongetdata[0,rnadd] == 0:
                        for j in range(len(persondata_month_Pre)):
                            if persondata_month_Pre[j,(int(df.iloc[i,0])-1)*5 + rnadd] == 0:
                                persondata_month_Pre[j,(int(df.iloc[i,0])-1)*5 + rnadd] = df.iloc[i,2]
                                #persondata[i,rnadd+5] = d
                                sumi = sumi + df.iloc[i,2]
                                counterjob = counterjob + 1
                                break
                        persongetdata[0,rnadd] = 1
                        condition1 = 1
        for i in range(passingjob,passingjob+counter):
            condition1 = 0
            if sum(sum(persongetdata)) == persons:
                persongetdata[0,:] = 0
            if df.iloc[i,3] == 2:
                while condition1 == 0:
                    rnadd = np.random.randint(0,persons)
                    if persongetdata[0,rnadd] == 0:
                        for j in range(len(persondata_month_Pre)):
                            if persondata_month_Pre[j,(int(df.iloc[i,0])-1)*5 + rnadd] == 0:
                                persondata_month_Pre[j,(int(df.iloc[i,0])-1)*5 + rnadd] = df.iloc[i,2]
                                #persondata[i,rnadd+5] = d
                                sumi = sumi + df.iloc[i,2]
                                counterjob = counterjob + 1
                                break
                        persongetdata[0,rnadd] = 1
                        condition1 = 1
                        break
        for i in range(passingjob,passingjob+counter):
            condition1 = 0
            if sum(sum(persongetdata)) == persons:
                persongetdata[0,:] = 0
            if df.iloc[i,3] == 1:
                while condition1 == 0:
                    rnadd = np.random.randint(0,persons)
                    if persongetdata[0,rnadd] == 0:
                        for j in range(len(persondata_month_Pre)):
                            if persondata_month_Pre[j,(int(df.iloc[i,0])-1)*5 + rnadd] == 0:
                                persondata_month_Pre[j,(int(df.iloc[i,0])-1)*5 + rnadd] = df.iloc[i,2]
                                #persondata[i,rnadd+5] = d
                                sumi = sumi + df.iloc[i,2]
                                counterjob = counterjob + 1
                                break
                        persongetdata[0,rnadd] = 1
                        condition1 = 1
                              
    meandata_month_Pre_C[t,0] = abs(sum(persondata_month_Pre[:,0])-sum(sum(persondata_month_Pre[:,:5]))/5)+abs(sum(persondata_month_Pre[:,1])-sum(sum(persondata_month_Pre[:,:5]))/5)+abs(sum(persondata_month_Pre[:,2])-sum(sum(persondata_month_Pre[:,:5]))/5)+abs(sum(persondata_month_Pre[:,3])-sum(sum(persondata_month_Pre[:,:5]))/5)+abs(sum(persondata_month_Pre[:,4])-sum(sum(persondata_month_Pre[:,:5]))/5)                     
    meandata_month_Pre_C[t,1] = abs(sum(persondata_month_Pre[:,5])-sum(sum(persondata_month_Pre[:,5:10]))/5)+abs(sum(persondata_month_Pre[:,6])-sum(sum(persondata_month_Pre[:,5:10]))/5)+abs(sum(persondata_month_Pre[:,7])-sum(sum(persondata_month_Pre[:,5:10]))/5)+abs(sum(persondata_month_Pre[:,8])-sum(sum(persondata_month_Pre[:,5:10]))/5)+abs(sum(persondata_month_Pre[:,9])-sum(sum(persondata_month_Pre[:,5:10]))/5)                     
    meandata_month_Pre_C[t,2] = abs(sum(persondata_month_Pre[:,10])-sum(sum(persondata_month_Pre[:,10:15]))/5)+abs(sum(persondata_month_Pre[:,11])-sum(sum(persondata_month_Pre[:,10:15]))/5)+abs(sum(persondata_month_Pre[:,12])-sum(sum(persondata_month_Pre[:,10:15]))/5)+abs(sum(persondata_month_Pre[:,13])-sum(sum(persondata_month_Pre[:,10:15]))/5)+abs(sum(persondata_month_Pre[:,14])-sum(sum(persondata_month_Pre[:,10:15]))/5)                     
    meandata_month_Pre_C[t,3] = abs(sum(persondata_month_Pre[:,15])-sum(sum(persondata_month_Pre[:,15:20]))/5)+abs(sum(persondata_month_Pre[:,16])-sum(sum(persondata_month_Pre[:,15:20]))/5)+abs(sum(persondata_month_Pre[:,17])-sum(sum(persondata_month_Pre[:,15:20]))/5)+abs(sum(persondata_month_Pre[:,18])-sum(sum(persondata_month_Pre[:,15:20]))/5)+abs(sum(persondata_month_Pre[:,19])-sum(sum(persondata_month_Pre[:,15:20]))/5)                     
    meandata_month_Pre_C[t,4] = abs(sum(persondata_month_Pre[:,20])-sum(sum(persondata_month_Pre[:,20:25]))/5)+abs(sum(persondata_month_Pre[:,21])-sum(sum(persondata_month_Pre[:,20:25]))/5)+abs(sum(persondata_month_Pre[:,22])-sum(sum(persondata_month_Pre[:,20:25]))/5)+abs(sum(persondata_month_Pre[:,23])-sum(sum(persondata_month_Pre[:,20:25]))/5)+abs(sum(persondata_month_Pre[:,24])-sum(sum(persondata_month_Pre[:,20:25]))/5)                     
    meandata_month_Pre_C[t,5] = abs(sum(persondata_month_Pre[:,25])-sum(sum(persondata_month_Pre[:,25:30]))/5)+abs(sum(persondata_month_Pre[:,26])-sum(sum(persondata_month_Pre[:,25:30]))/5)+abs(sum(persondata_month_Pre[:,27])-sum(sum(persondata_month_Pre[:,25:30]))/5)+abs(sum(persondata_month_Pre[:,28])-sum(sum(persondata_month_Pre[:,25:30]))/5)+abs(sum(persondata_month_Pre[:,29])-sum(sum(persondata_month_Pre[:,25:30]))/5)                     
    meandata_month_Pre_C[t,6] = abs(sum(persondata_month_Pre[:,30])-sum(sum(persondata_month_Pre[:,30:35]))/5)+abs(sum(persondata_month_Pre[:,31])-sum(sum(persondata_month_Pre[:,30:35]))/5)+abs(sum(persondata_month_Pre[:,32])-sum(sum(persondata_month_Pre[:,30:35]))/5)+abs(sum(persondata_month_Pre[:,33])-sum(sum(persondata_month_Pre[:,30:35]))/5)+abs(sum(persondata_month_Pre[:,34])-sum(sum(persondata_month_Pre[:,30:35]))/5)                     
    meandata_month_Pre_C[t,7] = abs(sum(persondata_month_Pre[:,35])-sum(sum(persondata_month_Pre[:,35:40]))/5)+abs(sum(persondata_month_Pre[:,36])-sum(sum(persondata_month_Pre[:,35:40]))/5)+abs(sum(persondata_month_Pre[:,37])-sum(sum(persondata_month_Pre[:,35:40]))/5)+abs(sum(persondata_month_Pre[:,38])-sum(sum(persondata_month_Pre[:,35:40]))/5)+abs(sum(persondata_month_Pre[:,39])-sum(sum(persondata_month_Pre[:,35:40]))/5)                     
    meandata_month_Pre_C[t,8] = abs(sum(persondata_month_Pre[:,40])-sum(sum(persondata_month_Pre[:,40:45]))/5)+abs(sum(persondata_month_Pre[:,41])-sum(sum(persondata_month_Pre[:,40:45]))/5)+abs(sum(persondata_month_Pre[:,42])-sum(sum(persondata_month_Pre[:,40:45]))/5)+abs(sum(persondata_month_Pre[:,43])-sum(sum(persondata_month_Pre[:,40:45]))/5)+abs(sum(persondata_month_Pre[:,44])-sum(sum(persondata_month_Pre[:,40:45]))/5)                     
    meandata_month_Pre_C[t,9] = abs(sum(persondata_month_Pre[:,45])-sum(sum(persondata_month_Pre[:,45:50]))/5)+abs(sum(persondata_month_Pre[:,46])-sum(sum(persondata_month_Pre[:,45:50]))/5)+abs(sum(persondata_month_Pre[:,47])-sum(sum(persondata_month_Pre[:,45:50]))/5)+abs(sum(persondata_month_Pre[:,48])-sum(sum(persondata_month_Pre[:,45:50]))/5)+abs(sum(persondata_month_Pre[:,49])-sum(sum(persondata_month_Pre[:,45:50]))/5)                     
    meandata_month_Pre_C[t,10] = abs(sum(persondata_month_Pre[:,50])-sum(sum(persondata_month_Pre[:,50:55]))/5)+abs(sum(persondata_month_Pre[:,51])-sum(sum(persondata_month_Pre[:,50:55]))/5)+abs(sum(persondata_month_Pre[:,52])-sum(sum(persondata_month_Pre[:,50:55]))/5)+abs(sum(persondata_month_Pre[:,53])-sum(sum(persondata_month_Pre[:,50:55]))/5)+abs(sum(persondata_month_Pre[:,54])-sum(sum(persondata_month_Pre[:,50:55]))/5)                     
    meandata_month_Pre_C[t,11] = abs(sum(persondata_month_Pre[:,55])-sum(sum(persondata_month_Pre[:,55:60]))/5)+abs(sum(persondata_month_Pre[:,56])-sum(sum(persondata_month_Pre[:,55:60]))/5)+abs(sum(persondata_month_Pre[:,57])-sum(sum(persondata_month_Pre[:,55:60]))/5)+abs(sum(persondata_month_Pre[:,58])-sum(sum(persondata_month_Pre[:,55:60]))/5)+abs(sum(persondata_month_Pre[:,59])-sum(sum(persondata_month_Pre[:,55:60]))/5)                     
    persondata_month_Pre = np.zeros((1000,persons*12))
    sumi = 0
    counterjob = 0
    passingjob = 0
    counter = 0
    alter = df.iloc[0,1]
    persongetdata[0,:] = 0
    
comperationn = np.zeros((12,3))
np.mean(meandata_month_Pre_C[:,0])
np.mean(meandata_month_Pre[:,0])
for i in range(12):
    comperationn[i,0] = np.mean(meandata_month_Pre[:,i])
    comperationn[i,1] = np.mean(meandata_month_Pre_C[:,i])   
    comperationn[i,2] = e[i]
comperationn[:,:2] = comperationn[:,:2]/5
e = [15440000,23620000,21520000,18440000,23080000,25180000,35580000,25400000,34000000,40560000,38820000,31920000]
comperationn = pd.DataFrame(comperationn)  
df_plot=pd.DataFrame({'month': range(1,13), 'Classifier Pre': comperationn.iloc[:,0]/comperationn.iloc[:,2]*100, 'Classifier Pre 100': comperationn.iloc[:,1]/comperationn.iloc[:,2]*100, 'Optimal': comperationFF.iloc[:,1]/persons/comperationFF.iloc[:,4]*100})
plt.figure(figsize=(16, 8))
plt.plot( 'month', 'Classifier Pre', data=df_plot, marker='o', markerfacecolor='blue', markersize=12, color='skyblue', linewidth=4)
plt.plot( 'month', 'Classifier Pre 100', data=df_plot, marker='o', color='olive', linewidth=3) 
plt.plot( 'month', 'Optimal', data=df_plot, marker='o', color='red', linewidth=3)
plt.title("Error percentage of Justice chart for comparing Classifier Predictor",size = 18)
plt.xlabel('Month',size = 18)
plt.ylabel('percentage',size = 18)
plt.grid()
plt.legend()



#Month1_Random
df = pd.read_excel('C:/Users/cmos/Desktop/New folder (2)/Month1.xlsx',encoding='latin-1')
df = df.iloc[:,1:]
days = df.shape[1]
jobs = df.shape[0]
persons = 5
rnadd = 0
sumi = 0
condition1 = 0
persondata = np.zeros((1000,persons+5))
persongetdata = np.zeros((1,persons))
meandata = np.zeros((1000,1))
for t in range(1000):
    print(t)
    for d in range(days):
        persongetdata[0,:] = 0
        for j in range(jobs - sum(df.iloc[:,d].isna())):
            condition1 = 0
            if sum(sum(persongetdata)) == persons:
                    persongetdata[0,:] = 0
            while condition1 == 0:
                rnadd = np.random.randint(0,persons)
                if persongetdata[0,rnadd] == 0:
                    for i in range(len(persondata)):
                        if persondata[i,rnadd] == 0:
                            persondata[i,rnadd] = df.iloc[j,d]
                            persondata[i,rnadd+5] = d
                            sumi = sumi + df.iloc[j,d]
                            break
                    persongetdata[0,rnadd] = 1
                    condition1 = 1
    meandata[t,0] = abs(sum(persondata[:,0])-sumi/5)+abs(sum(persondata[:,1])-sumi/5)+abs(sum(persondata[:,2])-sumi/5)+abs(sum(persondata[:,3])-sumi/5)+abs(sum(persondata[:,4])-sumi/5)                     
    persondata = np.zeros((1000,persons+5))
    sumi = 0
    persongetdata[0,:] = 0

plt.figure(figsize=(6, 6))
ax = sns.boxplot(data=meandata)
ax = sns.stripplot(data=meandata, color="orange", jitter=0.2, size=2.5)
plt.title("Boxplot of Methods", loc="right")
plt.grid(linestyle='-', linewidth=0.5)

print('Mean month 1 of random \n',np.mean(meandata))           

#Month1_Part2
df = pd.read_excel('C:/Users/cmos/Desktop/New folder (2)/Month1_1.xlsx',encoding='latin-1')
df = df.iloc[:,1:]
df1 = pd.read_excel('C:/Users/cmos/Desktop/New folder (2)/Part2_month1.xlsx',encoding='latin-1')
x_lag = 2
y_lag = 1               
days1 = df1.shape[0] 
jobs1 = df1.shape[1] 
persons = 5
rnadd = 0
sumi = 0
condition1 = 0
persondata1 = np.zeros((1000,persons))
for j in range(x_lag,jobs1):
    print(j)
    for d in range(y_lag,days1):
        print(d)
        if df1.iloc[d,j] == 1:
            for i in range(1000):
                print(i)
                if persondata1[i,int((df1.iloc[d,1])-1)] == 0:
                    print(i)
                    persondata1[i,(int(df1.iloc[d,1]))-1] = df.iloc[(int(df1.iloc[0,j]))-1,(int(df1.iloc[d,0])-1)]
                    sumi = sumi + df.iloc[(int(df1.iloc[0,j]))-1,(int(df1.iloc[d,0])-1)]
                    break
print('Mean month 1 of random \n',np.mean(meandata))  
print('Optimal solution for month 1: \n',abs(sum(persondata1[:,0])-sumi/5)+abs(sum(persondata1[:,1])-sumi/5)+abs(sum(persondata1[:,2])-sumi/5)+abs(sum(persondata1[:,3])-sumi/5)+abs(sum(persondata1[:,4])-sumi/5) )
comperation[0,0] = np.mean(meandata)
comperation[0,1] = abs(sum(persondata1[:,0])-sumi/5)+abs(sum(persondata1[:,1])-sumi/5)+abs(sum(persondata1[:,2])-sumi/5)+abs(sum(persondata1[:,3])-sumi/5)+abs(sum(persondata1[:,4])-sumi/5)

#Month2_Random
df = pd.read_excel('C:/Users/cmos/Desktop/New folder (2)/Month2.xlsx',encoding='latin-1')
df = df.iloc[:,1:]
days = df.shape[1]
jobs = df.shape[0]
persons = 5
rnadd = 0
sumi = 0
condition1 = 0
persondata = np.zeros((1000,persons+5))
persongetdata = np.zeros((1,persons))
meandata = np.zeros((1000,1))
for t in range(1000):
    print(t)
    for d in range(days):
        persongetdata[0,:] = 0
        for j in range(jobs - sum(df.iloc[:,d].isna())):
            condition1 = 0
            if sum(sum(persongetdata)) == persons:
                    persongetdata[0,:] = 0
            while condition1 == 0:
                rnadd = np.random.randint(0,persons)
                if persongetdata[0,rnadd] == 0:
                    for i in range(len(persondata)):
                        if persondata[i,rnadd] == 0:
                            persondata[i,rnadd] = df.iloc[j,d]
                            persondata[i,rnadd+5] = d
                            sumi = sumi + df.iloc[j,d]
                            break
                    persongetdata[0,rnadd] = 1
                    condition1 = 1
    meandata[t,0] = abs(sum(persondata[:,0])-sumi/5)+abs(sum(persondata[:,1])-sumi/5)+abs(sum(persondata[:,2])-sumi/5)+abs(sum(persondata[:,3])-sumi/5)+abs(sum(persondata[:,4])-sumi/5)                     
    persondata = np.zeros((1000,persons+5))
    sumi = 0
    persongetdata[0,:] = 0

plt.figure(figsize=(6, 6))
ax = sns.boxplot(data=meandata)
ax = sns.stripplot(data=meandata, color="orange", jitter=0.2, size=2.5)
plt.title("Boxplot of Methods", loc="right")
plt.grid(linestyle='-', linewidth=0.5)

print('Mean month 2 of random \n',np.mean(meandata))           

#Month2_Part2
df = pd.read_excel('C:/Users/cmos/Desktop/New folder (2)/Month2_2.xlsx',encoding='latin-1')
df = df.iloc[:,1:]
df1 = pd.read_excel('C:/Users/cmos/Desktop/New folder (2)/Part2_month2.xlsx',encoding='latin-1')
x_lag = 2
y_lag = 1               
days1 = df1.shape[0] 
jobs1 = df1.shape[1] 
persons = 5
rnadd = 0
sumi = 0
condition1 = 0
persondata1 = np.zeros((1000,persons))
for j in range(x_lag,jobs1):
    print(j)
    for d in range(y_lag,days1):
        print(d)
        if df1.iloc[d,j] == 1:
            for i in range(1000):
                print(i)
                if persondata1[i,int((df1.iloc[d,1])-1)] == 0:
                    print(i)
                    persondata1[i,(int(df1.iloc[d,1]))-1] = df.iloc[(int(df1.iloc[0,j]))-1,(int(df1.iloc[d,0])-1)]
                    sumi = sumi + df.iloc[(int(df1.iloc[0,j]))-1,(int(df1.iloc[d,0])-1)]
                    break
print('Mean month 2 of random \n',np.mean(meandata))  
print('Optimal solution for month 2: \n',abs(sum(persondata1[:,0])-sumi/5)+abs(sum(persondata1[:,1])-sumi/5)+abs(sum(persondata1[:,2])-sumi/5)+abs(sum(persondata1[:,3])-sumi/5)+abs(sum(persondata1[:,4])-sumi/5) )
comperation[1,0] = np.mean(meandata)
comperation[1,1] = abs(sum(persondata1[:,0])-sumi/5)+abs(sum(persondata1[:,1])-sumi/5)+abs(sum(persondata1[:,2])-sumi/5)+abs(sum(persondata1[:,3])-sumi/5)+abs(sum(persondata1[:,4])-sumi/5)

#Month3_Random
df = pd.read_excel('C:/Users/cmos/Desktop/New folder (2)/Month3.xlsx',encoding='latin-1')
df = df.iloc[:,1:]
days = df.shape[1]
jobs = df.shape[0]
persons = 5
rnadd = 0
sumi = 0
condition1 = 0
persondata = np.zeros((1000,persons+5))
persongetdata = np.zeros((1,persons))
meandata = np.zeros((1000,1))
for t in range(1000):
    print(t)
    for d in range(days):
        persongetdata[0,:] = 0
        for j in range(jobs - sum(df.iloc[:,d].isna())):
            condition1 = 0
            if sum(sum(persongetdata)) == persons:
                    persongetdata[0,:] = 0
            while condition1 == 0:
                rnadd = np.random.randint(0,persons)
                if persongetdata[0,rnadd] == 0:
                    for i in range(len(persondata)):
                        if persondata[i,rnadd] == 0:
                            persondata[i,rnadd] = df.iloc[j,d]
                            persondata[i,rnadd+5] = d
                            sumi = sumi + df.iloc[j,d]
                            break
                    persongetdata[0,rnadd] = 1
                    condition1 = 1
    meandata[t,0] = abs(sum(persondata[:,0])-sumi/5)+abs(sum(persondata[:,1])-sumi/5)+abs(sum(persondata[:,2])-sumi/5)+abs(sum(persondata[:,3])-sumi/5)+abs(sum(persondata[:,4])-sumi/5)                     
    persondata = np.zeros((1000,persons+5))
    sumi = 0
    persongetdata[0,:] = 0

plt.figure(figsize=(6, 6))
ax = sns.boxplot(data=meandata)
ax = sns.stripplot(data=meandata, color="orange", jitter=0.2, size=2.5)
plt.title("Boxplot of Methods", loc="right")
plt.grid(linestyle='-', linewidth=0.5)

print('Mean month 2 of random \n',np.mean(meandata))           

#Month3_Part2
df = pd.read_excel('C:/Users/cmos/Desktop/New folder (2)/Month3_3.xlsx',encoding='latin-1')
df = df.iloc[:,1:]
df1 = pd.read_excel('C:/Users/cmos/Desktop/New folder (2)/Part2_month3.xlsx',encoding='latin-1')
x_lag = 2
y_lag = 1               
days1 = df1.shape[0] 
jobs1 = df1.shape[1] 
persons = 5
rnadd = 0
sumi = 0
condition1 = 0
persondata1 = np.zeros((1000,persons))
for j in range(x_lag,jobs1):
    print(j)
    for d in range(y_lag,days1):
        print(d)
        if df1.iloc[d,j] == 1:
            for i in range(1000):
                print(i)
                if persondata1[i,int((df1.iloc[d,1])-1)] == 0:
                    print(i)
                    persondata1[i,(int(df1.iloc[d,1]))-1] = df.iloc[(int(df1.iloc[0,j]))-1,(int(df1.iloc[d,0])-1)]
                    sumi = sumi + df.iloc[(int(df1.iloc[0,j]))-1,(int(df1.iloc[d,0])-1)]
                    break
print('Mean month 3 of random \n',np.mean(meandata))  
print('Optimal solution for month 3: \n',abs(sum(persondata1[:,0])-sumi/5)+abs(sum(persondata1[:,1])-sumi/5)+abs(sum(persondata1[:,2])-sumi/5)+abs(sum(persondata1[:,3])-sumi/5)+abs(sum(persondata1[:,4])-sumi/5) )
comperation[2,0] = np.mean(meandata)
comperation[2,1] = abs(sum(persondata1[:,0])-sumi/5)+abs(sum(persondata1[:,1])-sumi/5)+abs(sum(persondata1[:,2])-sumi/5)+abs(sum(persondata1[:,3])-sumi/5)+abs(sum(persondata1[:,4])-sumi/5)

#Month4_Random
df = pd.read_excel('C:/Users/cmos/Desktop/New folder (2)/Month4.xlsx',encoding='latin-1')
df = df.iloc[:,1:]
days = df.shape[1]
jobs = df.shape[0]
persons = 5
rnadd = 0
sumi = 0
condition1 = 0
persondata = np.zeros((1000,persons+5))
persongetdata = np.zeros((1,persons))
meandata = np.zeros((1000,1))
for t in range(1000):
    print(t)
    for d in range(days):
        persongetdata[0,:] = 0
        for j in range(jobs - sum(df.iloc[:,d].isna())):
            condition1 = 0
            if sum(sum(persongetdata)) == persons:
                    persongetdata[0,:] = 0
            while condition1 == 0:
                rnadd = np.random.randint(0,persons)
                if persongetdata[0,rnadd] == 0:
                    for i in range(len(persondata)):
                        if persondata[i,rnadd] == 0:
                            persondata[i,rnadd] = df.iloc[j,d]
                            persondata[i,rnadd+5] = d
                            sumi = sumi + df.iloc[j,d]
                            break
                    persongetdata[0,rnadd] = 1
                    condition1 = 1
    meandata[t,0] = abs(sum(persondata[:,0])-sumi/5)+abs(sum(persondata[:,1])-sumi/5)+abs(sum(persondata[:,2])-sumi/5)+abs(sum(persondata[:,3])-sumi/5)+abs(sum(persondata[:,4])-sumi/5)                     
    persondata = np.zeros((1000,persons+5))
    sumi = 0
    persongetdata[0,:] = 0

plt.figure(figsize=(6, 6))
ax = sns.boxplot(data=meandata)
ax = sns.stripplot(data=meandata, color="orange", jitter=0.2, size=2.5)
plt.title("Boxplot of Methods", loc="right")
plt.grid(linestyle='-', linewidth=0.5)

print('Mean month 2 of random \n',np.mean(meandata))           

#Month4_Part2
df = pd.read_excel('C:/Users/cmos/Desktop/New folder (2)/Month4_4.xlsx',encoding='latin-1')
df = df.iloc[:,1:]
df1 = pd.read_excel('C:/Users/cmos/Desktop/New folder (2)/Part2_month4.xlsx',encoding='latin-1')
x_lag = 2
y_lag = 1               
days1 = df1.shape[0] 
jobs1 = df1.shape[1] 
persons = 5
rnadd = 0
sumi = 0
condition1 = 0
persondata1 = np.zeros((1000,persons))
for j in range(x_lag,jobs1):
    print(j)
    for d in range(y_lag,days1):
        print(d)
        if df1.iloc[d,j] == 1:
            for i in range(1000):
                print(i)
                if persondata1[i,int((df1.iloc[d,1])-1)] == 0:
                    print(i)
                    persondata1[i,(int(df1.iloc[d,1]))-1] = df.iloc[(int(df1.iloc[0,j]))-1,(int(df1.iloc[d,0])-1)]
                    sumi = sumi + df.iloc[(int(df1.iloc[0,j]))-1,(int(df1.iloc[d,0])-1)]
                    break
print('Mean month 4 of random \n',np.mean(meandata))  
print('Optimal solution for month 4: \n',abs(sum(persondata1[:,0])-sumi/5)+abs(sum(persondata1[:,1])-sumi/5)+abs(sum(persondata1[:,2])-sumi/5)+abs(sum(persondata1[:,3])-sumi/5)+abs(sum(persondata1[:,4])-sumi/5) )
comperation[3,0] = np.mean(meandata)
comperation[3,1] = abs(sum(persondata1[:,0])-sumi/5)+abs(sum(persondata1[:,1])-sumi/5)+abs(sum(persondata1[:,2])-sumi/5)+abs(sum(persondata1[:,3])-sumi/5)+abs(sum(persondata1[:,4])-sumi/5)

#Month5_Random
df = pd.read_excel('C:/Users/cmos/Desktop/New folder (2)/Month5.xlsx',encoding='latin-1')
df = df.iloc[:,1:]
days = df.shape[1]
jobs = df.shape[0]
persons = 5
rnadd = 0
sumi = 0
condition1 = 0
persondata = np.zeros((1000,persons+5))
persongetdata = np.zeros((1,persons))
meandata = np.zeros((1000,1))
for t in range(1000):
    print(t)
    for d in range(days):
        persongetdata[0,:] = 0
        for j in range(jobs - sum(df.iloc[:,d].isna())):
            condition1 = 0
            if sum(sum(persongetdata)) == persons:
                    persongetdata[0,:] = 0
            while condition1 == 0:
                rnadd = np.random.randint(0,persons)
                if persongetdata[0,rnadd] == 0:
                    for i in range(len(persondata)):
                        if persondata[i,rnadd] == 0:
                            persondata[i,rnadd] = df.iloc[j,d]
                            persondata[i,rnadd+5] = d
                            sumi = sumi + df.iloc[j,d]
                            break
                    persongetdata[0,rnadd] = 1
                    condition1 = 1
    meandata[t,0] = abs(sum(persondata[:,0])-sumi/5)+abs(sum(persondata[:,1])-sumi/5)+abs(sum(persondata[:,2])-sumi/5)+abs(sum(persondata[:,3])-sumi/5)+abs(sum(persondata[:,4])-sumi/5)                     
    persondata = np.zeros((1000,persons+5))
    sumi = 0
    persongetdata[0,:] = 0

plt.figure(figsize=(6, 6))
ax = sns.boxplot(data=meandata)
ax = sns.stripplot(data=meandata, color="orange", jitter=0.2, size=2.5)
plt.title("Boxplot of Methods", loc="right")
plt.grid(linestyle='-', linewidth=0.5)

print('Mean month 2 of random \n',np.mean(meandata))           

#Month5_Part2
df = pd.read_excel('C:/Users/cmos/Desktop/New folder (2)/Month5_5.xlsx',encoding='latin-1')
df = df.iloc[:,1:]
df1 = pd.read_excel('C:/Users/cmos/Desktop/New folder (2)/Part2_month5.xlsx',encoding='latin-1')
x_lag = 2
y_lag = 1               
days1 = df1.shape[0] 
jobs1 = df1.shape[1] 
persons = 5
rnadd = 0
sumi = 0
condition1 = 0
persondata1 = np.zeros((1000,persons))
for j in range(x_lag,jobs1):
    print(j)
    for d in range(y_lag,days1):
        print(d)
        if df1.iloc[d,j] == 1:
            for i in range(1000):
                print(i)
                if persondata1[i,int((df1.iloc[d,1])-1)] == 0:
                    print(i)
                    persondata1[i,(int(df1.iloc[d,1]))-1] = df.iloc[(int(df1.iloc[0,j]))-1,(int(df1.iloc[d,0])-1)]
                    sumi = sumi + df.iloc[(int(df1.iloc[0,j]))-1,(int(df1.iloc[d,0])-1)]
                    break
print('Mean month 5 of random \n',np.mean(meandata))  
print('Optimal solution for month 5: \n',abs(sum(persondata1[:,0])-sumi/5)+abs(sum(persondata1[:,1])-sumi/5)+abs(sum(persondata1[:,2])-sumi/5)+abs(sum(persondata1[:,3])-sumi/5)+abs(sum(persondata1[:,4])-sumi/5) )
comperation[4,0] = np.mean(meandata)
comperation[4,1] = abs(sum(persondata1[:,0])-sumi/5)+abs(sum(persondata1[:,1])-sumi/5)+abs(sum(persondata1[:,2])-sumi/5)+abs(sum(persondata1[:,3])-sumi/5)+abs(sum(persondata1[:,4])-sumi/5)

#Month6_Random
df = pd.read_excel('C:/Users/cmos/Desktop/New folder (2)/Month6.xlsx',encoding='latin-1')
df = df.iloc[:,1:]
days = df.shape[1]
jobs = df.shape[0]
persons = 5
rnadd = 0
sumi = 0
condition1 = 0
persondata = np.zeros((1000,persons+5))
persongetdata = np.zeros((1,persons))
meandata = np.zeros((1000,1))
for t in range(1000):
    print(t)
    for d in range(days):
        persongetdata[0,:] = 0
        for j in range(jobs - sum(df.iloc[:,d].isna())):
            condition1 = 0
            if sum(sum(persongetdata)) == persons:
                    persongetdata[0,:] = 0
            while condition1 == 0:
                rnadd = np.random.randint(0,persons)
                if persongetdata[0,rnadd] == 0:
                    for i in range(len(persondata)):
                        if persondata[i,rnadd] == 0:
                            persondata[i,rnadd] = df.iloc[j,d]
                            persondata[i,rnadd+5] = d
                            sumi = sumi + df.iloc[j,d]
                            break
                    persongetdata[0,rnadd] = 1
                    condition1 = 1
    meandata[t,0] = abs(sum(persondata[:,0])-sumi/5)+abs(sum(persondata[:,1])-sumi/5)+abs(sum(persondata[:,2])-sumi/5)+abs(sum(persondata[:,3])-sumi/5)+abs(sum(persondata[:,4])-sumi/5)                     
    persondata = np.zeros((1000,persons+5))
    sumi = 0
    persongetdata[0,:] = 0

plt.figure(figsize=(6, 6))
ax = sns.boxplot(data=meandata)
ax = sns.stripplot(data=meandata, color="orange", jitter=0.2, size=2.5)
plt.title("Boxplot of Methods", loc="right")
plt.grid(linestyle='-', linewidth=0.5)

print('Mean month 2 of random \n',np.mean(meandata))           

#Month6_Part2
df = pd.read_excel('C:/Users/cmos/Desktop/New folder (2)/Month6_6.xlsx',encoding='latin-1')
df = df.iloc[:,1:]
df1 = pd.read_excel('C:/Users/cmos/Desktop/New folder (2)/Part2_month6.xlsx',encoding='latin-1')
x_lag = 2
y_lag = 1               
days1 = df1.shape[0] 
jobs1 = df1.shape[1] 
persons = 5
rnadd = 0
sumi = 0
condition1 = 0
persondata1 = np.zeros((1000,persons))
for j in range(x_lag,jobs1):
    print(j)
    for d in range(y_lag,days1):
        print(d)
        if df1.iloc[d,j] == 1:
            for i in range(1000):
                print(i)
                if persondata1[i,int((df1.iloc[d,1])-1)] == 0:
                    print(i)
                    persondata1[i,(int(df1.iloc[d,1]))-1] = df.iloc[(int(df1.iloc[0,j]))-1,(int(df1.iloc[d,0])-1)]
                    sumi = sumi + df.iloc[(int(df1.iloc[0,j]))-1,(int(df1.iloc[d,0])-1)]
                    break
print('Mean month 6 of random \n',np.mean(meandata))  
print('Optimal solution for month 6: \n',abs(sum(persondata1[:,0])-sumi/5)+abs(sum(persondata1[:,1])-sumi/5)+abs(sum(persondata1[:,2])-sumi/5)+abs(sum(persondata1[:,3])-sumi/5)+abs(sum(persondata1[:,4])-sumi/5) )
comperation[5,0] = np.mean(meandata)
comperation[5,1] = abs(sum(persondata1[:,0])-sumi/5)+abs(sum(persondata1[:,1])-sumi/5)+abs(sum(persondata1[:,2])-sumi/5)+abs(sum(persondata1[:,3])-sumi/5)+abs(sum(persondata1[:,4])-sumi/5)

#Month7_Random
df = pd.read_excel('C:/Users/cmos/Desktop/New folder (2)/Month7.xlsx',encoding='latin-1')
df = df.iloc[:,1:]
days = df.shape[1]
jobs = df.shape[0]
persons = 5
rnadd = 0
sumi = 0
condition1 = 0
persondata = np.zeros((1000,persons+5))
persongetdata = np.zeros((1,persons))
meandata = np.zeros((1000,1))
for t in range(1000):
    print(t)
    for d in range(days):
        persongetdata[0,:] = 0
        for j in range(jobs - sum(df.iloc[:,d].isna())):
            condition1 = 0
            if sum(sum(persongetdata)) == persons:
                    persongetdata[0,:] = 0
            while condition1 == 0:
                rnadd = np.random.randint(0,persons)
                if persongetdata[0,rnadd] == 0:
                    for i in range(len(persondata)):
                        if persondata[i,rnadd] == 0:
                            persondata[i,rnadd] = df.iloc[j,d]
                            persondata[i,rnadd+5] = d
                            sumi = sumi + df.iloc[j,d]
                            break
                    persongetdata[0,rnadd] = 1
                    condition1 = 1
    meandata[t,0] = abs(sum(persondata[:,0])-sumi/5)+abs(sum(persondata[:,1])-sumi/5)+abs(sum(persondata[:,2])-sumi/5)+abs(sum(persondata[:,3])-sumi/5)+abs(sum(persondata[:,4])-sumi/5)                     
    persondata = np.zeros((1000,persons+5))
    sumi = 0
    persongetdata[0,:] = 0

plt.figure(figsize=(6, 6))
ax = sns.boxplot(data=meandata)
ax = sns.stripplot(data=meandata, color="orange", jitter=0.2, size=2.5)
plt.title("Boxplot of Methods", loc="right")
plt.grid(linestyle='-', linewidth=0.5)

print('Mean month 2 of random \n',np.mean(meandata))           

#Month7_Part2
df = pd.read_excel('C:/Users/cmos/Desktop/New folder (2)/Month7_7.xlsx',encoding='latin-1')
df = df.iloc[:,1:]
df1 = pd.read_excel('C:/Users/cmos/Desktop/New folder (2)/Part2_month7.xlsx',encoding='latin-1')
x_lag = 2
y_lag = 1               
days1 = df1.shape[0] 
jobs1 = df1.shape[1] 
persons = 5
rnadd = 0
sumi = 0
condition1 = 0
persondata1 = np.zeros((1000,persons))
for j in range(x_lag,jobs1):
    print(j)
    for d in range(y_lag,days1):
        print(d)
        if df1.iloc[d,j] == 1:
            for i in range(1000):
                print(i)
                if persondata1[i,int((df1.iloc[d,1])-1)] == 0:
                    print(i)
                    persondata1[i,(int(df1.iloc[d,1]))-1] = df.iloc[(int(df1.iloc[0,j]))-1,(int(df1.iloc[d,0])-1)]
                    sumi = sumi + df.iloc[(int(df1.iloc[0,j]))-1,(int(df1.iloc[d,0])-1)]
                    break
print('Mean month 7 of random \n',np.mean(meandata))  
print('Optimal solution for month 7: \n',abs(sum(persondata1[:,0])-sumi/5)+abs(sum(persondata1[:,1])-sumi/5)+abs(sum(persondata1[:,2])-sumi/5)+abs(sum(persondata1[:,3])-sumi/5)+abs(sum(persondata1[:,4])-sumi/5) )
comperation[6,0] = np.mean(meandata)
comperation[6,1] = abs(sum(persondata1[:,0])-sumi/5)+abs(sum(persondata1[:,1])-sumi/5)+abs(sum(persondata1[:,2])-sumi/5)+abs(sum(persondata1[:,3])-sumi/5)+abs(sum(persondata1[:,4])-sumi/5)

#Month8_Random
df = pd.read_excel('C:/Users/cmos/Desktop/New folder (2)/Month8.xlsx',encoding='latin-1')
df = df.iloc[:,1:]
days = df.shape[1]
jobs = df.shape[0]
persons = 5
rnadd = 0
sumi = 0
condition1 = 0
persondata = np.zeros((1000,persons+5))
persongetdata = np.zeros((1,persons))
meandata = np.zeros((1000,1))
for t in range(1000):
    print(t)
    for d in range(days):
        persongetdata[0,:] = 0
        for j in range(jobs - sum(df.iloc[:,d].isna())):
            condition1 = 0
            if sum(sum(persongetdata)) == persons:
                    persongetdata[0,:] = 0
            while condition1 == 0:
                rnadd = np.random.randint(0,persons)
                if persongetdata[0,rnadd] == 0:
                    for i in range(len(persondata)):
                        if persondata[i,rnadd] == 0:
                            persondata[i,rnadd] = df.iloc[j,d]
                            persondata[i,rnadd+5] = d
                            sumi = sumi + df.iloc[j,d]
                            break
                    persongetdata[0,rnadd] = 1
                    condition1 = 1
    meandata[t,0] = abs(sum(persondata[:,0])-sumi/5)+abs(sum(persondata[:,1])-sumi/5)+abs(sum(persondata[:,2])-sumi/5)+abs(sum(persondata[:,3])-sumi/5)+abs(sum(persondata[:,4])-sumi/5)                     
    persondata = np.zeros((1000,persons+5))
    sumi = 0
    persongetdata[0,:] = 0

plt.figure(figsize=(6, 6))
ax = sns.boxplot(data=meandata)
ax = sns.stripplot(data=meandata, color="orange", jitter=0.2, size=2.5)
plt.title("Boxplot of Methods", loc="right")
plt.grid(linestyle='-', linewidth=0.5)

print('Mean month 2 of random \n',np.mean(meandata))           

#Month8_Part2
df = pd.read_excel('C:/Users/cmos/Desktop/New folder (2)/Month8_8.xlsx',encoding='latin-1')
df = df.iloc[:,1:]
df1 = pd.read_excel('C:/Users/cmos/Desktop/New folder (2)/Part2_month8.xlsx',encoding='latin-1')
x_lag = 2
y_lag = 1               
days1 = df1.shape[0] 
jobs1 = df1.shape[1] 
persons = 5
rnadd = 0
sumi = 0
condition1 = 0
persondata1 = np.zeros((1000,persons))
for j in range(x_lag,jobs1):
    print(j)
    for d in range(y_lag,days1):
        print(d)
        if df1.iloc[d,j] == 1:
            for i in range(1000):
                print(i)
                if persondata1[i,int((df1.iloc[d,1])-1)] == 0:
                    print(i)
                    persondata1[i,(int(df1.iloc[d,1]))-1] = df.iloc[(int(df1.iloc[0,j]))-1,(int(df1.iloc[d,0])-1)]
                    sumi = sumi + df.iloc[(int(df1.iloc[0,j]))-1,(int(df1.iloc[d,0])-1)]
                    break
print('Mean month 5 of random \n',np.mean(meandata))  
print('Optimal solution for month 5: \n',abs(sum(persondata1[:,0])-sumi/5)+abs(sum(persondata1[:,1])-sumi/5)+abs(sum(persondata1[:,2])-sumi/5)+abs(sum(persondata1[:,3])-sumi/5)+abs(sum(persondata1[:,4])-sumi/5) )
comperation[7,0] = np.mean(meandata)
comperation[7,1] = abs(sum(persondata1[:,0])-sumi/5)+abs(sum(persondata1[:,1])-sumi/5)+abs(sum(persondata1[:,2])-sumi/5)+abs(sum(persondata1[:,3])-sumi/5)+abs(sum(persondata1[:,4])-sumi/5)


#Month9_Random
df = pd.read_excel('C:/Users/cmos/Desktop/New folder (2)/Month9.xlsx',encoding='latin-1')
df = df.iloc[:,1:]
days = df.shape[1]
jobs = df.shape[0]
persons = 5
rnadd = 0
sumi = 0
condition1 = 0
persondata = np.zeros((1000,persons+5))
persongetdata = np.zeros((1,persons))
meandata = np.zeros((1000,1))
for t in range(1000):
    print(t)
    for d in range(days):
        persongetdata[0,:] = 0
        for j in range(jobs - sum(df.iloc[:,d].isna())):
            condition1 = 0
            if sum(sum(persongetdata)) == persons:
                    persongetdata[0,:] = 0
            while condition1 == 0:
                rnadd = np.random.randint(0,persons)
                if persongetdata[0,rnadd] == 0:
                    for i in range(len(persondata)):
                        if persondata[i,rnadd] == 0:
                            persondata[i,rnadd] = df.iloc[j,d]
                            persondata[i,rnadd+5] = d
                            sumi = sumi + df.iloc[j,d]
                            break
                    persongetdata[0,rnadd] = 1
                    condition1 = 1
    meandata[t,0] = abs(sum(persondata[:,0])-sumi/5)+abs(sum(persondata[:,1])-sumi/5)+abs(sum(persondata[:,2])-sumi/5)+abs(sum(persondata[:,3])-sumi/5)+abs(sum(persondata[:,4])-sumi/5)                     
    persondata = np.zeros((1000,persons+5))
    sumi = 0
    persongetdata[0,:] = 0

plt.figure(figsize=(6, 6))
ax = sns.boxplot(data=meandata)
ax = sns.stripplot(data=meandata, color="orange", jitter=0.2, size=2.5)
plt.title("Boxplot of Methods", loc="right")
plt.grid(linestyle='-', linewidth=0.5)

print('Mean month 2 of random \n',np.mean(meandata))           

#Month9_Part2
df = pd.read_excel('C:/Users/cmos/Desktop/New folder (2)/Month9_9.xlsx',encoding='latin-1')
df = df.iloc[:,1:]
df1 = pd.read_excel('C:/Users/cmos/Desktop/New folder (2)/Part2_month9.xlsx',encoding='latin-1')
x_lag = 2
y_lag = 1               
days1 = df1.shape[0] 
jobs1 = df1.shape[1] 
persons = 5
rnadd = 0
sumi = 0
condition1 = 0
persondata1 = np.zeros((1000,persons))
for j in range(x_lag,jobs1):
    print(j)
    for d in range(y_lag,days1):
        print(d)
        if df1.iloc[d,j] == 1:
            for i in range(1000):
                print(i)
                if persondata1[i,int((df1.iloc[d,1])-1)] == 0:
                    print(i)
                    persondata1[i,(int(df1.iloc[d,1]))-1] = df.iloc[(int(df1.iloc[0,j]))-1,(int(df1.iloc[d,0])-1)]
                    sumi = sumi + df.iloc[(int(df1.iloc[0,j]))-1,(int(df1.iloc[d,0])-1)]
                    break
print('Mean month 9 of random \n',np.mean(meandata))  
print('Optimal solution for month 9: \n',abs(sum(persondata1[:,0])-sumi/5)+abs(sum(persondata1[:,1])-sumi/5)+abs(sum(persondata1[:,2])-sumi/5)+abs(sum(persondata1[:,3])-sumi/5)+abs(sum(persondata1[:,4])-sumi/5) )
comperation[8,0] = np.mean(meandata)
comperation[8,1] = abs(sum(persondata1[:,0])-sumi/5)+abs(sum(persondata1[:,1])-sumi/5)+abs(sum(persondata1[:,2])-sumi/5)+abs(sum(persondata1[:,3])-sumi/5)+abs(sum(persondata1[:,4])-sumi/5)


#Month10_Random
df = pd.read_excel('C:/Users/cmos/Desktop/New folder (2)/Month10.xlsx',encoding='latin-1')
df = df.iloc[:,1:]
days = df.shape[1]
jobs = df.shape[0]
persons = 5
rnadd = 0
sumi = 0
condition1 = 0
persondata = np.zeros((1000,persons+5))
persongetdata = np.zeros((1,persons))
meandata = np.zeros((1000,1))
for t in range(1000):
    print(t)
    for d in range(days):
        persongetdata[0,:] = 0
        for j in range(jobs - sum(df.iloc[:,d].isna())):
            condition1 = 0
            if sum(sum(persongetdata)) == persons:
                    persongetdata[0,:] = 0
            while condition1 == 0:
                rnadd = np.random.randint(0,persons)
                if persongetdata[0,rnadd] == 0:
                    for i in range(len(persondata)):
                        if persondata[i,rnadd] == 0:
                            persondata[i,rnadd] = df.iloc[j,d]
                            persondata[i,rnadd+5] = d
                            sumi = sumi + df.iloc[j,d]
                            break
                    persongetdata[0,rnadd] = 1
                    condition1 = 1
    meandata[t,0] = abs(sum(persondata[:,0])-sumi/5)+abs(sum(persondata[:,1])-sumi/5)+abs(sum(persondata[:,2])-sumi/5)+abs(sum(persondata[:,3])-sumi/5)+abs(sum(persondata[:,4])-sumi/5)                     
    persondata = np.zeros((1000,persons+5))
    sumi = 0
    persongetdata[0,:] = 0

plt.figure(figsize=(6, 6))
ax = sns.boxplot(data=meandata)
ax = sns.stripplot(data=meandata, color="orange", jitter=0.2, size=2.5)
plt.title("Boxplot of Methods", loc="right")
plt.grid(linestyle='-', linewidth=0.5)

print('Mean month 2 of random \n',np.mean(meandata))           

#Month10_Part2
df = pd.read_excel('C:/Users/cmos/Desktop/New folder (2)/Month10_10.xlsx',encoding='latin-1')
df = df.iloc[:,1:]
df1 = pd.read_excel('C:/Users/cmos/Desktop/New folder (2)/Part2_month10.xlsx',encoding='latin-1')
x_lag = 2
y_lag = 1               
days1 = df1.shape[0] 
jobs1 = df1.shape[1] 
persons = 5
rnadd = 0
sumi = 0
condition1 = 0
persondata1 = np.zeros((1000,persons))
for j in range(x_lag,jobs1):
    print(j)
    for d in range(y_lag,days1):
        print(d)
        if df1.iloc[d,j] == 1:
            for i in range(1000):
                print(i)
                if persondata1[i,int((df1.iloc[d,1])-1)] == 0:
                    print(i)
                    persondata1[i,(int(df1.iloc[d,1]))-1] = df.iloc[(int(df1.iloc[0,j]))-1,(int(df1.iloc[d,0])-1)]
                    sumi = sumi + df.iloc[(int(df1.iloc[0,j]))-1,(int(df1.iloc[d,0])-1)]
                    break
print('Mean month 10 of random \n',np.mean(meandata))  
print('Optimal solution for month 10: \n',abs(sum(persondata1[:,0])-sumi/5)+abs(sum(persondata1[:,1])-sumi/5)+abs(sum(persondata1[:,2])-sumi/5)+abs(sum(persondata1[:,3])-sumi/5)+abs(sum(persondata1[:,4])-sumi/5) )
comperation[9,0] = np.mean(meandata)
comperation[9,1] = abs(sum(persondata1[:,0])-sumi/5)+abs(sum(persondata1[:,1])-sumi/5)+abs(sum(persondata1[:,2])-sumi/5)+abs(sum(persondata1[:,3])-sumi/5)+abs(sum(persondata1[:,4])-sumi/5)


#Month11_Random
df = pd.read_excel('C:/Users/cmos/Desktop/New folder (2)/Month11.xlsx',encoding='latin-1')
df = df.iloc[:,1:]
days = df.shape[1]
jobs = df.shape[0]
persons = 5
rnadd = 0
sumi = 0
condition1 = 0
persondata = np.zeros((1000,persons+5))
persongetdata = np.zeros((1,persons))
meandata = np.zeros((1000,1))
for t in range(1000):
    print(t)
    for d in range(days):
        persongetdata[0,:] = 0
        for j in range(jobs - sum(df.iloc[:,d].isna())):
            condition1 = 0
            if sum(sum(persongetdata)) == persons:
                    persongetdata[0,:] = 0
            while condition1 == 0:
                rnadd = np.random.randint(0,persons)
                if persongetdata[0,rnadd] == 0:
                    for i in range(len(persondata)):
                        if persondata[i,rnadd] == 0:
                            persondata[i,rnadd] = df.iloc[j,d]
                            persondata[i,rnadd+5] = d
                            sumi = sumi + df.iloc[j,d]
                            break
                    persongetdata[0,rnadd] = 1
                    condition1 = 1
    meandata[t,0] = abs(sum(persondata[:,0])-sumi/5)+abs(sum(persondata[:,1])-sumi/5)+abs(sum(persondata[:,2])-sumi/5)+abs(sum(persondata[:,3])-sumi/5)+abs(sum(persondata[:,4])-sumi/5)                     
    persondata = np.zeros((1000,persons+5))
    sumi = 0
    persongetdata[0,:] = 0

plt.figure(figsize=(6, 6))
ax = sns.boxplot(data=meandata)
ax = sns.stripplot(data=meandata, color="orange", jitter=0.2, size=2.5)
plt.title("Boxplot of Methods", loc="right")
plt.grid(linestyle='-', linewidth=0.5)

print('Mean month 2 of random \n',np.mean(meandata))           

#Month11_Part2
df = pd.read_excel('C:/Users/cmos/Desktop/New folder (2)/Month11_11.xlsx',encoding='latin-1')
df = df.iloc[:,1:]
df1 = pd.read_excel('C:/Users/cmos/Desktop/New folder (2)/Part2_month11.xlsx',encoding='latin-1')
x_lag = 2
y_lag = 1               
days1 = df1.shape[0] 
jobs1 = df1.shape[1] 
persons = 5
rnadd = 0
sumi = 0
condition1 = 0
persondata1 = np.zeros((1000,persons))
for j in range(x_lag,jobs1):
    print(j)
    for d in range(y_lag,days1):
        print(d)
        if df1.iloc[d,j] == 1:
            for i in range(1000):
                print(i)
                if persondata1[i,int((df1.iloc[d,1])-1)] == 0:
                    print(i)
                    persondata1[i,(int(df1.iloc[d,1]))-1] = df.iloc[(int(df1.iloc[0,j]))-1,(int(df1.iloc[d,0])-1)]
                    sumi = sumi + df.iloc[(int(df1.iloc[0,j]))-1,(int(df1.iloc[d,0])-1)]
                    break
print('Mean month 11 of random \n',np.mean(meandata))  
print('Optimal solution for month 11: \n',abs(sum(persondata1[:,0])-sumi/5)+abs(sum(persondata1[:,1])-sumi/5)+abs(sum(persondata1[:,2])-sumi/5)+abs(sum(persondata1[:,3])-sumi/5)+abs(sum(persondata1[:,4])-sumi/5) )
comperation[10,0] = np.mean(meandata)
comperation[10,1] = abs(sum(persondata1[:,0])-sumi/5)+abs(sum(persondata1[:,1])-sumi/5)+abs(sum(persondata1[:,2])-sumi/5)+abs(sum(persondata1[:,3])-sumi/5)+abs(sum(persondata1[:,4])-sumi/5)


#Month12_Random
df = pd.read_excel('C:/Users/cmos/Desktop/New folder (2)/Month12.xlsx',encoding='latin-1')
df = df.iloc[:,1:]
days = df.shape[1]
jobs = df.shape[0]
persons = 5
rnadd = 0
sumi = 0
condition1 = 0
persondata = np.zeros((1000,persons+5))
persongetdata = np.zeros((1,persons))
meandata = np.zeros((1000,1))
for t in range(1000):
    print(t)
    for d in range(days):
        persongetdata[0,:] = 0
        for j in range(jobs - sum(df.iloc[:,d].isna())):
            condition1 = 0
            if sum(sum(persongetdata)) == persons:
                    persongetdata[0,:] = 0
            while condition1 == 0:
                rnadd = np.random.randint(0,persons)
                if persongetdata[0,rnadd] == 0:
                    for i in range(len(persondata)):
                        if persondata[i,rnadd] == 0:
                            persondata[i,rnadd] = df.iloc[j,d]
                            persondata[i,rnadd+5] = d
                            sumi = sumi + df.iloc[j,d]
                            break
                    persongetdata[0,rnadd] = 1
                    condition1 = 1
    meandata[t,0] = abs(sum(persondata[:,0])-sumi/5)+abs(sum(persondata[:,1])-sumi/5)+abs(sum(persondata[:,2])-sumi/5)+abs(sum(persondata[:,3])-sumi/5)+abs(sum(persondata[:,4])-sumi/5)                     
    persondata = np.zeros((1000,persons+5))
    sumi = 0
    persongetdata[0,:] = 0

plt.figure(figsize=(6, 6))
ax = sns.boxplot(data=meandata)
ax = sns.stripplot(data=meandata, color="orange", jitter=0.2, size=2.5)
plt.title("Boxplot of Methods", loc="right")
plt.grid(linestyle='-', linewidth=0.5)

print('Mean month 2 of random \n',np.mean(meandata))           

#Month12_Part2
df = pd.read_excel('C:/Users/cmos/Desktop/New folder (2)/Month12_12.xlsx',encoding='latin-1')
df = df.iloc[:,1:]
df1 = pd.read_excel('C:/Users/cmos/Desktop/New folder (2)/Part2_month12.xlsx',encoding='latin-1')
x_lag = 2
y_lag = 1               
days1 = df1.shape[0] 
jobs1 = df1.shape[1] 
persons = 5
rnadd = 0
sumi = 0
condition1 = 0
persondata1 = np.zeros((1000,persons))
for j in range(x_lag,jobs1):
    print(j)
    for d in range(y_lag,days1):
        print(d)
        if df1.iloc[d,j] == 1:
            for i in range(1000):
                print(i)
                if persondata1[i,int((df1.iloc[d,1])-1)] == 0:
                    print(i)
                    persondata1[i,(int(df1.iloc[d,1]))-1] = df.iloc[(int(df1.iloc[0,j]))-1,(int(df1.iloc[d,0])-1)]
                    sumi = sumi + df.iloc[(int(df1.iloc[0,j]))-1,(int(df1.iloc[d,0])-1)]
                    break
print('Mean month 12 of random \n',np.mean(meandata))  
print('Optimal solution for month 12: \n',abs(sum(persondata1[:,0])-sumi/5)+abs(sum(persondata1[:,1])-sumi/5)+abs(sum(persondata1[:,2])-sumi/5)+abs(sum(persondata1[:,3])-sumi/5)+abs(sum(persondata1[:,4])-sumi/5) )
comperation[11,0] = np.mean(meandata)
comperation[11,1] = abs(sum(persondata1[:,0])-sumi/5)+abs(sum(persondata1[:,1])-sumi/5)+abs(sum(persondata1[:,2])-sumi/5)+abs(sum(persondata1[:,3])-sumi/5)+abs(sum(persondata1[:,4])-sumi/5)








plt.figure(figsize=(6, 6))
ax = sns.boxplot(data=meandata)
ax = sns.stripplot(data=meandata, color="orange", jitter=0.2, size=2.5)
plt.title("Boxplot of Methods", loc="right")
plt.grid(linestyle='-', linewidth=0.5)

print('Mean month 2 of random \n',np.mean(meandata))  

sum(comperation[:,1])





#Month1_Part2
persondata1 = np.zeros((10000,persons))
df = pd.read_excel('C:/Users/cmos/Desktop/New folder (2)/Month1_1.xlsx',encoding='latin-1')
df = df.iloc[:,1:]
df1 = pd.read_excel('C:/Users/cmos/Desktop/New folder (2)/Part2_month1.xlsx',encoding='latin-1')
x_lag = 2
y_lag = 1               
days1 = df1.shape[0] 
jobs1 = df1.shape[1] 
persons = 5
rnadd = 0
sumi = 0
condition1 = 0
for j in range(x_lag,jobs1):
    print(j)
    for d in range(y_lag,days1):
        print(d)
        if df1.iloc[d,j] == 1:
            for i in range(10000):
                print(i)
                if persondata1[i,int((df1.iloc[d,1])-1)] == 0:
                    print(i)
                    persondata1[i,(int(df1.iloc[d,1]))-1] = df.iloc[(int(df1.iloc[0,j]))-1,(int(df1.iloc[d,0])-1)]
                    sumi = sumi + df.iloc[(int(df1.iloc[0,j]))-1,(int(df1.iloc[d,0])-1)]
                    break
#Month2_Part2
df = pd.read_excel('C:/Users/cmos/Desktop/New folder (2)/Month2_2.xlsx',encoding='latin-1')
df = df.iloc[:,1:]
df1 = pd.read_excel('C:/Users/cmos/Desktop/New folder (2)/Part2_month2.xlsx',encoding='latin-1')
x_lag = 2
y_lag = 1               
days1 = df1.shape[0] 
jobs1 = df1.shape[1] 
persons = 5
rnadd = 0
sumi = 0
condition1 = 0
for j in range(x_lag,jobs1):
    print(j)
    for d in range(y_lag,days1):
        print(d)
        if df1.iloc[d,j] == 1:
            for i in range(10000):
                print(i)
                if persondata1[i,int((df1.iloc[d,1])-1)] == 0:
                    print(i)
                    persondata1[i,(int(df1.iloc[d,1]))-1] = df.iloc[(int(df1.iloc[0,j]))-1,(int(df1.iloc[d,0])-1)]
                    sumi = sumi + df.iloc[(int(df1.iloc[0,j]))-1,(int(df1.iloc[d,0])-1)]
                    break
#Month3_Part2
df = pd.read_excel('C:/Users/cmos/Desktop/New folder (2)/Month3_3.xlsx',encoding='latin-1')
df = df.iloc[:,1:]
df1 = pd.read_excel('C:/Users/cmos/Desktop/New folder (2)/Part2_month3.xlsx',encoding='latin-1')
x_lag = 2
y_lag = 1               
days1 = df1.shape[0] 
jobs1 = df1.shape[1] 
persons = 5
rnadd = 0
sumi = 0
condition1 = 0
for j in range(x_lag,jobs1):
    print(j)
    for d in range(y_lag,days1):
        print(d)
        if df1.iloc[d,j] == 1:
            for i in range(10000):
                print(i)
                if persondata1[i,int((df1.iloc[d,1])-1)] == 0:
                    print(i)
                    persondata1[i,(int(df1.iloc[d,1]))-1] = df.iloc[(int(df1.iloc[0,j]))-1,(int(df1.iloc[d,0])-1)]
                    sumi = sumi + df.iloc[(int(df1.iloc[0,j]))-1,(int(df1.iloc[d,0])-1)]
                    break
#Month4_Part2
df = pd.read_excel('C:/Users/cmos/Desktop/New folder (2)/Month4_4.xlsx',encoding='latin-1')
df = df.iloc[:,1:]
df1 = pd.read_excel('C:/Users/cmos/Desktop/New folder (2)/Part2_month4.xlsx',encoding='latin-1')
x_lag = 2
y_lag = 1               
days1 = df1.shape[0] 
jobs1 = df1.shape[1] 
persons = 5
rnadd = 0
sumi = 0
condition1 = 0
for j in range(x_lag,jobs1):
    print(j)
    for d in range(y_lag,days1):
        print(d)
        if df1.iloc[d,j] == 1:
            for i in range(10000):
                print(i)
                if persondata1[i,int((df1.iloc[d,1])-1)] == 0:
                    print(i)
                    persondata1[i,(int(df1.iloc[d,1]))-1] = df.iloc[(int(df1.iloc[0,j]))-1,(int(df1.iloc[d,0])-1)]
                    sumi = sumi + df.iloc[(int(df1.iloc[0,j]))-1,(int(df1.iloc[d,0])-1)]
                    break
#Month5_Part2
df = pd.read_excel('C:/Users/cmos/Desktop/New folder (2)/Month5_5.xlsx',encoding='latin-1')
df = df.iloc[:,1:]
df1 = pd.read_excel('C:/Users/cmos/Desktop/New folder (2)/Part2_month5.xlsx',encoding='latin-1')
x_lag = 2
y_lag = 1               
days1 = df1.shape[0] 
jobs1 = df1.shape[1] 
persons = 5
rnadd = 0
sumi = 0
condition1 = 0
for j in range(x_lag,jobs1):
    print(j)
    for d in range(y_lag,days1):
        print(d)
        if df1.iloc[d,j] == 1:
            for i in range(10000):
                print(i)
                if persondata1[i,int((df1.iloc[d,1])-1)] == 0:
                    print(i)
                    persondata1[i,(int(df1.iloc[d,1]))-1] = df.iloc[(int(df1.iloc[0,j]))-1,(int(df1.iloc[d,0])-1)]
                    sumi = sumi + df.iloc[(int(df1.iloc[0,j]))-1,(int(df1.iloc[d,0])-1)]
                    break
#Month6_Part2
df = pd.read_excel('C:/Users/cmos/Desktop/New folder (2)/Month6_6.xlsx',encoding='latin-1')
df = df.iloc[:,1:]
df1 = pd.read_excel('C:/Users/cmos/Desktop/New folder (2)/Part2_month6.xlsx',encoding='latin-1')
x_lag = 2
y_lag = 1               
days1 = df1.shape[0] 
jobs1 = df1.shape[1] 
persons = 5
rnadd = 0
sumi = 0
condition1 = 0
for j in range(x_lag,jobs1):
    print(j)
    for d in range(y_lag,days1):
        print(d)
        if df1.iloc[d,j] == 1:
            for i in range(10000):
                print(i)
                if persondata1[i,int((df1.iloc[d,1])-1)] == 0:
                    print(i)
                    persondata1[i,(int(df1.iloc[d,1]))-1] = df.iloc[(int(df1.iloc[0,j]))-1,(int(df1.iloc[d,0])-1)]
                    sumi = sumi + df.iloc[(int(df1.iloc[0,j]))-1,(int(df1.iloc[d,0])-1)]
                    break
#Month7_Part2
df = pd.read_excel('C:/Users/cmos/Desktop/New folder (2)/Month7_7.xlsx',encoding='latin-1')
df = df.iloc[:,1:]
df1 = pd.read_excel('C:/Users/cmos/Desktop/New folder (2)/Part2_month7.xlsx',encoding='latin-1')
x_lag = 2
y_lag = 1               
days1 = df1.shape[0] 
jobs1 = df1.shape[1] 
persons = 5
rnadd = 0
sumi = 0
condition1 = 0
for j in range(x_lag,jobs1):
    print(j)
    for d in range(y_lag,days1):
        print(d)
        if df1.iloc[d,j] == 1:
            for i in range(10000):
                print(i)
                if persondata1[i,int((df1.iloc[d,1])-1)] == 0:
                    print(i)
                    persondata1[i,(int(df1.iloc[d,1]))-1] = df.iloc[(int(df1.iloc[0,j]))-1,(int(df1.iloc[d,0])-1)]
                    sumi = sumi + df.iloc[(int(df1.iloc[0,j]))-1,(int(df1.iloc[d,0])-1)]
                    break
#Month8_Part2
df = pd.read_excel('C:/Users/cmos/Desktop/New folder (2)/Month8_8.xlsx',encoding='latin-1')
df = df.iloc[:,1:]
df1 = pd.read_excel('C:/Users/cmos/Desktop/New folder (2)/Part2_month8.xlsx',encoding='latin-1')
x_lag = 2
y_lag = 1               
days1 = df1.shape[0] 
jobs1 = df1.shape[1] 
persons = 5
rnadd = 0
sumi = 0
condition1 = 0
for j in range(x_lag,jobs1):
    print(j)
    for d in range(y_lag,days1):
        print(d)
        if df1.iloc[d,j] == 1:
            for i in range(10000):
                print(i)
                if persondata1[i,int((df1.iloc[d,1])-1)] == 0:
                    print(i)
                    persondata1[i,(int(df1.iloc[d,1]))-1] = df.iloc[(int(df1.iloc[0,j]))-1,(int(df1.iloc[d,0])-1)]
                    sumi = sumi + df.iloc[(int(df1.iloc[0,j]))-1,(int(df1.iloc[d,0])-1)]
                    break
#Month9_Part2
df = pd.read_excel('C:/Users/cmos/Desktop/New folder (2)/Month9_9.xlsx',encoding='latin-1')
df = df.iloc[:,1:]
df1 = pd.read_excel('C:/Users/cmos/Desktop/New folder (2)/Part2_month9.xlsx',encoding='latin-1')
x_lag = 2
y_lag = 1               
days1 = df1.shape[0] 
jobs1 = df1.shape[1] 
persons = 5
rnadd = 0
sumi = 0
condition1 = 0
for j in range(x_lag,jobs1):
    print(j)
    for d in range(y_lag,days1):
        print(d)
        if df1.iloc[d,j] == 1:
            for i in range(10000):
                print(i)
                if persondata1[i,int((df1.iloc[d,1])-1)] == 0:
                    print(i)
                    persondata1[i,(int(df1.iloc[d,1]))-1] = df.iloc[(int(df1.iloc[0,j]))-1,(int(df1.iloc[d,0])-1)]
                    sumi = sumi + df.iloc[(int(df1.iloc[0,j]))-1,(int(df1.iloc[d,0])-1)]
                    break
#Month10_Part2
df = pd.read_excel('C:/Users/cmos/Desktop/New folder (2)/Month10_10.xlsx',encoding='latin-1')
df = df.iloc[:,1:]
df1 = pd.read_excel('C:/Users/cmos/Desktop/New folder (2)/Part2_month10.xlsx',encoding='latin-1')
x_lag = 2
y_lag = 1               
days1 = df1.shape[0] 
jobs1 = df1.shape[1] 
persons = 5
rnadd = 0
sumi = 0
condition1 = 0
for j in range(x_lag,jobs1):
    print(j)
    for d in range(y_lag,days1):
        print(d)
        if df1.iloc[d,j] == 1:
            for i in range(10000):
                print(i)
                if persondata1[i,int((df1.iloc[d,1])-1)] == 0:
                    print(i)
                    persondata1[i,(int(df1.iloc[d,1]))-1] = df.iloc[(int(df1.iloc[0,j]))-1,(int(df1.iloc[d,0])-1)]
                    sumi = sumi + df.iloc[(int(df1.iloc[0,j]))-1,(int(df1.iloc[d,0])-1)]
                    break
#Month11_Part2
df = pd.read_excel('C:/Users/cmos/Desktop/New folder (2)/Month11_11.xlsx',encoding='latin-1')
df = df.iloc[:,1:]
df1 = pd.read_excel('C:/Users/cmos/Desktop/New folder (2)/Part2_month11.xlsx',encoding='latin-1')
x_lag = 2
y_lag = 1               
days1 = df1.shape[0] 
jobs1 = df1.shape[1] 
persons = 5
rnadd = 0
sumi = 0
condition1 = 0
for j in range(x_lag,jobs1):
    print(j)
    for d in range(y_lag,days1):
        print(d)
        if df1.iloc[d,j] == 1:
            for i in range(10000):
                print(i)
                if persondata1[i,int((df1.iloc[d,1])-1)] == 0:
                    print(i)
                    persondata1[i,(int(df1.iloc[d,1]))-1] = df.iloc[(int(df1.iloc[0,j]))-1,(int(df1.iloc[d,0])-1)]
                    sumi = sumi + df.iloc[(int(df1.iloc[0,j]))-1,(int(df1.iloc[d,0])-1)]
                    break
#Month12_Part2
df = pd.read_excel('C:/Users/cmos/Desktop/New folder (2)/Month12_12.xlsx',encoding='latin-1')
df = df.iloc[:,1:]
df1 = pd.read_excel('C:/Users/cmos/Desktop/New folder (2)/Part2_month12.xlsx',encoding='latin-1')
x_lag = 2
y_lag = 1               
days1 = df1.shape[0] 
jobs1 = df1.shape[1] 
persons = 5
rnadd = 0
sumi = 0
condition1 = 0
for j in range(x_lag,jobs1):
    print(j)
    for d in range(y_lag,days1):
        print(d)
        if df1.iloc[d,j] == 1:
            for i in range(10000):
                print(i)
                if persondata1[i,int((df1.iloc[d,1])-1)] == 0:
                    print(i)
                    persondata1[i,(int(df1.iloc[d,1]))-1] = df.iloc[(int(df1.iloc[0,j]))-1,(int(df1.iloc[d,0])-1)]
                    sumi = sumi + df.iloc[(int(df1.iloc[0,j]))-1,(int(df1.iloc[d,0])-1)]
                    break

persondata
sumi = sum(sum(persondata1))
sumi2 = sum(sum(persondata))
ALL_Random = abs(sum(persondata[:,0])-sumi2/5)+abs(sum(persondata[:,1])-sumi2/5)+abs(sum(persondata[:,2])-sumi2/5)+abs(sum(persondata[:,3])-sumi2/5)+abs(sum(persondata[:,4])-sumi2/5)
ALL_Optimal = abs(sum(persondata1[:,0])-sumi/5)+abs(sum(persondata1[:,1])-sumi/5)+abs(sum(persondata1[:,2])-sumi/5)+abs(sum(persondata1[:,3])-sumi/5)+abs(sum(persondata1[:,4])-sumi/5)


#pppppllllllllloooooooottttttt
w = np.zeros((12,1))
for i in range(12):
    w[i,0] = sum(meandata_month_Pre[:,i])/500
pd.DataFrame(w)
comperationF = pd.concat([pd.DataFrame(comperation),pd.DataFrame(w)],1)
e = [240000,160000,160000,240000,160000,160000,160000,0,0,240000,160000,160000]
pd.DataFrame(e)
comperationF.columns = ['Random','Optimal','Predictor']
comperationFF = pd.concat([pd.DataFrame(comperationF),pd.DataFrame(e)],1)
e = [15440000,23620000,21520000,18440000,23080000,25180000,35580000,25400000,34000000,40560000,38820000,31920000]
pd.DataFrame(e)
comperationFF = pd.concat([pd.DataFrame(comperationFF),pd.DataFrame(e)],1)
comperationFF.columns = ['Random','Optimal','Predictor','Best','mean']
e = [139,210,171,153,178,183,248,187,214,238,228,167]
pd.DataFrame(e)
comperationFF = pd.concat([pd.DataFrame(comperationFF),pd.DataFrame(e)],1)
comperationFF.columns = ['Random','Optimal','Predictor','Best','mean','number']


df_plot=pd.DataFrame({'month': range(1,13), 'Random': comperationFF.iloc[:,0]/persons/comperationFF.iloc[:,4]*100, 'Optimal': comperationFF.iloc[:,1]/persons/comperationFF.iloc[:,4]*100, 'Predictor': comperationFF.iloc[:,2]/persons/comperationFF.iloc[:,4]*100,'Best': comperationFF.iloc[:,3]/persons,'Mean': comperationFF.iloc[:,4]/comperationFF.iloc[:,4]*100,'Number': comperationFF.iloc[:,5]})
plt.figure(figsize=(16, 8))
#plt.subplot(131,132)
#plt.plot( 'month', 'Random', data=df_plot, marker='o', markerfacecolor='blue', markersize=12, color='skyblue', linewidth=4)
plt.plot( 'month', 'Optimal', data=df_plot, marker='o', color='olive', linewidth=3)
#plt.plot( 'month', 'Predictor', data=df_plot, marker='', color='red', linewidth=3, linestyle='dashed')
#plt.plot( 'month', 'Best', data=df_plot,  marker='o', markerfacecolor='red', markersize=8, color='black', linewidth=3, linestyle='-.')
#plt.plot( 'month', 'Mean', data=df_plot, marker='o', markerfacecolor='red', color='black', linewidth=3, linestyle='-.')
#plt.plot( 'month', 'Number', data=df_plot, marker='', color='red', linewidth=3, linestyle='dashed')

plt.title("Error percentage of Justice chart for comparing methods",size = 18)
plt.xlabel('Month',size = 18)
plt.ylabel('percentage',size = 18)
plt.grid()
plt.legend()
plt.subplot(122)
ax = sns.boxplot(data=df_plot.iloc[:,2])
plt.title("Boxplot of Optimal distribution",size = 18)
plt.ylabel('percentage',size = 18)
plt.xlabel('Optimal',size = 18)
plt.grid(linestyle='-', linewidth=0.5)
plt.show()


plt.figure(figsize=(6, 6))
ax = sns.boxplot(data=comperationFF.iloc[:,3]/persons)
plt.title("Boxplot of Methods", loc="right")
plt.grid(linestyle='-', linewidth=0.5)


x_coordinates = [0, 12]
y_coordinates = [ALL_Random, ALL_Random]
plt.plot(x_coordinates, y_coordinates,color = 'b')
x_coordinates = [0, 12]
y_coordinates = [ALL_Optimal, ALL_Optimal]
plt.plot(x_coordinates, y_coordinates,color = 'r')
x_coordinates = [0, 12]
y_coordinates = [All_Predictor, All_Predictor]
plt.plot(x_coordinates, y_coordinates,color = 'g')
sum(comperationFF.iloc[:,3])
ALL_Best = 70000
#Barplot
height = [ALL_Random/persons/meanpre*100, ALL_Optimal/persons/meanpre*100, All_Predictor/persons/meanpre*100,ALL_Best/persons/meanpre*100]
bars = ('Random', 'Optimal', 'Predictor','Best')
y_pos = np.arange(len(bars))
plt.figure(figsize=(12, 8))
plt.bar(y_pos[0], height[0], color=(0 ,0, 1))
plt.bar(y_pos[1], height[1], color=(1 ,0, 0))
plt.bar(y_pos[2], height[2], color=(0 ,1, 0.4))
plt.bar(y_pos[3], height[3], color=(0 ,0, 0))
plt.xticks(y_pos, bars)
plt.title("Barplot for comparing Methods during the year")
plt.ylabel('Value')
plt.text(x = y_pos[0]-0.2 , y = height[0]+0.01, s = np.round(height[0],2), size = 20)
plt.text(x = y_pos[1] , y = height[1]+0.01, s = np.round(height[1],2), size = 6)
plt.text(x = y_pos[2] , y = height[2]+0.01, s = np.round(height[2],2), size = 6)
plt.text(x = y_pos[3] , y = height[3]+0.01, s = np.round(height[3],2), size = 6)
plt.grid()
plt.show()



# Show Boxplot
a = pd.DataFrame({ 'Error' : np.repeat('Random',1000), 'value': meandata[:,0] })
b = pd.DataFrame({ 'Error' : np.repeat('Predictor',1000), 'value': meandata_Pre[:,0] })
df_box=a.append(b) 
# Usual boxplot
plt.figure(figsize=(12, 8))
sns.boxplot(x='Error', y='value', data=df_box)
plt.title('Boxplot for Year')
plt.figure(figsize=(14, 10))
ax = sns.boxplot(x='Error', y='value', data=df_box)
ax = sns.stripplot(x='Error', y='value', data=df_box, color="orange", jitter=0.2, size=2.5)
plt.title("Boxplot for Year", loc="right")
plt.grid(linestyle='-', linewidth=0.5)

plt.figure(figsize=(12, 8))
sns.distplot( a=meandata, kde=True,kde_kws={"color": "g", "alpha":0.3, "linewidth": 5, "shade":True })
sns.distplot( a=meandata_Pre, kde=True,kde_kws={"color": "r", "alpha":0.3, "linewidth": 5, "shade":True })
plt.title("histogram for random and predictor", loc="right")
plt.xlabel('Frequency',)
plt.ylabel('Value')
plt.grid()
plt.legend('RP')

# Show Boxplot
a = pd.DataFrame({ 'Error' : np.repeat('Month 3',len(month[month[:,3] != 0,3])), 'value': month[month[:,3] != 0,3] })
b = pd.DataFrame({ 'Error' : np.repeat('Month 9',len(month[month[:,9] != 0,9])), 'value': month[month[:,9] != 0,9] })
df_box=a.append(b) 
# Usual boxplot
plt.figure(figsize=(12, 8))
sns.boxplot(x='Error', y='value', data=df_box)
plt.title('Boxplot for Year')
plt.figure(figsize=(14, 10))
ax = sns.boxplot(x='Error', y='value', data=df_box)
ax = sns.stripplot(x='Error', y='value', data=df_box, color="orange", jitter=0.2, size=2.5)
plt.title("Boxplot for Year", loc="right")
plt.grid(linestyle='-', linewidth=0.5)

plt.figure(figsize=(12, 8))
sns.distplot( day[:,3], kde=True,kde_kws={"color": "b", "alpha":0.3, "linewidth": 5, "shade":True })
#sns.distplot( month[month[:,8] != 0,8], kde=True,kde_kws={"color": "g", "alpha":0.3, "linewidth": 5, "shade":True })
sns.distplot( day[:,9], kde=True,kde_kws={"color": "r", "alpha":0.3, "linewidth": 5, "shade":True })
plt.title("histogram for random and predictor", loc="right")
plt.xticks(size = 10)
plt.yticks(size = 0)
plt.xlabel('Value')
plt.grid()
plt.legend('389')


df_plot=pd.DataFrame({'Day': range(1,32), 'Month 3': day[:,3], 'Month 10': day[:,9]})
plt.figure(figsize=(16, 8))
#plt.subplot(131,132)
#plt.plot( 'month', 'Random', data=df_plot, marker='o', markerfacecolor='blue', markersize=12, color='skyblue', linewidth=4)
plt.bar( 'Day', 'Month 3', data=df_plot, marker='o', color='olive', linewidth=3)
plt.plot( 'Day', 'Month 10', data=df_plot, marker='', color='red', linewidth=3, linestyle='dashed')
#plt.plot( 'month', 'Best', data=df_plot,  marker='o', markerfacecolor='red', markersize=8, color='black', linewidth=3, linestyle='-.')
#plt.plot( 'month', 'Mean', data=df_plot, marker='o', markerfacecolor='red', color='black', linewidth=3, linestyle='-.')
#plt.plot( 'month', 'Number', data=df_plot, marker='', color='red', linewidth=3, linestyle='dashed')
plt.grid()
plt.legend()

height = day[:,3]
bars = range(1,32)
y_pos = np.arange(len(bars))
 
# Create bars and choose color
plt.figure(figsize=(16, 8))
plt.bar(y_pos, day[:,3], color = (0.5,0.1,0.9,0.6), label='Month 4')
plt.bar(y_pos, day[:,9], color = (0.5,0.5,0.1,0.6), label='Month 10')
# Add title and axis names
plt.title('Barplot of month4 and month10',size = 20)
plt.xlabel('Days',size = 20)
plt.ylabel('number',size = 20)

plt.show()
plt.grid()
plt.legend()
plt.ylim(0,60)
plt.xticks(y_pos, bars)
plt.show()

month4 = persondata1
month10 = persondata1
month4_sum = [sum(month4[:,0]),sum(month4[:,1]),sum(month4[:,2]),sum(month4[:,3]),sum(month4[:,4])]
month10_sum = [sum(month10[:,0]),sum(month10[:,1]),sum(month10[:,2]),sum(month10[:,3]),sum(month10[:,4])]
bars = range(1,6)
y_pos = np.arange(len(bars))
plt.figure(figsize=(16, 8))
plt.bar(y_pos, month4_sum, color = (0.0,0.0,0.9,0.6), label='Month 4')
plt.bar(y_pos, month10_sum, color = (0.2,0.9,0.5,0.6), label='Month 10')
x_coordinates = [0, 4]
y_coordinates = [sum(month4_sum)/5, sum(month4_sum)/5]
plt.plot(x_coordinates, y_coordinates,color = 'b',linewidth=3,label='Ideal for month 4')
x_coordinates = [0, 4]
y_coordinates = [sum(month10_sum)/5, sum(month10_sum)/5]
plt.plot(x_coordinates, y_coordinates,color = 'r',linewidth=3,label='Ideal for month 10')
# Add title and axis names
plt.title('Barplot of month4 and month10',size = 20)
plt.xlabel('Personel',size = 20)
plt.ylabel('Received',size = 20)
plt.grid()
plt.legend()
plt.xticks(y_pos, bars)
plt.show()


#MonthALL_Random
df = pd.read_excel('C:/Users/cmos/Desktop/New folder (2)/ALL.xlsx',encoding='latin-1')
df = df.iloc[:,1:]
days = df.shape[1]
jobs = df.shape[0]
persons = 5
rnadd = 0
sumi = 0
condition1 = 0
persondata = np.zeros((1000,persons+5))
persongetdata = np.zeros((1,persons))
meandata = np.zeros((1000,1))
for t in range(1000):
    print(t)
    for d in range(days):
        persongetdata[0,:] = 0
        for j in range(jobs - sum(df.iloc[:,d].isna())):
            condition1 = 0
            if sum(sum(persongetdata)) == persons:
                    persongetdata[0,:] = 0
            while condition1 == 0:
                rnadd = np.random.randint(0,persons)
                if persongetdata[0,rnadd] == 0:
                    for i in range(len(persondata)):
                        if persondata[i,rnadd] == 0:
                            persondata[i,rnadd] = df.iloc[j,d]
                            persondata[i,rnadd+5] = d
                            sumi = sumi + df.iloc[j,d]
                            break
                    persongetdata[0,rnadd] = 1
                    condition1 = 1
    meandata[t,0] = abs(sum(persondata[:,0])-sumi/5)+abs(sum(persondata[:,1])-sumi/5)+abs(sum(persondata[:,2])-sumi/5)+abs(sum(persondata[:,3])-sumi/5)+abs(sum(persondata[:,4])-sumi/5)                     
    persondata = np.zeros((1000,persons+5))
    sumi = 0
    persongetdata[0,:] = 0



#MonthALL_Predictor
df = pd.read_excel('C:/Users/cmos/Desktop/New folder (2)/Predictor.xlsx',encoding='latin-1')
df = df.iloc[:,2:]
jobs = df.shape[0]
persons = 5
rnadd = 0
sumi = 0
condition1 = 0
persondata = np.zeros((1000,persons))
persongetdata = np.zeros((1,persons))
meandata_Pre = np.zeros((1000,1))
counterjob = 0
passingjob = 0
counter = 0
alter = df.iloc[0,1]
for t in range(1000):
    print(t)
    while counterjob<jobs:
        #print(counterjob)
        passingjob = passingjob + counter
        counter = 0
        for i in range(passingjob,jobs):
            if alter == df.iloc[i,1]:
                counter = counter + 1
            else:
                persongetdata[0,:] = 0
                alter = df.iloc[i,1]
                break
        for i in range(passingjob,passingjob+counter):
            condition1 = 0
            if sum(sum(persongetdata)) == persons:
                persongetdata[0,:] = 0
            if df.iloc[i,3] == 3:
                while condition1 == 0:
                    rnadd = np.random.randint(0,persons)
                    if persongetdata[0,rnadd] == 0:
                        for j in range(len(persondata)):
                            if persondata[j,rnadd] == 0:
                                persondata[j,rnadd] = df.iloc[i,2]
                                #persondata[i,rnadd+5] = d
                                sumi = sumi + df.iloc[i,2]
                                counterjob = counterjob + 1
                                break
                        persongetdata[0,rnadd] = 1
                        condition1 = 1
        for i in range(passingjob,passingjob+counter):
            condition1 = 0
            if sum(sum(persongetdata)) == persons:
                persongetdata[0,:] = 0
            if df.iloc[i,3] == 2:
                while condition1 == 0:
                    rnadd = np.random.randint(0,persons)
                    if persongetdata[0,rnadd] == 0:
                        for j in range(len(persondata)):
                            if persondata[j,rnadd] == 0:
                                persondata[j,rnadd] = df.iloc[i,2]
                                #persondata[i,rnadd+5] = d
                                sumi = sumi + df.iloc[i,2]
                                counterjob = counterjob + 1
                                break
                        persongetdata[0,rnadd] = 1
                        condition1 = 1
                        break
        for i in range(passingjob,passingjob+counter):
            condition1 = 0
            if sum(sum(persongetdata)) == persons:
                persongetdata[0,:] = 0
            if df.iloc[i,3] == 1:
                while condition1 == 0:
                    rnadd = np.random.randint(0,persons)
                    if persongetdata[0,rnadd] == 0:
                        for j in range(len(persondata)):
                            if persondata[j,rnadd] == 0:
                                persondata[j,rnadd] = df.iloc[i,2]
                                #persondata[i,rnadd+5] = d
                                sumi = sumi + df.iloc[i,2]
                                counterjob = counterjob + 1
                                break
                        persongetdata[0,rnadd] = 1
                        condition1 = 1
                              
    meandata_Pre[t,0] = abs(sum(persondata[:,0])-sumi/5)+abs(sum(persondata[:,1])-sumi/5)+abs(sum(persondata[:,2])-sumi/5)+abs(sum(persondata[:,3])-sumi/5)+abs(sum(persondata[:,4])-sumi/5)                     
    persondata = np.zeros((1000,persons+5))
    sumi = 0
    counterjob = 0
    passingjob = 0
    counter = 0
    alter = df.iloc[0,1]
    persongetdata[0,:] = 0
    
    
All_Predictor = np.mean(meandata_Pre)
plt.figure(figsize=(6, 6))
ax = sns.boxplot(data=meandata)
ax = sns.stripplot(data=meandata, color="orange", jitter=0.2, size=2.5)
plt.title("Boxplot of Methods", loc="right")
plt.grid(linestyle='-', linewidth=0.5)


month = np.zeros((500,12))
for i in range(len(df)):
    for j in range(len(month)):
        if month[j,int(df.iloc[i,0])-1] == 0:
            month[j,int(df.iloc[i,0])-1] = df.iloc[i,2]
            break
        
day = np.zeros((31,12))
for i in range(len(df)):
    day[int(df.iloc[i,1])-1,int(df.iloc[i,0])-1] = day[int(df.iloc[i,1])-1,int(df.iloc[i,0])-1] + 1
                