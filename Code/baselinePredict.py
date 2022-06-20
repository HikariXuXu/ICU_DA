# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
import pickle

f = open('E:/WashU/Research/ICU/Data/train/X_train_sliced_norm_GAN.pkl','rb')
X_train_sliced = pickle.load(f)
f.close()
f = open('E:/WashU/Research/ICU/Data/train/y_train.pkl','rb')
y_train = pickle.load(f)
f.close()
f = open('E:/WashU/Research/ICU/Data/train/train_mask_mat.pkl','rb')
train_mask_mat = pickle.load(f)
f.close()

f = open('E:/WashU/Research/ICU/Data/val/X_val_sliced_norm_GAN.pkl','rb')
X_test_sliced = pickle.load(f)
f.close()
f = open('E:/WashU/Research/ICU/Data/val/y_val.pkl','rb')
y_test = pickle.load(f)
f.close()
f = open('E:/WashU/Research/ICU/Data/val/val_mask_mat.pkl','rb')
test_mask_mat = pickle.load(f)
f.close()

X_train = []
for i in range(len(X_train_sliced)):
    a = X_train_sliced[i].flatten()
    X_train.append(a)

X_test = []
for i in range(len(X_test_sliced)):
    a = X_test_sliced[i].flatten()
    X_test.append(a)

def score1(method_Pred, ytest):
    score1 = 0
    TP = 0
    FN = 0
    FP = 0
    TN = 0
    for i in range (len(ytest)):
        if (ytest[i] == 1) & (method_Pred[i] == 1):
            TP = TP + 1
        if (ytest[i] == 1) & (method_Pred[i] == 0):
            FN = FN + 1
        if (ytest[i] == 0) & (method_Pred[i] == 1):
            FP = FP + 1
        if (ytest[i] == 0) & (method_Pred[i] == 0):
            TN = TN + 1
    if ((TP == 0) & (FN == 0)):
        Se = 0
    else:
        Se = TP/(TP+FN)
        
    if ((TP == 0) & (FP == 0)):
        P = 0
    else:
        P = TP/(TP+FP)
    
    if Se > P:
        score1 = P
    else:
        score1 = Se
    return score1

# def predict_model(X_train, y_train, X_test, y_test):
adaboost = AdaBoostClassifier(n_estimators=100, random_state=0)
adaboost.fit(X_train, y_train)
adaboost_predict = adaboost.predict(X_test)
adaboost_pred_proba = adaboost.predict_proba(X_test)
adaboost_auc = metrics.roc_auc_score(y_test, adaboost_pred_proba[:, 1])
adaboost_score1 = score1(adaboost_predict, y_test)
print('adaboost_auc is %f' %(adaboost_auc), end = "\n")
print('adaboost_score1 is %f' %(adaboost_score1), end = "\n")

rforest = RandomForestClassifier(max_depth=2, random_state=0)
rforest.fit(X_train, y_train)
rforest_predict = rforest.predict(X_test)
rforest_pred_proba = rforest.predict_proba(X_test)
rforest_auc = metrics.roc_auc_score(y_test, rforest_pred_proba[:, 1])
rforest_score1 = score1(rforest_predict, y_test)
print('rforest_auc is %f' %(rforest_auc), end = "\n")
print('rforest_score1 is %f' %(rforest_score1), end = "\n")

gradboost = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0, max_depth=1, random_state=0)
gradboost.fit(X_train,y_train)
gradboost_predict = gradboost.predict(X_test)
gradboost_pred_proba = gradboost.predict_proba(X_test)
gradboost_auc = metrics.roc_auc_score(y_test, gradboost_pred_proba[:, 1])
gradboost_score1 = score1(gradboost_predict, y_test)
print('gradboost_auc is %f' %(gradboost_auc), end = "\n")
print('gradboost_score1 is %f' %(gradboost_score1), end = "\n")

svmsigmoid = SVC(kernel='sigmoid', probability=True)
svmsigmoid.fit(X_train, y_train)
svmsigmoid_predict = svmsigmoid.predict(X_test)
svmsigmoid_pred = svmsigmoid.predict_proba(X_test)
svmsigmoid_auc = metrics.roc_auc_score(y_test, svmsigmoid_pred[:, 1])
svmsigmoid_score1 = score1(svmsigmoid_pred[:, 0], y_test)
print('svm_sigmoid_auc is %f' %(svmsigmoid_auc), end = "\n")
print('svm_sigmoid_score1 is %f' %(svmsigmoid_score1), end = "\n")

svmrbf = SVC(kernel='rbf', probability=True)
svmrbf.fit(X_train, y_train)
rbf_predict = svmrbf.predict(X_test)
rbf_pred_proba = svmrbf.predict_proba(X_test)
rbf_auc = metrics.roc_auc_score(y_test, rbf_pred_proba[:, 1])
rbf_score1 = score1(rbf_predict, y_test)
print('svm_rbf_auc is %f' %(rbf_auc), end = "\n")
print('svm_rbf_score1 is %f' %(rbf_score1), end = "\n")

svmpoly = SVC(kernel='poly', probability=True)
svmpoly.fit(X_train, y_train)
poly_predict = svmpoly.predict(X_test)
poly_pred_proba = svmpoly.predict_proba(X_test)
poly_auc = metrics.roc_auc_score(y_test, poly_pred_proba[:, 1])
poly_score1 = score1(poly_predict, y_test)
print('svm_poly_auc is %f' %(poly_auc), end = "\n")
print('svm_poly_score1 is %f' %(poly_score1), end = "\n")

logreg = LogisticRegression()
logreg.fit(X_train, y_train)
log_predict = logreg.predict(X_test)
log_pred_proba = logreg.predict_proba(X_test)
log_auc = metrics.roc_auc_score(y_test, log_pred_proba[:, 1])
log_score1 = score1(log_predict, y_test)
print('logistic_auc is %f' %(log_auc), end = "\n")
print('logistic_score1 is %f' %(log_score1), end = "\n")

# predict_model(X_train, y_train, X_test, y_test)