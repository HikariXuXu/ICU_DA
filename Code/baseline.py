# -*- coding: utf-8 -*-

from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
import pickle
import basicImputation

f = open('../Data/train/X_train_sliced.pkl','rb')
X_train_sliced = pickle.load(f)
f.close()
f = open('../Data/train/y_train.pkl','rb')
y_train = pickle.load(f)
f.close()
f = open('../Data/train/train_mask_mat.pkl','rb')
train_mask_mat = pickle.load(f)
f.close()

f = open('../Data/test1/X_test1_sliced.pkl','rb')
X_test1_sliced = pickle.load(f)
f.close()
f = open('../Data/test1/y_test1.pkl','rb')
y_test1 = pickle.load(f)
f.close()
f = open('../Data/test1/test1_mask_mat.pkl','rb')
test1_mask_mat = pickle.load(f)
f.close()

f = open('../Data/test2/X_test2_sliced.pkl','rb')
X_test2_sliced = pickle.load(f)
f.close()
f = open('../Data/test2/y_test2.pkl','rb')
y_test2 = pickle.load(f)
f.close()
f = open('../Data/test2/test2_mask_mat.pkl','rb')
test2_mask_mat = pickle.load(f)
f.close()

f = open('../Data/mean.pkl','rb')
mean = pickle.load(f)
f.close()

def predict_model(xtrain, ytrain, xtest, ytest):
    adaboost = AdaBoostClassifier(n_estimators=100, random_state=0).fit(xtrain, ytrain)
    adaboost_pred = adaboost.predict_proba(xtest)
    adaboost_auc = metrics.roc_auc_score(ytest, adaboost_pred[:, 1])

    rforest = RandomForestClassifier(max_depth=2, random_state=0).fit(xtrain, ytrain)
    rforest_pred = rforest.predict_proba(xtest)
    rforest_auc = metrics.roc_auc_score(ytest, rforest_pred[:, 1])

    gradboost = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0, max_depth=1, random_state=0).fit(xtrain,ytrain)
    gradboost_pred = gradboost.predict_proba(xtest)
    gradboost_auc = metrics.roc_auc_score(ytest, gradboost_pred[:, 1])

    svmsigmoid = SVC(kernel='sigmoid', probability=True)
    svmsigmoid.fit(xtrain, ytrain)
    svmsigmoid_pred = svmsigmoid.predict_proba(xtest)
    sigmoid_score = svmsigmoid.score(xtest, ytest)
    sigmoid_auc = metrics.roc_auc_score(ytest, svmsigmoid_pred[:, 1])

    svmrbf = SVC(kernel='rbf', probability=True)
    svmrbf.fit(xtrain, ytrain)
    rbf_pred = svmrbf.predict_proba(xtest)
    rbf_score = svmrbf.score(xtest, ytest)
    rbf_auc = metrics.roc_auc_score(ytest, rbf_pred[:, 1])

    svmpoly = SVC(kernel='poly', probability=True)
    svmpoly.fit(xtrain, ytrain)
    poly_pred = svmpoly.predict_proba(xtest)
    poly_score = svmpoly.score(xtest, ytest)
    poly_auc = metrics.roc_auc_score(ytest, poly_pred[:, 1])

    logreg = LogisticRegression()
    logreg.fit(xtrain, ytrain)
    log_pred = logreg.predict_proba(xtest)
    log_score = logreg.score(xtest, ytest)
    log_auc = metrics.roc_auc_score(ytest, log_pred[:, 1])

    return adaboost_auc, rforest_auc, gradboost_auc, sigmoid_auc, rbf_auc, poly_auc, log_auc


trainning_imputeMean = basicImputation.imputeDataMean(X_train_sliced, train_mask_mat, mean)
xtrain_imputeMean = basicImputation.rearrange_data(trainning_imputeMean)

trainning_imputeNearest = basicImputation.imputeDataNearest(X_train_sliced, mean)
xtrain_imputeNearest = basicImputation.rearrange_data(trainning_imputeNearest)

trainning_imputeLast = basicImputation.imputeLast(X_train_sliced, mean)
xtrain_imputeLast = basicImputation.rearrange_data(trainning_imputeLast)

testing_imputeMean = basicImputation.imputeDataMean(X_test1_sliced, test1_mask_mat, mean)
xtest_imputeMean = basicImputation.rearrange_data(testing_imputeMean)

testing_imputeNearest = basicImputation.imputeDataNearest(X_test1_sliced, mean)
xtest_imputeNearest = basicImputation.rearrange_data(testing_imputeNearest)

testing_imputeLast = basicImputation.imputeLast(X_test1_sliced, mean)
xtest_imputeLast = basicImputation.rearrange_data(testing_imputeLast)

y_test = y_test1


# AUC score
adaboost_auc, rforest_auc, gradboost_auc, sigmoid_auc, rbf_auc, poly_auc, log_auc = predict_model(xtrain_imputeMean, y_train, xtest_imputeMean, y_test)

print('adaboost_auc is %f' %(adaboost_auc), end = "\n")
print('rforest_auc is %f' %(rforest_auc), end = "\n")
print('gradboost_auc is %f' %(gradboost_auc), end = "\n")
print('sigmoid_auc is %f' %(sigmoid_auc), end = "\n")
print('rbf_auc is %f' %(rbf_auc), end = "\n")
print('poly_auc is %f' %(poly_auc), end = "\n")
print('log_auc is %f' %(log_auc), end = "\n")