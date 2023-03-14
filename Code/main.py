import numpy as np
import pickle
from train import train, imputation
from basicImputation import imputeMean, imputeNearest, imputeLast


f = open('E:/WashU/Research/ICU/Data/train/X_train_sliced_norm.pkl','rb')
X_train_sliced = np.array(pickle.load(f))
f.close()
f = open('E:/WashU/Research/ICU/Data/train/y_train.pkl','rb')
y_train = np.array(pickle.load(f))
f.close()
f = open('E:/WashU/Research/ICU/Data/train/train_delta_mat.pkl','rb')
train_delta_mat = np.array(pickle.load(f))
f.close()
f = open('E:/WashU/Research/ICU/Data/train/train_mask_mat.pkl','rb')
train_mask_mat = np.array(pickle.load(f))
f.close()

f = open('E:/WashU/Research/ICU/Data/val/X_val_sliced_norm.pkl','rb')
X_val_sliced = np.array(pickle.load(f))
f.close()
f = open('E:/WashU/Research/ICU/Data/val/y_val.pkl','rb')
y_val = np.array(pickle.load(f))
f.close()
f = open('E:/WashU/Research/ICU/Data/val/val_delta_mat.pkl','rb')
val_delta_mat = np.array(pickle.load(f))
f.close()
f = open('E:/WashU/Research/ICU/Data/val/val_mask_mat.pkl','rb')
val_mask_mat = np.array(pickle.load(f))
f.close()
'''
f = open('E:/WashU/Research/ICU/Data/test/X_test_sliced_norm.pkl','rb')
X_test_sliced = np.array(pickle.load(f))
f.close()
f = open('E:/WashU/Research/ICU/Data/test/y_test.pkl','rb')
y_test = np.array(pickle.load(f))
f.close()
f = open('E:/WashU/Research/ICU/Data/test/test_delta_mat.pkl','rb')
test_delta_mat = np.array(pickle.load(f))
f.close()
f = open('E:/WashU/Research/ICU/Data/test/test_mask_mat.pkl','rb')
test_mask_mat = np.array(pickle.load(f))
f.close()
'''

f = open('E:/WashU/Research/ICU/Data/mean_norm.pkl','rb')
mean = np.array(pickle.load(f))
f.close()


X_train_sliced = np.array(imputeLast(X_train_sliced, mean))
X_val_sliced = np.array(imputeLast(X_val_sliced, mean))

generator, discriminator = train(X_train_sliced, train_mask_mat, train_delta_mat, 128, 22, 5, 0.01, 8)
imputation(generator, discriminator, X_train_sliced, train_mask_mat, 64, 0.01, 5, 150)