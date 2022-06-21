# -*- coding: utf-8 -*-
import pickle
import numpy as np
import torch
from predictor import Predictor, train_grui_model


f = open('E:/WashU/Research/ICU/Data/train/X_train_sliced_norm_GAN.pkl','rb')
X_train_sliced = pickle.load(f)
f.close()
f = open('E:/WashU/Research/ICU/Data/train/y_train.pkl','rb')
y_train = pickle.load(f)
f.close()
f = open('E:/WashU/Research/ICU/Data/train/train_delta_mat.pkl','rb')
train_delta_mat = pickle.load(f)
f.close()

f = open('E:/WashU/Research/ICU/Data/val/X_val_sliced_norm_GAN.pkl','rb')
X_test_sliced = pickle.load(f)
f.close()
f = open('E:/WashU/Research/ICU/Data/val/y_val.pkl','rb')
y_test = pickle.load(f)
f.close()
f = open('E:/WashU/Research/ICU/Data/val/val_delta_mat.pkl','rb')
test_delta_mat = pickle.load(f)
f.close()


X_train_sliced = np.array(X_train_sliced)
X_test_sliced = np.array(X_test_sliced)
y_train = np.array(y_train)
y_test = np.array(y_test)
train_delta_mat = np.array(train_delta_mat)
test_delta_mat = np.array(test_delta_mat)


X_resampled = []
y_resampled = []
delta_resampled = []
y_one_index = np.argwhere(y_train==1).reshape(-1)
for i in np.argwhere(y_train==0).reshape(-1):
    X_resampled.append(X_train_sliced[i])
    y_resampled.append(y_train[i])
    delta_resampled.append(train_delta_mat[i])
    sample_index = np.random.randint(0, len(y_one_index)-1)
    X_resampled.append(X_train_sliced[y_one_index[sample_index]])
    y_resampled.append(y_train[y_one_index[sample_index]])
    delta_resampled.append(train_delta_mat[y_one_index[sample_index]])
X_resampled = np.array(X_resampled)
delta_resampled = np.array(delta_resampled)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = Predictor(num_inputs=X_resampled.shape[2], num_hiddens=64, imputeMethod='GAN', scaleMethod='Norm').to(device)
model = train_grui_model(model, X_resampled, y_resampled, delta_resampled, X_test_sliced, y_test, test_delta_mat, batch_size=32, lr=0.1, num_epoch=100)
