# -*- coding: utf-8 -*-
import numpy as np
from sklearn import metrics
import torch
import torch.nn as nn
from GRUI import GRUIModel

torch.manual_seed(0)

class Predictor(nn.Module):
    def __init__(self, num_inputs, num_hiddens, imputeMethod, scaleMethod):
        super(Predictor, self).__init__()
        self.name = 'GRUI'
        self.num_inputs = num_inputs
        self.num_hiddens = num_hiddens
        self.grui_model = GRUIModel(num_inputs, num_hiddens)
        self.imputeMethod = imputeMethod
        self.scaleMethod = scaleMethod
        
        def normal(shape):
            return torch.randn(size=shape) * 0.01

        # Parameters of fully connected layer
        self.W_fc = nn.Parameter(normal((num_hiddens, 1)))
        self.b_fc = nn.Parameter(torch.zeros(1))
        
    def forward(self, X, Delta, H):
        states = self.grui_model(X, Delta, H)
        states = states.view(X.shape[0], X.shape[1], self.num_hiddens)
        # Full connect
        risk = states[-1] @ self.W_fc + self.b_fc
        
        return risk


def grad_clipping(net, theta):
    params = [p for p in net.parameters() if p.requires_grad]
    norm = torch.sqrt(sum(torch.sum((p.grad ** 2)) for p in params))
    if norm > theta:
        for param in params:
            param.grad[:] *= theta / norm


def train_grui_model(model, X_train, y_train, train_delta_mat, X_val, y_val, val_delta_mat, batch_size, lr, num_epoch=25):
    train_loss_all, train_acc_all = [], []
    optimizer = torch.optim.SGD(model.parameters(), lr)
    criterion = nn.CrossEntropyLoss()
    threshold = torch.tensor([0.5])
    
    train_num = X_train.shape[0]
    val_num = X_val.shape[0]
    X_train = torch.from_numpy(X_train.transpose(1, 0, 2)).float()
    X_val = torch.from_numpy(X_val.transpose(1, 0, 2)).float()
    train_delta_mat = torch.from_numpy(train_delta_mat.transpose(1, 0, 2)).float()
    val_delta_mat = torch.from_numpy(val_delta_mat.transpose(1, 0, 2)).float()
    y_train = torch.from_numpy(np.array(y_train)).type(torch.LongTensor)
    y_val = torch.from_numpy(np.array(y_val)).type(torch.LongTensor)
    
    # Compute the AUC of validation set
    output = model(X_val, val_delta_mat, None)
    print('Val Auc: {:.4f}'.format(metrics.roc_auc_score(y_val, output.detach().numpy())))
    
    for epoch in range(num_epoch):
        print("-" * 40)
        print('Epoch {}/{}'.format(epoch+1, num_epoch))
        
        # training stage
        train_loss, train_corrects = 0, 0
        
        for step in range(train_num//batch_size+1):
            if (step+1)*batch_size <= train_num:
                X = X_train[:, int(step*batch_size):int((step+1)*batch_size), :]
                y = y_train[int(step*batch_size):int((step+1)*batch_size)]
                Delta = train_delta_mat[:, int(step*batch_size):int((step+1)*batch_size), :]
            else:
                X = X_train[:, int(step*batch_size):, :]
                y = y_train[int(step*batch_size):]
                Delta = train_delta_mat[:, int(step*batch_size):, :]
            
            output = model(X, Delta, None)
            y_hat = torch.cat((1-output,output),dim=1)
            loss = criterion(y_hat, y)
            y_predict = (output > threshold).float() * 1
            
            optimizer.zero_grad()
            loss.backward()
            grad_clipping(model, 1)
            optimizer.step()
            train_loss += loss.item() * len(y)
            train_corrects += torch.sum(y_predict.squeeze() == y)
        
        # Compute the mean of loss and accuracy for every epoch
        train_loss_all.append(train_loss / train_num)
        train_acc_all.append(train_corrects.double().item() / train_num)
        
        print("Train Loss: {:.4f} Train Acc{:.4f}".format(train_loss_all[-1], train_acc_all[-1]))
        
        if (epoch+1) % 5 == 0:
            # Compute the AUC of validation set
            output = model(X_val, val_delta_mat, None)
            y_predict = (output > threshold).float() * 1
            print('Val Auc: {:.4f}, Val Score 1: {:.4f}'.format(metrics.roc_auc_score(y_val, output.detach().numpy()), score1(y_predict, y_val)))
    
    # save
    torch.save(model.state_dict(), 'E:/WashU/Research/ICU/modelParams/{}_{}_{}_{}_{}_{}_{}.pth'.format(model.imputeMethod, 
                                                                                                          model.scaleMethod, 
                                                                                                          model.name, 
                                                                                                          model.num_hiddens, 
                                                                                                          batch_size, 
                                                                                                          lr, 
                                                                                                          num_epoch))
    
    return model


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