import torch
import torch.nn as nn


class MRLoss(nn.Module):
    '''Masked Reconstruction Loss'''
    def __init__(self):
        super(MRLoss, self).__init__()
        pass
    
    def forward(self, X, M, X_g):
        mr_loss = 0
        for i in range(X.shape[0]):
            mr_loss += torch.norm(X[i] * M[i] - X_g[i] * M[i])
        mr_loss = mr_loss / X.shape[0]
        return mr_loss
    
    
class DLoss(nn.Module):
    '''Discriminative Loss'''
    def __init__(self):
        super(DLoss, self).__init__()
        pass
    
    def forward(self, D_real_logits, D_fake_logits):
        d_loss_real = - torch.mean(D_real_logits)
        d_loss_fake = torch.mean(D_fake_logits)
        d_loss = d_loss_real + d_loss_fake
        return d_loss


class GLoss(nn.Module):
    '''Loss for Generator'''
    def __init__(self):
        super(GLoss, self).__init__()
        pass
    
    def forward(self, D_fake_logits):
        g_loss = - torch.mean(D_fake_logits)
        return g_loss


class ImpLoss(nn.Module):
    '''Imputation Loss'''
    def __init__(self, g_loss_lambda):
        super(ImpLoss, self).__init__()
        self.g_loss_lambda = g_loss_lambda
    
    def forward(self, X, M, X_imp, imputed_fake_logits):
        imp_loss = 0
        for i in range(X.shape[0]):
            imp_loss += torch.norm(X[i] * M[i] - X_imp[:,i,:] * M[i]) - self.g_loss_lambda * imputed_fake_logits[i]
        imp_loss = imp_loss / X.shape[0]
        return imp_loss



