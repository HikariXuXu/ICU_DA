import torch
import torch.nn as nn
from GRUM import GRUMModel

torch.manual_seed(0)

class PretrainGenerator(nn.Module):
    def __init__(self, num_inputs, num_hiddens, slice_gaps, drop_prob):
        super(PretrainGenerator, self).__init__()
        self.num_inputs = num_inputs
        self.num_hiddens = num_hiddens
        self.grum_model = GRUMModel(num_inputs, num_hiddens)
        self.slice_gaps = slice_gaps
        self.time_steps = 2880//slice_gaps
        self.drop_prob = drop_prob
        
        def normal(shape):
            return torch.randn(size=shape) * 0.01
        
        # Parameters of fully connected layer
        self.W_fc = nn.ParameterList([nn.Parameter(normal((num_hiddens, num_inputs))) for i in range(self.time_steps)])
        self.b_fc = nn.ParameterList([nn.Parameter(torch.zeros(num_inputs)) for i in range(self.time_steps)])
        '''
        self.W_fc = nn.Parameter(normal((num_hiddens, num_inputs)))
        self.b_fc = nn.Parameter(torch.zeros(num_inputs))
        '''
    
    def forward(self, X, Delta, H):
        states = self.grum_model(X, Delta, H)
        states = states.view(X.shape[0], X.shape[1], self.num_hiddens)
        # Full connect
        f_dropout = nn.Dropout(p = self.drop_prob)
        imputed_result = torch.tensor([])
        for i in range(states.shape[0]):
            out_imputed = f_dropout(states[i]) @ self.W_fc[i] + self.b_fc[i]
            imputed_result = torch.cat((imputed_result, out_imputed), dim = 0)
        imputed_result = imputed_result.view(X.shape[0], X.shape[1], self.num_inputs)
        
        return imputed_result
    
    
class Generator(nn.Module):
    def __init__(self, num_inputs, num_hiddens, slice_gaps, z_dim, drop_prob):
        super(Generator, self).__init__()
        self.num_inputs = num_inputs
        self.num_hiddens = num_hiddens
        self.grum_model = GRUMModel(num_inputs, num_hiddens)
        self.slice_gaps = slice_gaps
        self.time_steps = 2880//slice_gaps
        self.z_dim = z_dim
        self.drop_prob = drop_prob
        
        def normal(shape):
            return torch.randn(size=shape) * 0.01
        
        # Parameters of random vector to X
        self.W_rv2x = nn.Parameter(normal((z_dim, num_inputs)))
        self.b_x = nn.Parameter(torch.zeros(num_inputs))
        # Parameters of fully connected layer
        self.W_fc = nn.ParameterList([nn.Parameter(normal((num_hiddens, num_inputs))) for i in range(self.time_steps)])
        self.b_fc = nn.ParameterList([nn.Parameter(torch.zeros(num_inputs)) for i in range(self.time_steps)])
        '''
        self.W_fc = nn.Parameter(normal((num_hiddens, num_inputs)))
        self.b_fc = nn.Parameter(torch.zeros(num_inputs))
        '''
    
    def forward(self, rv, H, batch_size):
        if rv is None:
            rv = torch.randn((self.time_steps, batch_size, self.z_dim))
        X = torch.tensor([])
        for i in range(rv.shape[0]):
            X = torch.cat((X, rv[i] @ self.W_rv2x + self.b_x), dim = 0)
        X = X.view(self.time_steps, batch_size, self.num_inputs)
        Delta = self.slice_gaps*torch.ones(self.time_steps, batch_size, self.num_inputs)
        
        states = self.grum_model(X, Delta, H)
        states = states.view(X.shape[0], X.shape[1], self.num_hiddens)
        # Full connect
        f_dropout = nn.Dropout(p = self.drop_prob)
        imputed_result = torch.tensor([])
        for i in range(states.shape[0]):
            out_imputed = f_dropout(states[i]) @ self.W_fc[i] + self.b_fc[i]
            imputed_result = torch.cat((imputed_result, out_imputed), dim = 0)
        imputed_result = imputed_result.view(X.shape[0], X.shape[1], self.num_inputs)
        
        # batch_normalization???
        
        return imputed_result, Delta, states[-1]


class Discriminator(nn.Module):
    def __init__(self, num_inputs, num_hiddens, drop_prob):
        super(Discriminator, self).__init__()
        self.num_inputs = num_inputs
        self.num_hiddens = num_hiddens
        self.grum_model = GRUMModel(num_inputs, num_hiddens)
        self.drop_prob = drop_prob
        
        def normal(shape):
            return torch.randn(size=shape) * 0.01

        # Parameters of fully connected layer
        self.W_fc = nn.Parameter(normal((num_hiddens, 1)))
        self.b_fc = nn.Parameter(torch.zeros(1))
        
    def forward(self, X, Delta, H):
        states = self.grum_model(X, Delta, H)
        states = states.view(X.shape[0], X.shape[1], self.num_hiddens)
        # Full connect
        f_dropout = nn.Dropout(p = self.drop_prob)
        real_logits = f_dropout(states[-1]) @ self.W_fc + self.b_fc
        real_probs = torch.sigmoid(real_logits)
        
        return real_probs, real_logits




