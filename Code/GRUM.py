import torch
import torch.nn as nn

torch.manual_seed(0)

class GRUMCell(nn.Module):
    def __init__(self, num_inputs, num_hiddens):
        super(GRUMCell, self).__init__()
        self.num_inputs = num_inputs
        self.num_hiddens = num_hiddens
        # self.num_outputs = num_outputs
        def normal(shape):
            return torch.randn(size=shape) * 0.01
        def three():
            return (nn.Parameter(normal((num_inputs, num_hiddens))), 
                    nn.Parameter(normal((num_hiddens, num_hiddens))), 
                    nn.Parameter(torch.zeros(num_hiddens)))
        self.W_xz, self.W_hz, self.b_z = three() # Parameters of update gate
        self.W_xr, self.W_hr, self.b_r = three() # Parameters of reset gate
        self.W_xh, self.W_hh, self.b_h = three() # Parameters of candidate hidden state
        # Parameters of decay vector
        self.W_beta = nn.Parameter(normal((num_inputs, num_hiddens)))
        self.b_beta = nn.Parameter(torch.zeros(num_hiddens))
        # Parameters of output layer
        # self.W_hq = nn.Parameter(normal((num_hiddens, num_outputs)))
        # self.b_q = nn.Parameter(torch.zeros(num_outputs))

    def forward(self, X, Delta, H):
        H.detach()
        beta = torch.exp(torch.minimum(torch.zeros(self.num_hiddens), Delta @ self.W_beta + self.b_beta))
        #H = beta * H
        Z = torch.sigmoid((X @ self.W_xz) * beta + (H @ self.W_hz) + self.b_z)
        R = torch.sigmoid((X @ self.W_xr) * beta + (H @ self.W_hr) + self.b_r)
        H_tilde = torch.tanh((X @ self.W_xh) * beta + ((R * H) @ self.W_hh) + self.b_h)
        H = Z * H + (1 - Z) * H_tilde
        # H.detach()
        # Y = H @ self.W_hq + self.b_q
        return H

class GRUMModel(nn.Module):
    def __init__(self, num_inputs, num_hiddens):
        super(GRUMModel, self).__init__()
        self.name = 'GRUM'
        self.num_hiddens = num_hiddens
        self.grumcell = GRUMCell(num_inputs, num_hiddens)

    def forward(self, X, Delta, H):
        if H is None:
            H_new = torch.zeros(X.shape[1], self.num_hiddens)
        else:
            H_new = H
        H = torch.tensor([])
        for index in range(X.shape[0]):
            H_new = self.grumcell(X[index], Delta[index], H_new)
            H = torch.cat((H, H_new), dim = 0)
        return H