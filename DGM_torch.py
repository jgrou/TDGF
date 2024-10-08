import torch

# LSTM-like layer used in DGM 
class LSTMLayer(torch.nn.Module):
    def __init__(self, output_dim, input_dim):
        super(LSTMLayer, self).__init__() # create an instance of a Layer object 
        
        # weighting vectors for inputs original inputs x
        self.Uz = torch.nn.Linear(input_dim, output_dim, bias=False)
        self.Ug = torch.nn.Linear(input_dim, output_dim, bias=False)
        self.Ur = torch.nn.Linear(input_dim, output_dim, bias=False)
        self.Uh = torch.nn.Linear(input_dim, output_dim, bias=False)
        
        # weighting vectors for output of previous layer    
        self.Wz = torch.nn.Linear(output_dim, output_dim)
        self.Wg = torch.nn.Linear(output_dim, output_dim)
        self.Wr = torch.nn.Linear(output_dim, output_dim)
        self.Wh = torch.nn.Linear(output_dim, output_dim)

        # DGM layer - initialization
        torch.nn.init.xavier_uniform_(self.Uz.weight)
        torch.nn.init.xavier_uniform_(self.Ug.weight)
        torch.nn.init.xavier_uniform_(self.Ur.weight)
        torch.nn.init.xavier_uniform_(self.Uh.weight)
        torch.nn.init.xavier_uniform_(self.Wz.weight)
        torch.nn.init.xavier_uniform_(self.Wg.weight)
        torch.nn.init.xavier_uniform_(self.Wr.weight)
        torch.nn.init.xavier_uniform_(self.Wh.weight)
    
    def forward(self, S, X):
        # compute components of LSTM layer output
        Z = torch.tanh(self.Uz(X) + self.Wz(S))
        G = torch.tanh(self.Ug(X) + self.Wg(S))
        R = torch.tanh(self.Ur(X) + self.Wr(S))
        H = torch.tanh(self.Uh(X) + self.Wh(torch.mul(S, R)))
        return torch.mul((1.0 - G), H) + torch.mul(Z, S)

# Neural network architecture used in DGM - modification of Keras Model class
class DGMNet(torch.nn.Module):    
    def __init__(self, layer_width, input_dim):
        super(DGMNet, self).__init__() # create an instance of a Layer object 

        # define initial layer as fully connected (use Xavier initialization)
        self.W = torch.nn.Linear(input_dim, layer_width)
        torch.nn.init.xavier_uniform_(self.W.weight)
        
        # define intermediate LSTM layers
        self.LSTMLayer1 = LSTMLayer(layer_width, input_dim) # list does not work
        self.LSTMLayer2 = LSTMLayer(layer_width, input_dim)
        self.LSTMLayer3 = LSTMLayer(layer_width, input_dim)
        
        # define final layer as fully connected with a single output (function value)
        self.W4 = torch.nn.Linear(layer_width, 1, bias=False)
        self.W4.weight = torch.nn.Parameter(torch.ones((1, layer_width)))
        
    def forward(self, *args):
        # define input vector
        X = torch.concat([*args], 1)
        
        # call initial layer
        S0 = torch.tanh(self.W(X))
            
        # call intermediate LSTM layers
        S1 = self.LSTMLayer1(S0, X)
        S2 = self.LSTMLayer2(S1, X)
        S3 = self.LSTMLayer3(S2, X)
            
        return torch.log(torch.exp(self.W4(S3)) + 1)