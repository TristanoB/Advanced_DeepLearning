import torch.nn as nn  


# the Layer must herite from nn.Module 
class LastLayer(nn.Module) : 
    # architecture : init + forward 
    
    def __init__(self, x, nn.Module, in_features, out_features=2) :
        self.out_features = 2 
        self.in_features = x.shape[1] # Suppose (B,D)
        self.fc = nn.Linear()
        self.x = x 
    
    # use a forward linear layer ? 
    def forward(self,x) :
        return nn.Linear(x.shape,self.out_features) 
    