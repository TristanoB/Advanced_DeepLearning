import torch.nn as nn  


# the Layer must herite from nn.Module 
class LastLayer(nn.Module) : 
    # architecture : init + forward 
    
    def __init__(self, out_features=2) :
        # init the parent class 
        super().__init__()
        self.fc = nn.LazyLinear(out_features)
        self.out_features= out_features
    
    # use a forward linear layer ? 
    def forward(self,x) :
        return self.fc(x)