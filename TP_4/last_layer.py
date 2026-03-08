import torch.nn as nn  


# from which torch class it must herites from ? 
def LastLayer(nn.Layer) : 
    # architecture : init + forward 
    
    def __init__(self,nn.Module,out_features=2) :
        self.out_features = 2 
        
        pass
    
    # use a forward linear layer ? 
    def forward(self,x) :
        return nn.Linear(x.shape,self.out_features) 
    