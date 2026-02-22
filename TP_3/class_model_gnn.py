import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.nn as graphnn

# Define model ( in your class_model_gnn.py)
class StudentModel(nn.Module):
    def __init__(self):
        super(StudentModel, self).__init__()
        n_features = 50
        hidden_dim = 256
        n_classes = 121
        self.conv1 = graphnn.GCNConv(n_features, hidden_dim)
        self.att1 = graphnn.GATConv(n_features,hidden_dim,heads=3)
        self.conv2 = graphnn.GCNConv(hidden_dim, hidden_dim)
        self.att2 = graphnn.GATConv(hidden_dim,hidden_dim,heads=3)
        self.conv3 = graphnn.GCNConv(hidden_dim, hidden_dim)
        self.att3 = graphnn.GATConv(hidden_dim,hidden_dim,heads=3)
        self.fc1 = nn.Linear(hidden_dim, n_classes) 
        self.elu = nn.ELU()

    def forward(self, x, edge_index):
        x = self.att1(x, edge_index)
        x = self.elu(x)
        x = self.att2(x, edge_index)
        x = self.elu(x)
        x = self.att3(x, edge_index)
        x = self.elu(x)
        x = self.fc1(x)
        return x


# Initialize model
model = StudentModel()

## Save the model
torch.save(model.state_dict(), "model.pth")


### This is the part we will run in the inference to grade your model
## Load the model
model = StudentModel()  # !  Important : No argument
model.load_state_dict(torch.load("model.pth", weights_only=True))
model.eval()
print("Model loaded successfully")
