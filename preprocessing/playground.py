import torch_geometric.nn as pyg_nn
import torch
import torch.nn as nn
import torch.nn.functional as F

# Define LSTM with Attention for temporal processing
class TemporalAttentionLayer(nn.Module):
    def __init__(self, in_features, out_features):
        super(TemporalAttentionLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        self.LSTM = nn.LSTM(in_features, out_features, batch_first=True)
        self.attention = nn.Linear(out_features, 1)
        
    def forward(self, x):
        lstm_out, _ = self.LSTM(x)
        attention_w = F.softmax(self.attention(lstm_out), dim=1)
        out = torch.sum(attention_w * lstm_out, dim=1)
        return out


# Define the complete GNN model using PyTorch Geometric
class SpatioTemporalGNN_pyg(nn.Module):
    def __init__(self, node_features, time_features, hidden_features, num_classes):
        super(SpatioTemporalGNN_pyg, self).__init__()
        
        # Spatial Graph Attention Layers
        self.gat1 = pyg_nn.GATConv(node_features, hidden_features, heads=1)
        self.gat2 = pyg_nn.GATConv(hidden_features, hidden_features, heads=1)
        
        # Temporal Attention Layer
        self.temporal_attention = TemporalAttentionLayer(hidden_features, hidden_features)
        
        # Output Layer
        self.fc = nn.Linear(hidden_features, num_classes)
        
    def forward(self, x, edge_index):
        x = F.relu(self.gat1(x, edge_index))
        print(x.shape)
        x = F.relu(self.gat2(x, edge_index))
        print(x.shape)

        
        # Reshaping the tensor to fit the temporal attention layer
        # Assuming the batch size is 1 for this example
        x = x.unsqueeze(0)
        print(x.shape)

        
        x = self.temporal_attention(x)
        print(x.shape)

        
        x = self.fc(x)
        return F.softmax(x, dim=1)

# Initialize the model
node_features = 100  # Number of node features
time_features = 10000  # Number of time steps in the 10s EEG signal
hidden_features = 64  # Number of hidden features
num_classes = 2  # Number of classes (seizure or not)

# Initialize the model using PyTorch Geometric
model_pyg = SpatioTemporalGNN_pyg(node_features, time_features, hidden_features, num_classes)
# Example data
num_nodes = 8  # Number of EEG channels
batch_size = 1  # Single example for demonstration

# Example data
# Random EEG data (num_nodes, node_features)
x_pyg = torch.rand(num_nodes, node_features)

# Random adjacency matrix for demonstration (2, num_edges)
edge_index_pyg = torch.randint(0, num_nodes, (2, num_nodes * 2))

# Forward pass through the model
output_pyg = model_pyg(x_pyg, edge_index_pyg)

print(output_pyg)