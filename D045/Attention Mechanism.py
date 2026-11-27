
import torch
import torch.nn as nn
import torch.nn.functional as F

class Attention(nn.Module):
    def __init__(self, input_dim):
        super().__init__()

        self.query = nn.Linear(input_dim, input_dim)
        self.key = nn.Linear(input_dim, input_dim)
        self.value = nn.Linear(input_dim, input_dim)

    def forward(self, x):
        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)

        scores = torch.matmul(Q, K.transpose(-2, -1))
        scores = scores / (K.size(-1) ** 0.5)

        weights = F.softmax(scores, dim=-1)

        output = torch.matmul(weights, V)

        return output, weights

# Sample sequential data
x = torch.rand(1, 5, 8)

model = Attention(input_dim=8)

output, weights = model(x)

print("Input Shape:", x.shape)
print("Attention Output:", output.shape)
print("Attention Weights:")
print(weights)