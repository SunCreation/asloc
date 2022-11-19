class TestnetAD(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(17*12,50)
        self.linear2 = nn.Linear(50,1)
    
    def forward(self, data:th.Tensor):
        data = data.flatten()
        data = self.linear1(data)
        data = F.relu(data)
        data = self.linear2(data)
        return F.sigmoid(data)