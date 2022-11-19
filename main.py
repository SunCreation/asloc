import torch as th
from torch import nn



class Testnet(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear()
        self.linear2 = nn.Linear()