#%%
import pandas
data = pandas.read_csv("allo_data.csv")
data.head()
# %%
import datetime
from dateutil.parser import parse
data["Date"] = data["Date"].apply(parse)
data.head()
# %%
import matplotlib.pyplot as plt
import numpy as np

# def myplot(inputs,col,val_x,val_y):
#     tomatos = set(inputs[col])
#     plt.figure(figsize=(20,10))
#     for tomato in (i for i in tomatos if not pd.isnull(i)):
#         # print(tomato)
#         check_set = set(inputs[inputs[col]==tomato]['Sample_no'])
#         input_sample = inputs[np.vectorize(lambda x: x in check_set)(inputs['Sample_no'])]
#         plt.scatter(input_sample[val_x],input_sample[val_y],s=10,label=tomato)
#     plt.legend(bbox_to_anchor=(1.07, 1))
#     plt.xlabel(f'{val_x}')
#     plt.ylabel(f'{val_y}')
#     plt.show()

plt.figure(figsize=(20,10))
for col in data.columns:
    if col=="Date":continue
    plt.plot(data['Date'],data[col],label=col)

plt.legend(bbox_to_anchor=(1.07, 1))
plt.show()
# %%
def myplot(data,cols):
    plt.figure(figsize=(20,10))
    for col in cols:
        if col=="Date":continue
        plt.plot(data['Date'],data[col],label=col)

    plt.legend(bbox_to_anchor=(1.07, 1))
    plt.show()

myplot(data[:120],data.columns) # 전체
myplot(data[:120],['VWO', 'BND']) # 카나리아
myplot(data[:120],['SPY', 'QQQ', 'IWM', 'VGK', 'EWJ', 'EEM', 'VNQ','GLD', 'DBC', 'HYG', 'LQD', 'TLT']) # 공격
myplot(data[:120],['SHY', 'IEF', 'LQD.1']) # 방어
# %%
myplot(data[120:240],data.columns)
myplot(data[120:240],['VWO', 'BND'])
myplot(data[120:240],['SPY', 'QQQ', 'IWM', 'VGK', 'EWJ', 'EEM', 'VNQ','GLD', 'DBC', 'HYG', 'LQD', 'TLT'])
myplot(data[120:240],['SHY', 'IEF', 'LQD.1'])
# %%
money = data.loc[:,['VWO', 'BND', 'SPY', 'QQQ', 'IWM', 'VGK', 'EWJ', 'EEM', 'VNQ','GLD', 'DBC', 'HYG', 'LQD', 'TLT', 'SHY', 'IEF', 'LQD.1']].to_numpy()

# %%
import torch as th
from torch import nn
import torch.optim as optim
import torch.nn.functional as F


class Testnet(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(17,50)
        self.linear2 = nn.Linear(50,17)
    
    def forward(self, data):
        data = self.linear1(data)
        data = self.linear2(data)
        return data


model = Testnet()
optimizer = optim.Adam(model.parameters(), lr=0.001)
# %%
model = Testnet()
optimizer = optim.Adam(model.parameters(), lr=0.001)
data_ = th.tensor(money).type(th.float32)
input_ = data_[:-1]
output_ = data_[1:]
model.train()
batch_size = 16
for epoch in range(10):
    total = 0
    for idx in range(len(input_)//16):
        optimizer.zero_grad()
        pred = model(input_[idx*16:idx*16+batch_size])
        loss:th.Tensor = sum((output_[idx*16:idx*16+batch_size] - pred)**2)
        loss:th.Tensor = sum(loss)
        total += loss.item()
        loss.backward()
        optimizer.step()
    print(total)





# %%
for epoch in range(100):
    total = 0
    for idx in range(len(input_)//16):
        optimizer.zero_grad()
        pred = model(input_[idx*16:idx*16+batch_size])
        loss:th.Tensor = sum((output_[idx*16:idx*16+batch_size] - pred)**2)
        loss:th.Tensor = sum(loss)
        total += loss.item()
        loss.backward()
        optimizer.step()
    print(total)
# %%
import pandas as pd
l2 = pd.read_csv("l2.csv")
# %%
l2

#%%
l2.value_counts()
# %%
l2 = l2.to_numpy()
#%%
l2 = th.tensor(l2)
# %%
len(input_)
# %%
import torch as th
from torch import nn
import torch.optim as optim
import torch.nn.functional as F

class TestnetAD(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Linear(17,64)
        self.lstm = nn.LSTM(64,64)
        self.linear1 = nn.Linear(64,32)
        self.linear2 = nn.Linear(64,1)
    
    def forward(self, data:th.Tensor):
        data = self.embed(data)
        data = self.lstm(data)[0][:,-1]
        # print(data.size())
        # data = data.flatten()
        data = self.linear1(data)
        data = F.relu(data)
        data = self.linear2(data)
        return F.sigmoid(data)
# %%

model = TestnetAD()
optimizer = optim.Adam(model.parameters(), lr=0.001)
model.train()
# batch_size = 16
for epoch in range(60):
    total = 0
    acc = 0
    for idx in range(len(input_)-213):
        optimizer.zero_grad()
        pred = model(input_[idx:idx+12].unsqueeze(0))
        # print(pred,l2[idx])
        check:th.Tensor = pred - l2[idx]
        # print(pred,l2[idx])
        loss:th.Tensor = F.binary_cross_entropy(pred,l2[idx].unsqueeze(0).type(th.float32))
        # print(loss)
        # break
        # loss:th.Tensor = sum(loss)
        acc += round(abs(check.item()))
        loss = loss**2
        loss.backward()
        total += loss.item()
        optimizer.step()
    print(total)
    print((419 - acc)/419)

#%%
for epoch in range(60):
    total = 0
    acc = 0
    for idx in range(len(input_)-213):
        optimizer.zero_grad()
        pred = model(input_[idx:idx+12].unsqueeze(0))
        # print(pred,l2[idx])
        check:th.Tensor =pred-l2[idx]
        # print(pred,l2[idx])
        loss:th.Tensor = F.binary_cross_entropy(pred,l2[idx].unsqueeze(0).type(th.float32))
        # print(loss)
        # break
        # loss:th.Tensor = sum(loss)
        acc += round(abs(check.item()))
        loss = loss**2
        loss.backward()
        total += loss.item()
        optimizer.step()
    print(total)
    print((419 - acc)/419)
# %%
total = 0
acc = 0
for idx in range(len(input_)-213,len(input_)-13):
    optimizer.zero_grad()
    pred = model(input_[idx:idx+12].unsqueeze(0))
    loss:th.Tensor =(pred-l2[idx])
    # loss:th.Tensor = sum(loss)
    acc += round(abs(loss.item()))
    # print(loss.item())
    # print(acc)
    # print('------------------')
    loss = loss**2
    total += loss.item()
    # loss.backward()
    # optimizer.step()
# print(total)
print(acc)
print((200 - acc)/200)
# %%
len(input_)-13
# %%
