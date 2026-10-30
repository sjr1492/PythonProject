import torch
import torch.nn as nn
import torch.optim as optim

def function(x):
    return x ** 2
torch.manual_seed(42)
x_train = torch.rand(100,1)
y = function(x_train)
nosie = torch.randn(100,1) * 0.3
y_train = y + nosie

class Net(nn.Module):
    def __init__(self,input_dim,width,output_dim):
        super(Net, self).__init__()
        self.hidden = nn.Linear(input_dim,width)
        self.relu = nn.ReLU()
        self.output = nn.Linear(width,output_dim)
    def forward(self, x):
        out = self.hidden(x)
        out = self.relu(out)
        out = self.output(out)
        return out
model = Net(input_dim=1,width=15,output_dim=1)

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(),lr=0.01)

for epoch in range(1000):
    optimizer.zero_grad()
    pre = model(x_train)
    loss = criterion(pre,y_train)
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 100 == 0:
        print(f"第 {epoch + 1:4d} 轮 | 当前误差 (Loss): {loss.item():.4f}")




