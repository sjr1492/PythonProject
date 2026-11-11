import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False

def function(x):
    return x ** 2

torch.manual_seed(42)
x_train = torch.rand(100, 1)
y = function(x_train)
noise = torch.randn(100, 1) * 0.3
y_train = y + noise

class Net(nn.Module):
    def __init__(self, input_dim, width, output_dim):
        super(Net, self).__init__()
        self.layer1 = nn.Linear(input_dim, width)
        self.relu1 = nn.ReLU()
        self.layer2 = nn.Linear(width, width)
        self.relu2 = nn.ReLU()
        self.layer3 = nn.Linear(width, output_dim)

    def forward(self, x):
        out = self.layer1(x)
        out = self.relu1(out)

        out = self.layer2(out)
        out = self.relu2(out)

        out = self.layer3(out)
        return out

model = Net(input_dim=1, width=15, output_dim=1)

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

for epoch in range(1000):
    optimizer.zero_grad()
    pred = model(x_train)
    loss = criterion(pred, y_train)
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 100 == 0:
        print(f"第 {epoch + 1:4d} 轮 | 当前误差 (Loss): {loss.item():.4f}")

model.eval()
with torch.no_grad():
    x_test = torch.linspace(0, 1, 200).unsqueeze(1)
    y_true = function(x_test)
    y_pred = model(x_test)

plt.figure(figsize=(8, 5))
plt.scatter(x_train, y_train, c="gray", label="带噪声训练样本", alpha=0.6)
plt.plot(x_test, y_true, "r-", label="真实函数 y=x²", linewidth=2)
plt.plot(x_test, y_pred, "b--", label="三层网络拟合结果", linewidth=2)
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.title("三层神经网络拟合 y=x²")
plt.grid(True, alpha=0.3)
plt.show()




