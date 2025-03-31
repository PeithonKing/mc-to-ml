import torch
import torch.nn as nn
import torch.nn.functional as F

class PolynomialModel(nn.Module):
    def __init__(self, n, d):
        # n is the number of variables
        # d is the degree of polynomial to use
        super().__init__()
        self.d = d
        self.conv_layers = nn.ModuleList([
            nn.Conv2d(n, 1, kernel_size=1)
            for _ in range(d)
        ])

    def forward(self, x):
        a = 1
        for i in range(self.d):
            a = self.conv_layers[i](a*x)            
        return a

class ExpAct(nn.Module):
    def __init__(self): super().__init__()
    def forward(self, x): return torch.exp(x)

class CustomConvLayer(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels,
            2 * in_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            padding_mode="circular",
            # bias=False
        )
        self.poly = PolynomialModel(n=2*in_channels, d=2)

    def forward(self, x, p):
        x = self.conv(x)
        if p:
            print("after conv")
            print(x)
        x = self.poly(x)
        if p:
            print("after poly")
            print(x)
        return x

class HamLearn(nn.Module):
    def __init__(self, in_channels=1):
        self.conv = CustomConvLayer(in_channels)
        self.lin = nn.Conv2d(1, 1, 1)
    def forward(self, x):
        x = self.conv(x)
        return self.lin(x)

class Model(nn.Module):
    def __init__(self, in_channels=1, hidden_dim=4, hamlearn=None):
        super().__init__()
        self.hamlearn = hamlearn if hamlearn else HamLearn(in_channels)
        self.exp = ExpAct()
        self.mlp = nn.Sequential(
            nn.Conv2d(1, hidden_dim, 1),
            nn.ReLU(),
            nn.Conv2d(hidden_dim, 1, 1),
        )

    def forward(self, x, T):
        x = self.hamlearn(x)
        x /= T.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        x = self.exp(x)
        x = self.mlp(x)
        return x

# class Model(nn.Module):
#     def __init__(self, in_channels=1, hidden_dim=1):
#         super().__init__()
#         self.conv = CustomConvLayer(in_channels)
#         self.lin1 = nn.Conv2d(1, 1, 1)
#         self.lin2 = nn.Conv2d(1, 1, 1)

#     def forward(self, x, T, p=False):
#         x = self.conv(x, p=p)
#         x = self.lin1(x)
#         if p:
#             print("after lin1")
#             print(x)
#         x /= T.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)

#         # return x

#         # a, b, c, d = x.shape
#         # x = x.view(a, -1)
#         # x = F.softmax(x, dim=1)
#         # x = x.view(a, b, c, d)
#         # return F.sigmoid(x)

#         return self.lin2(x)
