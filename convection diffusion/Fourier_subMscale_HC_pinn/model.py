import torch
import torch.nn as nn
import math

class Sin(nn.Module):
    def forward(self, x):
        return torch.sin(x)
    
# ----------------------------
# 单个子网络
# ----------------------------
class SubNet(nn.Module):
    def __init__(self, in_dim, layers, k):
        '''
        :param in_dim: 2
        :param layers: list[10,25,20,20,15]
        :param k: a scalor factor
        '''
        super().__init__()
        self.k = k

        self.first_weight = nn.Parameter(
            torch.randn(layers[0] // 2, in_dim)   
        )
        # self.first_weight:tensor(5,2)

        net = []
        for i in range(len(layers) - 1):
            net.append(nn.Linear(layers[i], layers[i+1]))
            if i != len(layers) - 2:
                net.append(Sin())
        self.net = nn.Sequential(*net)

    def forward(self, x):
        # x:tensor(batch size,in_dim=2)
        # Fourier feature mapping
        x = self.k * x
        z = torch.matmul(x, self.first_weight.T)    # x*self.first_weight.T
        z = torch.cat([torch.cos(z), torch.sin(z)], dim=1)
        return self.net(z)


# ----------------------------
# Mscale PINN
# ----------------------------
class MscalePINN(nn.Module):
    def __init__(self, in_dim=2, n_subnets=20):
        super().__init__()

        layers = [10, 25, 20, 20, 15, 1]
        ks = list(range(1, n_subnets + 1))

        self.subnets = nn.ModuleList(
            [SubNet(in_dim, layers, k) for k in ks]
        )

        self.linear = nn.Linear(n_subnets, 1)

    def forward(self, x):
        outs = [net(x) for net in self.subnets]
        out = torch.cat(outs, dim=1)
        return self.linear(out)
        # tensor(batch size,1)
