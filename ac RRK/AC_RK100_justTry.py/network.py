import torch.nn as nn
from collections import OrderedDict
from config import DTYPE

class Network(nn.Module):
    def __init__(self,input_size,hide_size,out_size,depth,act=nn.Tanh):
        super().__init__()

        layers=[("input",nn.Linear(input_size,hide_size,dtype=DTYPE))]
        layers.append(('input_activation', act()))

        for i in range(depth):
            layers.append((f'hidden_{i}',nn.Linear(hide_size,hide_size,dtype=DTYPE)))
            layers.append((f'activation_{i}',act()))

        layers.append(('output',nn.Linear(hide_size,out_size,dtype=DTYPE)))

        self.layers=nn.Sequential(OrderedDict(layers))
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m,nn.Linear):
                # Xavier
                std=(2/(m.in_features+m.out_features))**0.5
                nn.init.trunc_normal_(m.weight,0.0,std,-2*std,2*std)
                # 偏置设为0
                nn.init.constant_(m.bias,0.0)

    def forward(self,x):
        return self.layers(x)