import torch

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

Q = 10  # 🐋Q = 100
DT = 0.1   # 🐋DT = 0.8

N_LAYERS = 4
N_NEURONS = 100   # 🐋N_NEURONS = 200
N_TRAIN = 200    #   # t=0.1时刻的采样点数量N_n

DTYPE = torch.float32  # 这里改成32了，应该会比原来快很多