import torch
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

Q = 100
DT = 0.8

N_LAYERS = 4
N_NEURONS = 200   # here,from 100 to 200,time ↑
N_TRAIN = 200   

DTYPE = torch.float64  