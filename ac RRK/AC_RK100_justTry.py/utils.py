import torch
import numpy as np
import scipy.io as sio
from config import N_TRAIN,DEVICE,DTYPE,Q

def load_data(mat_file_path):
    mat_data = sio.loadmat(mat_file_path)
    x = mat_data['x'].flatten()  # x:array(512,)
    tt = mat_data['tt']  
    uu = mat_data['uu']  
    idx_t0 = 20  
    idx_t1 = 180
    u_t0 = uu[:, idx_t0]   # t=0.1    u_t0:(512,)
    u_t1 = uu[:, idx_t1]   # t=0.9    u_t1:(512,)
    return x, u_t0, u_t1

def sample_training_data(x, u0, N_train=N_TRAIN):
    """在 t=0.1 随机采样 200 个训练点"""
    idx = np.random.choice(len(x), N_train, replace=False)
    x_train = x[idx]  # (200,)
    u_train = u0[idx] # (200,)
    x_train = torch.tensor(x_train, dtype=DTYPE, requires_grad=True, device=DEVICE).unsqueeze(1)  # tensor(200,1)
    u_train = torch.tensor(u_train, dtype=DTYPE, device=DEVICE).unsqueeze(1)         # tensor(200,1)
    return x_train, u_train

def get_boundary_points():
    """获取固定边界点 x=-1 和 x=1"""
    x_boundary = np.array([-1.0, 1.0]) 
    x_boundary = torch.tensor(x_boundary, dtype=DTYPE, device=DEVICE).unsqueeze(1)  # tensor(2, 1)
    x_boundary.requires_grad = True
    return x_boundary

def get_RK_coefficients(txt_file_path):
    """
    从文件读取Runge-Kutta系数 (c, b, A)
    "D:\Butcher_IRK100.txt"
    """
    coeffs = np.loadtxt(txt_file_path, dtype=np.float64) #(Q^2+2Q,)
    total_len = len(coeffs)
    Q_inferred = int(np.sqrt(total_len+1)-1)
    if Q_inferred != Q:
        print(f"警告: 文件中推断的Q={Q_inferred}与全局Q={Q}不匹配，将使用文件中的Q={Q_inferred}")
    
    # 提取
    A_end_idx = Q_inferred ** 2
    A = coeffs[:A_end_idx].reshape(Q_inferred, Q_inferred) #(Q_inferred,Q_inferred)
    b_end_idx = A_end_idx + Q_inferred
    b = coeffs[A_end_idx:b_end_idx]     # (Q_inferred,)
    c = coeffs[b_end_idx:] #(Q_inferred,)
    
    return c, b, A