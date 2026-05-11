from utils import load_data,sample_training_data,get_boundary_points,get_RK_coefficients
from PINN_RK import AllenCahnDiscretePINN
from config import N_TRAIN,Q
import torch
import numpy as np
import time
import matplotlib.pyplot as plt
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# 1. 加载数据
mat_file_path = 'D:\AC.mat'
print(f"从 {mat_file_path} 加载数据...")
x, u_t0, u_t1 = load_data(mat_file_path)
print("-"*20)

# 2. 采样训练数据和边界点
print("采样训练数据和边界点...")
x_train, u_train = sample_training_data(x, u_t0, N_train=N_TRAIN)
x_boundary = get_boundary_points()
print("-"*20)

# 3. 获取RK系数
print(f"读取 {Q} 阶段 Gauss-Legendre RK 系数...")
c, b, A = get_RK_coefficients(r"D:\Butcher_IRK10.txt")
# 🐋c, b, A = get_RK_coefficients(r"D:\Butcher_IRK100.txt")
print("-"*20)

# 4. 创建PINN模型
print("创建PINN模型...")
pinn = AllenCahnDiscretePINN(x_train, u_train, x_boundary, c, b, A)

# # 5. 训练模型
# print("模型训练...")
# start = time.time()
# pinn.train()
# end = time.time()
# all=end-start
# print(f"训练共用时: {all}")

# # 6. 保存模型
# model_path = os.path.join(BASE_DIR, 'ac10_pinn.pth')  # 🐋model_path = 'ac100_pinn.pth'
# torch.save(pinn.model.state_dict(), model_path)
# print(f"模型已保存到 {model_path}")


# ==========================================================
# ⭐ 读取已保存模型（用于评估/测试）
# ==========================================================
print("加载已训练模型...")

model_path = os.path.join(BASE_DIR, 'ac10_pinn.pth')  # 🐋model_path = 'ac100_pinn.pth'
state_dict = torch.load(model_path, map_location=torch.device('cpu'))
pinn.model.load_state_dict(state_dict)
pinn.model.eval()

print(f"模型已从 {model_path} 加载完成")
print("-"*20)


# 7. 预测并评估
print("开始预测...")
x_test = x  # 512个测试点(文件中所有的x点)
u_pred = pinn.predict(x_test) 

# 计算相对L2误差
l2_error = np.linalg.norm(u_pred - u_t1) / np.linalg.norm(u_t1)
print(f"相对 L2 误差: {l2_error:.3e}")

# 画图
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(x, u_t1, 'b-', linewidth=2, label='Exact Solution (t=0.2)')
# 🐋ax.plot(x, u_t1, 'b-', linewidth=2, label='Exact Solution (t=0.9)')
ax.plot(x_test, u_pred, 'r--', linewidth=2, label=f'PINN Prediction (Q={Q})')
ax.set_xlabel('x', fontsize=12)
ax.set_ylabel('u(x, t=0.2)', fontsize=12)
# 🐋ax.set_ylabel('u(x, t=0.9)', fontsize=12)
ax.set_title(f'Allen-Cahn Equation: Discrete Time Model\n'
             f'Relative L2 Error = {l2_error:.3e}', fontsize=14)
ax.legend(fontsize=12)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(BASE_DIR, 'ac10_pinn.png'), dpi=300, bbox_inches='tight')
# 🐋plt.savefig('ac100_pinn.png', dpi=300, bbox_inches='tight')


'''
Iter  3400 Total 3.405e-03
训练共用时: 345.31871795654297秒
相对 L2 误差: 2.222e-03
'''
