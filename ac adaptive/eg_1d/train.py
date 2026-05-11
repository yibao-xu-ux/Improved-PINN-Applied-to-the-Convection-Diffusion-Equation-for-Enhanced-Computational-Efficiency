import torch
import torch.nn as nn
from collections import OrderedDict
import numpy as np
import time
import os


DT = 0.1
N_STEPS = 10
N_LAYERS = 4
N_NEURONS = 100  # ⭐⭐⭐选择加宽模型的深度比增加层数对于优化大梯度区域更有效

N_COLLOCATIONS = 5000
N_INITIAL = 200
N_BOUND = 200

ADAM_ITER = 500
PRINT_ITER = 50
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE = torch.float32
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(SCRIPT_DIR)
MODEL_FOLDER = os.path.join(BASE_DIR, "1D_10_subModel")


# =====================================================
# 神经网络结构
# =====================================================
class Network(nn.Module):
    def __init__(self, input_size, hide_size, out_size, depth, act=nn.Tanh):
        super().__init__()
        layers = [("input", nn.Linear(input_size, hide_size, dtype=DTYPE)), ("act_in", act())]
        for i in range(depth):
            layers.append((f"hidden_{i}", nn.Linear(hide_size, hide_size, dtype=DTYPE)))
            layers.append((f"act_{i}", act()))
        layers.append(("output", nn.Linear(hide_size, out_size, dtype=DTYPE)))
        self.net = nn.Sequential(OrderedDict(layers))
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.net(x)


# =====================================================
# 单时间步 PINN（时间自适应方法 II 的“子网络”）
# =====================================================
class SingleStepPINN:
    def __init__(self, x_f, x_b1, x_b2, x_init, step_id):
        self.step_id = step_id
        self.model = Network(2, N_NEURONS, 1, N_LAYERS).to(DEVICE)
        self.x_f = x_f
        self.x_b1 = x_b1
        self.x_b2 = x_b2
        self.x_init = x_init
        self.iter = 0  # 初始化迭代计数器
        self.last_lbfgs_loss = None # 记录lbfgs最终步骤损失
        self.adam = torch.optim.Adam(self.model.parameters(), lr=1e-4)
        self.lbfgs = torch.optim.LBFGS(
            self.model.parameters(),
            max_iter=5000,
            tolerance_grad=1e-12,   # 可以更严格
            tolerance_change=np.finfo(float).eps,  # 很小，相当于禁用损失变化作为收敛依据
            # tolerance_grad=1e-6,
            # tolerance_change=1e-9,
            line_search_fn="strong_wolfe"
        )

    # -----------------------------
    # PDE residual
    # -----------------------------
    def loss_pde(self):
        u = self.model(self.x_f)
        grads = torch.autograd.grad(
            u, self.x_f, grad_outputs=torch.ones_like(u), create_graph=True
        )[0]
        u_x = grads[:, 0:1]
        u_t = grads[:, 1:2]
        u_xx = torch.autograd.grad(
            u_x, self.x_f, grad_outputs=torch.ones_like(u_x), create_graph=True
        )[0][:, 0:1]
        f = u_t - 0.0001 * u_xx + 5 * u**3 - 5 * u   # ⭐方程
        return torch.mean(f**2)

    # -----------------------------
    # Periodic boundary
    # -----------------------------
    def loss_bc(self):
        u1 = self.model(self.x_b1)
        u2 = self.model(self.x_b2)
        loss_u = torch.mean((u1 - u2)**2)
        du1 = torch.autograd.grad(u1, self.x_b1, grad_outputs=torch.ones_like(u1), create_graph=True)[0][:, 0:1]
        du2 = torch.autograd.grad(u2, self.x_b2, grad_outputs=torch.ones_like(u2), create_graph=True)[0][:, 0:1]
        loss_du = torch.mean((du1 - du2)**2)
        return loss_u + loss_du

    # -----------------------------
    # Initial condition
    # -----------------------------
    def loss_init(self):
        u_pred = self.model(self.x_init[:, :2])
        u_true = self.x_init[:, 2:3]
        return torch.mean((u_pred - u_true)**2)

    def total_loss(self):
        return self.loss_pde() + self.loss_bc() + 100 * self.loss_init()   # ⭐初值权重


    # ===== 模型保存 =====
    def save(self, folder=MODEL_FOLDER):
        os.makedirs(folder, exist_ok=True)
        path = os.path.join(folder, f"step_{self.step_id:03d}.pth")
        torch.save(self.model.state_dict(), path)

    # -----------------------------
    # 训练
    # -----------------------------
    def train(self):
        print(f"\n>>> Training step {self.step_id}")
        # 记录每个模型的训练时长
        start_time = time.time()
        
        # Adam优化器
        for _ in range(ADAM_ITER):
            self.adam.zero_grad()
            loss = self.total_loss()
            loss.backward()
            self.adam.step()
            self.iter += 1
            if self.iter % PRINT_ITER == 0:
                print(f"Iter {self.iter}, loss = {loss.item():.3e}")

        # L-BFGS优化器
        def closure():
            self.lbfgs.zero_grad()
            loss = self.total_loss()
            loss.backward()
            self.iter += 1  # 在每次L-BFGS迭代时增加迭代次数
            
            # ===== 记录 L-BFGS 当前 loss =====
            self.last_lbfgs_loss = loss.item()
            
            if self.iter % PRINT_ITER == 0:
                print(f"Iter {self.iter}, loss = {loss.item():.3e}")
            return loss

        self.lbfgs.step(closure)
        # 记录每个小模型训练时长
        end_time = time.time()
        duration = end_time - start_time
        self.save()

        print(f"Total iter {self.iter}")
        print(f"[L-BFGS final loss] = {self.last_lbfgs_loss:.3e}")
        print(f"Step {self.step_id} training time: {duration:.2f} seconds")

    # =================================================
    # 返回 torch.Tensor，用于下一个时间步
    # =================================================
    def predict(self, x, t):
        xt = torch.cat([x, t], dim=1)
        with torch.no_grad():
            return self.model(xt)


# =====================================================
# 时间自适应管理器（方法 II）
# =====================================================
class TimeAdaptivePINN:
    def __init__(self):
        self.models = []

    def train(self):
        total_start_time = time.time()  # 记录总训练时长
        prev_model = None
        for k in range(N_STEPS):
            t0 = k * DT
            t1 = (k + 1) * DT
            
            # collocation
            x = torch.rand(N_COLLOCATIONS, 1, device=DEVICE, dtype=DTYPE) * 2 - 1
            t = torch.rand(N_COLLOCATIONS, 1, device=DEVICE, dtype=DTYPE) * DT + t0
            x_f = torch.cat([x, t], 1).requires_grad_(True)
            
            # boundary
            tb = torch.rand(N_BOUND, 1, device=DEVICE, dtype=DTYPE) * DT + t0
            x_b1 = torch.cat([torch.full_like(tb, -1.0), tb], 1).requires_grad_(True)
            x_b2 = torch.cat([torch.full_like(tb, 1.0), tb], 1).requires_grad_(True)
            
            # initial
            xi = torch.rand(N_INITIAL, 1, device=DEVICE, dtype=DTYPE) * 2 - 1
            ti = torch.full_like(xi, t0)
            if k == 0:
                ui = xi**2 * torch.cos(torch.pi * xi)
            else:
                # 用上一个子网络的预测作为初值
                ui = prev_model.predict(xi, ti).detach()
            x_init = torch.cat([xi, ti, ui], 1)
            
            model = SingleStepPINN(x_f, x_b1, x_b2, x_init, k + 1)
            
            # ===== 参数延续=====
            if prev_model is not None:
                model.model.load_state_dict(
                    prev_model.model.state_dict()
                )
                
            model.train()
            self.models.append(model)
            prev_model = model
        
        total_end_time = time.time()
        total_duration = total_end_time - total_start_time
        print(f"Total training time: {total_duration:.2f} seconds")


# =====================================================
# 主程序
# =====================================================
# 👉只要其他文件import了train函数，就必须写下面这一行作为保护
# 👉否则，其他文件import train的时候就会开始运行训练逻辑
if __name__ == "__main__":  
    pinn = TimeAdaptivePINN()
    pinn.train()




'''
👉每个小模型配置
先进行500步adam优化,再使用lbfgs精修,lbffgs优化器的终止条件很严格
如果lbfgs的终止条件设置地更宽松,将会达到更快的优化速度


🐋文章写法
每个模型用时大...-...之间,共用时...,
(说明lbfgs优化器的配置),如果lbfgs的终止条件设置地更宽松,将会达到更快的优化速度
同等条件下,选择加宽模型的深度比增加层数对于优化大梯度区域更有效
即,关键是设施合适的模型深度和lbfgs终止条件

'''

'''
在这个配置下模型优化得很好,比高精度真解还要好(写在图片上)
[L-BFGS final loss] = 9.240e-04
Step 1 training time: 131.29 seconds-max
[L-BFGS final loss] = 2.575e-04
Step 2 training time: 33.84 seconds
[L-BFGS final loss] = 2.050e-04
Step 3 training time: 21.83 seconds-min
[L-BFGS final loss] = 9.238e-05
Step 4 training time: 95.89 seconds
[L-BFGS final loss] = 2.045e-04
Step 5 training time: 37.87 seconds
[L-BFGS final loss] = 3.589e-05
Step 6 training time: 98.55 seconds
[L-BFGS final loss] = 5.115e-05
Step 7 training time: 42.86 seconds
[L-BFGS final loss] = 2.454e-05
Step 8 training time: 146.78 seconds
[L-BFGS final loss] = 4.153e-05
Step 9 training time: 67.52 seconds
[L-BFGS final loss] = 4.854e-05
Step 10 training time: 68.18 seconds
Total training time: 746.47 seconds
'''

