import os
import time
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR

from model import MscalePINN
from pde import pde_residual
from utils import sample_collocation

# ----------------------------
# 配置/config
# ----------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

a = 1   # q
kappa = 0.01/torch.pi  # p

EPOCHS = 5000
N_COLLOC = 4000
BATCH_SIZE = 200  # 8000/256=31.25,each epoch have 32 batchs

SAVE_EVERY = 500   
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "checkpoints")
os.makedirs(MODEL_DIR, exist_ok=True)

# ----------------------------
# 模型
# ----------------------------
model = MscalePINN().to(DEVICE)

optimizer = Adam(model.parameters(), lr=1e-2)
scheduler = StepLR(optimizer, step_size=100, gamma=0.975)
# 每100epoch学习率的衰减率为2.5%

# ----------------------------
# 训练
# ----------------------------
start_time = time.perf_counter()
last_print_time = start_time

x, t = sample_collocation(N_COLLOC, DEVICE)  # 优化同一批点

for epoch in range(1, EPOCHS + 1):  # 1 ~ 5000
    # x, t = sample_collocation(N_COLLOC, DEVICE) # 每一个周期都重新采样8000个点，相当于8000个点只迭代一次/一个epoch

    perm = torch.randperm(N_COLLOC) 
    x, t = x[perm], t[perm] 

    epoch_loss = 0.0

    for i in range(0, N_COLLOC, BATCH_SIZE):  # 32
        # 这里保证每个点在当前 epoch 只用一次
        xb = x[i:i+BATCH_SIZE]
        tb = t[i:i+BATCH_SIZE]

        r = pde_residual(model, xb, tb, a, kappa)
        loss = torch.mean(r**2)
        
        # 一个batch：adam一步更新
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()  # 累计 batch loss（只是监控用），每个batch的mse之和，32×每个batch的mse

    scheduler.step()  # 每个周期更新学习率（因为是每100个 周期 更新）

    # ==========================
    # ⏱ 时间 & 日志
    # ==========================
    if epoch % 1 == 0:
        now = time.perf_counter()
        last_100_time = now - last_print_time
        total_time = now - start_time
        last_print_time = now

        print(
            f"Epoch {epoch:5d} | "
            f"Loss = {epoch_loss:.3e} | "
            f"Δt(100ep) = {last_100_time:7.2f}s | "
            f"Total = {total_time/60:6.2f} min"
        )


    # ----------------------------
    # 保存模型
    # ----------------------------
    if epoch % SAVE_EVERY == 0 or epoch == EPOCHS:
        ckpt_path = os.path.join(MODEL_DIR, f"pinn_epoch_{epoch}.pt")
        torch.save(
            {
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "scheduler_state": scheduler.state_dict(),
            },
            ckpt_path,
        )
        print(f"✓ Model saved at epoch {epoch}")

end_time = time.perf_counter()
total_time = end_time - start_time
print(f"\nTraining finished. Total time = {total_time/60:.2f} minutes")




# self ❓：
# ❓每个epoch都resampling？-> gpt say 常见
# ❓既然都重采样了，那 randperm 还有意义吗？-> gpt say yes
