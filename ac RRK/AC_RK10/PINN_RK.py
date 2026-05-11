import torch
from torch.autograd import grad
from network import Network
from config import *

class AllenCahnDiscretePINN:

    def __init__(self,x_train,u_train,x_boundary,c,b,A):

        self.device=DEVICE

        self.c=torch.tensor(c,dtype=DTYPE,device=DEVICE)
        self.b=torch.tensor(b,dtype=DTYPE,device=DEVICE)
        self.A=torch.tensor(A,dtype=DTYPE,device=DEVICE)

        self.model=Network(
            1,N_NEURONS,Q+1,N_LAYERS
        ).to(DEVICE)
        # u_all = self.model(x)
        # u_all shape:[N, Q+1]

        self.x_train=x_train    # 标签数据
        self.u_train=u_train    # 标签数据
        self.x_boundary=x_boundary   # 边界

        self.iter=1

        self.lbfgs=torch.optim.LBFGS(
            self.model.parameters(),
            lr=1.0,
            max_iter=10000,
            max_eval=15000,
            history_size=100,
            line_search_fn="strong_wolfe"
        )

        self.adam=torch.optim.Adam(
            self.model.parameters(),
            lr=1e-4,  # 🐋lr=1e-3,
            amsgrad=True
        )

        self.scheduler=torch.optim.lr_scheduler.ExponentialLR(
            self.adam,gamma=0.98)

    # ================================
    # 🔥向量化 N operator
    # ================================
    def compute_N_operator(self,u,x):
        '''
        u : [N,1]
        x : [N,1]
        '''
        ones=torch.ones_like(u)
        u_x=grad(u,x,ones,create_graph=True,retain_graph=True)[0]
        u_xx=grad(u_x,x,ones,create_graph=True,retain_graph=True)[0]
        return -1e-4*u_xx+5*u**3-5*u

    # ================================
    # 🔥核心优化：RK向量化
    # ================================
    def compute_rk_errors(self,x,u_target):
        '''
        x: [N,1],require_grad
        u_target: must [N,1],not [N,]!
        '''
        u_all=self.model(x)  # u_all shape:[N, Q+1],下面拆分
        u_inter=u_all[:,:Q]    # [N,Q]
        u_final=u_all[:,Q:Q+1]  # [N,1]

        # -------- 向量化N计算 --------
        N_list=[]  # list,each [N,1],表示中间Q个时间步的N个x值上的u值
        for j in range(Q):
            N_list.append(
                self.compute_N_operator(
                    u_inter[:,j:j+1],x
                )
            )
        N=torch.cat(N_list,dim=1)  # [N,Q]

        # -------- 向量化RK --------
        sum_A=N@self.A.T
        sum_b=N@self.b.unsqueeze(1)

        u_i_n=u_inter+DT*sum_A
        u_q_n=u_final+DT*sum_b
        
        # u_target : [N,1]，表示t_n时刻的再N个x点上的真解u
        err_mid=u_i_n-u_target
        err_final=u_q_n-u_target

        return err_mid,err_final

    # ================================
    # inside loss（无list）
    # ================================
    def inside_loss(self):

        err_mid,err_final=self.compute_rk_errors(
            self.x_train,self.u_train)

        return (err_mid**2).sum()+(err_final**2).sum()

    # ================================
    # 🔥边界loss完全向量化
    # ================================
    def boundary_loss(self):
        u=self.model(self.x_boundary)
        loss_val=((u[0]-u[1])**2).sum()
        grads=[]
        for i in range(Q+1):
            grads.append(
                grad(u[:,i:i+1],
                     self.x_boundary,
                     torch.ones_like(u[:,i:i+1]),
                     create_graph=True,
                     retain_graph=True)[0]
            )
        g=torch.cat(grads,dim=1)
        loss_deriv=((g[0]-g[1])**2).sum()
        return loss_val+loss_deriv

    def compute_loss(self):
        loss_eq=self.inside_loss()
        loss_bc=self.boundary_loss()
        loss=loss_eq+loss_bc
        if self.iter%10==0:         # ⭐每10步迭代打印loss
            print(f"Iter {self.iter:5d} "
                  f"Total {loss.item():.3e}")
        return loss

    def loss_func(self):  # 用于lbfgs
        self.lbfgs.zero_grad()
        loss=self.compute_loss()
        loss.backward()
        self.iter+=1  # 每调用一次算作iter +1
        return loss

    def adam_step(self):
        self.adam.zero_grad()
        loss=self.compute_loss()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            self.model.parameters(),0.5)
        self.adam.step()
        self.scheduler.step()
        self.iter+=1  # 迭代步数+1
        return loss

    def train(self,adam_iter=500):  # 🐋def train(self,adam_iter=300):  
        print("Adam阶段")
        for _ in range(adam_iter):
            self.adam_step()

        print("LBFGS阶段")
        self.lbfgs.step(self.loss_func)

    def predict(self,x):
        x=torch.tensor(
            x,dtype=DTYPE,
            device=DEVICE
        ).unsqueeze(1)

        with torch.no_grad():
            return self.model(x)[:,Q].cpu().numpy()
