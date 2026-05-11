import torch
# alpha = 0.1
# beta = 30

def pde_residual(model, x, t, a, kappa): # a=q,kappa=p
    '''
    :param model: a MscalePINN
    :param x: tensor(batch size,1)
    :param t: tensor(batch size,1)
    :param a: 对流系数
    :param kappa: 扩散系数
    '''
    x.requires_grad_(True)
    t.requires_grad_(True)

    xt = torch.cat([x, t], dim=1) # tensor(batch size,2)

    # G and D
    G = -torch.sin(torch.pi * x) # tensor(batch size,1)
    D = t * (1 - x**2)             # tensor(batch size,1)

    # G = torch.sin(torch.pi * x)+0.1*torch.sin(beta*torch.pi * x) 
    # D = t *(x)*(x-2)/20

    NN = model(xt) # tensor(batch size,1)
    u = G + D * NN # 应该是逐元素相乘,u:tensor(batch size,1)

    u_t = torch.autograd.grad(
        u, t, torch.ones_like(u),
        retain_graph=True, create_graph=True
    )[0]

    u_x = torch.autograd.grad(
        u, x, torch.ones_like(u),
        retain_graph=True, create_graph=True
    )[0]

    u_xx = torch.autograd.grad(
        u_x, x, torch.ones_like(u_x),
        retain_graph=True, create_graph=True
    )[0]
    
    # f = torch.exp(-alpha * t) * (-alpha * (torch.sin(torch.pi * x) + 0.1 * torch.sin(beta * torch.pi * x)) +
    #                            0.02 * torch.pi**2 * (torch.sin(torch.pi * x) + 0.1 * beta**2 * torch.sin(beta * torch.pi * x)) +
    #                            0.01 * torch.pi * (torch.cos(torch.pi * x) + 0.1 * beta * torch.cos(beta * torch.pi * x)))

    return u_t + a * u_x - kappa * u_xx 
    # return residual:tensor(batch size,1)
