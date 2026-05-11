import torch

def sample_collocation(n_points, device):
    '''
    :param n_points: 8000
    :param device: cuda
    '''
    x = -1 + 2 * torch.rand(n_points, 1, device=device)  # tensor(n_points=8000,1),each in(-1,1)
    t = 2 * torch.rand(n_points, 1, device=device)    # tensor(n_points=8000,1),each in(0,2)

    # x = 2 * torch.rand(n_points, 1, device=device)  # tensor(n_points=8000,1),each in(0,2)
    # t = torch.rand(n_points, 1, device=device)    # tensor(n_points=8000,1),each in(0,1)
    return x, t
