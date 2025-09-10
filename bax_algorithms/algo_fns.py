import torch

def global_opt(f_x, x_grid, minimize=True):
    if minimize:
        y_opt, opt_idx = torch.min(f_x, dim=-2)
    else:
        y_opt, opt_idx = torch.max(f_x, dim=-2)
     
    opt_idx = opt_idx.squeeze(dim=-1)
    x_opt = x_grid[opt_idx]
    
    return x_opt, y_opt

def single_level_band(f_x, x_grid, min_val = None, max_val = None):
    idxs = torch.where((f_x >= min_val) & (f_x < max_val))

    # To do: maybe add some shape checking here
    y_opt = f_x[idxs].unsqueeze(-1)
    # 1:-1 avoids sampling idx + y property idx
    x_opt = x_grid[idxs[1:-1]]
    
    return x_opt, y_opt

def multi_level_band(f_x, x_grid, bounds_list = None):
    assert f_x.shape[-1] == len(bounds_list), f"len(bounds_list) ({len(bounds_list)}) must match number of property dimensions ({f_x.shape[-1]})"
    
    # Start with a mask of all True values
    condition = torch.ones(f_x.shape[:-1], dtype=torch.bool, device=f_x.device)

    for i, (lower, upper) in enumerate(bounds_list):
        condition &= (f_x[..., i] >= lower) & (f_x[..., i] < upper)
            
    idxs = torch.where(condition)
    y_opt = f_x[idxs]
    # :-1 avoids y property idx
    x_opt = x_grid[idxs[:-1]]
    
    return x_opt, y_opt



# Implementation adapted from: https://stackoverflow.com/questions/32791911/fast-calculation-of-pareto-front-in-python (Peter)
def obtain_discrete_pareto_optima(f_x, x_grid):
    is_efficient = torch.arange(f_x.shape[0])
    next_point_index = 0  # Next index in the is_efficient array to search for
    while next_point_index < len(f_x):
        nondominated_point_mask = torch.any(f_x >= f_x[next_point_index], axis=1)
        is_efficient = is_efficient[nondominated_point_mask]  # Remove dominated points
        f_x = f_x[nondominated_point_mask]
        next_point_index = torch.sum(nondominated_point_mask[:next_point_index]) + 1

    y_opt = f_x[is_efficient]
    x_opt = x_grid[is_efficient]
    
    return x_opt, y_opt