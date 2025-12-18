"""GPU array initialization for particle simulation."""


def setup_cupy_arrays(n, cp):
    gpu_arrays = {}
    
    gpu_arrays['x'] = cp.zeros(n, dtype=cp.float32)
    gpu_arrays['y'] = cp.zeros(n, dtype=cp.float32)
    gpu_arrays['vx'] = cp.zeros(n, dtype=cp.float32)
    gpu_arrays['vy'] = cp.zeros(n, dtype=cp.float32)
    gpu_arrays['mass'] = cp.zeros(n, dtype=cp.float32)
    gpu_arrays['radius'] = cp.zeros(n, dtype=cp.float32)
    gpu_arrays['active'] = cp.zeros(n, dtype=cp.bool_)
    gpu_arrays['bounce_cooldown'] = cp.zeros(n, dtype=cp.float32)
    gpu_arrays['color_state'] = cp.zeros(n, dtype=cp.float32)  # 0=default, >0=hit (fades)
    gpu_arrays['glow_intensity'] = cp.zeros(n, dtype=cp.float32)  # GPU-computed glow
    gpu_arrays['should_split'] = cp.zeros(n, dtype=cp.bool_)  # Mark for splitting
    gpu_arrays['split_cooldown'] = cp.zeros(n, dtype=cp.float32)  # 5-second cooldown after split
    gpu_arrays['ball_color'] = cp.zeros((n, 3), dtype=cp.float32)
    gpu_arrays['health'] = cp.zeros(n, dtype=cp.float32)
    gpu_arrays['consec_non_own'] = cp.zeros(n, dtype=cp.int32)
    
    big_ball_colors = cp.array([
        [1.0, 0.2, 0.2],
        [0.2, 1.0, 0.2],
        [0.2, 0.4, 1.0],
        [1.0, 0.8, 0.2],
    ], dtype=cp.float32)
    
    big_positions = [(450, 350), (550, 350), (450, 450), (550, 450)]
    
    for i in range(4):
        gpu_arrays['x'][i] = big_positions[i][0]
        gpu_arrays['y'][i] = big_positions[i][1]
        gpu_arrays['vx'][i] = 0.0
        gpu_arrays['vy'][i] = 0.0
        gpu_arrays['mass'][i] = 1000.0
        gpu_arrays['radius'][i] = 36.0
        gpu_arrays['active'][i] = True
        gpu_arrays['ball_color'][i] = big_ball_colors[i]
        gpu_arrays['health'][i] = 50.0
    
    # Initial counters
    counters = {
        'active_count': 4,
        'small_ball_count': 0,
        'drop_timer': 0.0,
        'gravity_strength': 500.0,
        'small_ball_speed': 300.0,
        'initial_balls': 1,
        'max_balls_cap': 100000,
        'split_enabled': False
    }
    
    return gpu_arrays, counters


def setup_torch_arrays(n, torch):
    device = torch.device('cuda')
    gpu_arrays = {}
    
    gpu_arrays['x'] = torch.zeros(n, device=device, dtype=torch.float32)
    gpu_arrays['y'] = torch.zeros(n, device=device, dtype=torch.float32)
    gpu_arrays['vx'] = torch.zeros(n, device=device, dtype=torch.float32)
    gpu_arrays['vy'] = torch.zeros(n, device=device, dtype=torch.float32)
    gpu_arrays['mass'] = torch.zeros(n, device=device, dtype=torch.float32)
    gpu_arrays['radius'] = torch.zeros(n, device=device, dtype=torch.float32)
    gpu_arrays['active'] = torch.zeros(n, device=device, dtype=torch.bool)
    gpu_arrays['bounce_cooldown'] = torch.zeros(n, device=device, dtype=torch.float32)
    gpu_arrays['color_state'] = torch.zeros(n, device=device, dtype=torch.float32)
    gpu_arrays['glow_intensity'] = torch.zeros(n, device=device, dtype=torch.float32)
    gpu_arrays['should_split'] = torch.zeros(n, device=device, dtype=torch.bool)
    gpu_arrays['split_cooldown'] = torch.zeros(n, device=device, dtype=torch.float32)
    gpu_arrays['ball_color'] = torch.zeros((n, 3), device=device, dtype=torch.float32)
    # Health and consecutive non-own-color hit counters
    gpu_arrays['health'] = torch.zeros(n, device=device, dtype=torch.float32)
    gpu_arrays['consec_non_own'] = torch.zeros(n, device=device, dtype=torch.int32)
    
    big_ball_colors = torch.tensor([
        [1.0, 0.2, 0.2],
        [0.2, 1.0, 0.2],
        [0.2, 0.4, 1.0],
        [1.0, 0.8, 0.2],
    ], device=device, dtype=torch.float32)
    
    big_positions = [(450, 350), (550, 350), (450, 450), (550, 450)]
    
    for i in range(4):
        gpu_arrays['x'][i] = big_positions[i][0]
        gpu_arrays['y'][i] = big_positions[i][1]
        gpu_arrays['vx'][i] = 0.0
        gpu_arrays['vy'][i] = 0.0
        gpu_arrays['mass'][i] = 1000.0
        gpu_arrays['radius'][i] = 36.0
        gpu_arrays['active'][i] = True
        gpu_arrays['ball_color'][i] = big_ball_colors[i]
        gpu_arrays['health'][i] = 50.0  # initial health for big balls
    
    counters = {
        'active_count': 4,
        'small_ball_count': 0,
        'drop_timer': 0.0,
        'gravity_strength': 500.0,
        'small_ball_speed': 300.0,
        'initial_balls': 1,
        'max_balls_cap': 100000,
        'split_enabled': False
    }
    
    return gpu_arrays, counters
