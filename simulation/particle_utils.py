"""Utility functions for particle sampling and visualization."""

import numpy as np


def get_particle_sample(gpu_arrays, method, max_samples=2000):
    try:
        x = gpu_arrays.get('x')
        y = gpu_arrays.get('y')
        mass = gpu_arrays.get('mass')
        active = gpu_arrays.get('active')
        color_state = gpu_arrays.get('color_state')
        glow_intensity = gpu_arrays.get('glow_intensity')
        ball_color = gpu_arrays.get('ball_color')
        
        if x is None or y is None or mass is None or active is None:
            return None, None, None, None
        
        if method == 'cupy':
            active_mask = active.get()
            x_all = x.get()
            y_all = y.get()
            mass_all = mass.get()
            color_all = color_state.get() if color_state is not None else np.zeros(len(active_mask))
            glow_all = glow_intensity.get() if glow_intensity is not None else np.zeros(len(active_mask))
            ball_color_all = ball_color.get() if ball_color is not None else np.zeros((len(active_mask), 3))
            
            x_active = x_all[active_mask]
            y_active = y_all[active_mask]
            mass_active = mass_all[active_mask]
            color_active = color_all[active_mask]
            glow_active = glow_all[active_mask]
            ball_color_active = ball_color_all[active_mask]
            
        elif method == 'torch':
            active_mask = active.cpu().numpy()
            x_all = x.cpu().numpy()
            y_all = y.cpu().numpy()
            mass_all = mass.cpu().numpy()
            color_all = color_state.cpu().numpy() if color_state is not None else np.zeros(len(active_mask))
            glow_all = glow_intensity.cpu().numpy() if glow_intensity is not None else np.zeros(len(active_mask))
            ball_color_all = ball_color.cpu().numpy() if ball_color is not None else np.zeros((len(active_mask), 3))
            
            x_active = x_all[active_mask]
            y_active = y_all[active_mask]
            mass_active = mass_all[active_mask]
            color_active = color_all[active_mask]
            glow_active = glow_all[active_mask]
            ball_color_active = ball_color_all[active_mask]
        else:
            return None, None, None, None
        
        # Sample if too many
        n_active = len(x_active)
        if n_active > max_samples:
            step = n_active // max_samples
            x_active = x_active[::step]
            y_active = y_active[::step]
            mass_active = mass_active[::step]
            color_active = color_active[::step]
            glow_active = glow_active[::step]
            ball_color_active = ball_color_active[::step]
        
        positions = np.column_stack([x_active, y_active])
        return positions, mass_active, ball_color_active, glow_active
        
    except Exception:
        return None, None, None, None


def get_influence_boundaries(gpu_arrays, method, gravity_strength=500.0):
    try:
        x = gpu_arrays.get('x')
        y = gpu_arrays.get('y')
        mass = gpu_arrays.get('mass')
        active = gpu_arrays.get('active')
        
        if x is None or y is None or mass is None or active is None:
            return []
        
        if method == 'cupy':
            x_all = x.get()
            y_all = y.get()
            mass_all = mass.get()
            active_mask = active.get()
        elif method == 'torch':
            x_all = x.cpu().numpy()
            y_all = y.cpu().numpy()
            mass_all = mass.cpu().numpy()
            active_mask = active.cpu().numpy()
        else:
            return []
        
        large_mask = (mass_all >= 1000.0) & active_mask
        
        boundaries = []
        for i in range(len(x_all)):
            if large_mask[i]:
                grav_radius = max(50.0, np.sqrt(gravity_strength * mass_all[i] / 1.0) / 5.0)
                boundaries.append((float(x_all[i]), float(y_all[i]), float(grav_radius)))
        
        return boundaries
        
    except Exception:
        return []


def spawn_big_balls(gpu_arrays, method, x, y, count, current_active_count):
    import random
    
    spawned = 0
    if method == 'cupy':
        import cupy as cp
        active = gpu_arrays['active']
        
        for i in range(count):
            inactive_indices = cp.where(~active)[0]
            if len(inactive_indices) == 0:
                print("[Spawn] No inactive slots available!")
                break
            
            idx = int(inactive_indices[0])
            offset_x = random.uniform(-20, 20) if count > 1 else 0
            offset_y = random.uniform(-20, 20) if count > 1 else 0
            color = cp.array([random.uniform(0.3, 1.0), random.uniform(0.3, 1.0), random.uniform(0.3, 1.0)], dtype=cp.float32)
            
            gpu_arrays['x'][idx] = x + offset_x
            gpu_arrays['y'][idx] = y + offset_y
            gpu_arrays['vx'][idx] = 0.0
            gpu_arrays['vy'][idx] = 0.0
            gpu_arrays['mass'][idx] = 1000.0
            gpu_arrays['radius'][idx] = 36.0
            gpu_arrays['active'][idx] = True
            gpu_arrays['ball_color'][idx] = color
            current_active_count += 1
            spawned += 1
    
    elif method == 'torch':
        import torch
        
        for i in range(count):
            inactive_indices = torch.where(~gpu_arrays['active'])[0]
            if len(inactive_indices) == 0:
                break
            
            idx = int(inactive_indices[-1])
            offset_x = random.uniform(-20, 20) if count > 1 else 0
            offset_y = random.uniform(-20, 20) if count > 1 else 0
            final_x = x + offset_x
            final_y = y + offset_y
            color = torch.tensor([random.uniform(0.3, 1.0), random.uniform(0.3, 1.0), random.uniform(0.3, 1.0)], 
                                device=gpu_arrays['x'].device, dtype=torch.float32)
            
            gpu_arrays['x'][idx] = final_x
            gpu_arrays['y'][idx] = final_y
            gpu_arrays['vx'][idx] = 0.0
            gpu_arrays['vy'][idx] = 0.0
            gpu_arrays['mass'][idx] = 1000.0
            gpu_arrays['radius'][idx] = 36.0
            gpu_arrays['ball_color'][idx] = color
            gpu_arrays['active'][idx] = True
            
            current_active_count += 1
            spawned += 1
    
    return current_active_count


def update_big_ball_count(gpu_arrays, method, target_count, current_active_count):
    """Dynamically adjust the number of big balls."""
    import random
    
    target_count = max(1, min(100, target_count))
    
    if method == 'cupy':
        import cupy as cp
        mass = gpu_arrays['mass']
        active = gpu_arrays['active']
        big_ball_mask = (mass >= 100.0) & active
        current_big_balls = int(cp.sum(big_ball_mask))
        
        if target_count > current_big_balls:
            to_spawn = target_count - current_big_balls
            for i in range(to_spawn):
                inactive_indices = cp.where(~active)[0]
                if len(inactive_indices) == 0:
                    break
                
                idx = int(inactive_indices[0])
                x = random.uniform(100, 900)
                y = random.uniform(100, 700)
                color = cp.array([random.uniform(0.3, 1.0), random.uniform(0.3, 1.0), random.uniform(0.3, 1.0)], dtype=cp.float32)
                
                gpu_arrays['x'][idx] = x
                gpu_arrays['y'][idx] = y
                gpu_arrays['vx'][idx] = 0.0
                gpu_arrays['vy'][idx] = 0.0
                gpu_arrays['mass'][idx] = 1000.0
                gpu_arrays['radius'][idx] = 36.0
                gpu_arrays['active'][idx] = True
                gpu_arrays['ball_color'][idx] = color
                current_active_count += 1
    
    elif method == 'torch':
        import torch
        mass = gpu_arrays['mass']
        active = gpu_arrays['active']
        big_ball_mask = (mass >= 100.0) & active
        current_big_balls = int(torch.sum(big_ball_mask))
        
        if target_count > current_big_balls:
            to_spawn = target_count - current_big_balls
            for i in range(to_spawn):
                inactive_indices = torch.where(~active)[0]
                if len(inactive_indices) == 0:
                    break
                
                idx = int(inactive_indices[0])
                x = random.uniform(100, 900)
                y = random.uniform(100, 700)
                color = torch.tensor([random.uniform(0.3, 1.0), random.uniform(0.3, 1.0), random.uniform(0.3, 1.0)], 
                                    device=gpu_arrays['x'].device, dtype=torch.float32)
                
                gpu_arrays['x'][idx] = x
                gpu_arrays['y'][idx] = y
                gpu_arrays['vx'][idx] = 0.0
                gpu_arrays['vy'][idx] = 0.0
                gpu_arrays['mass'][idx] = 1000.0
                gpu_arrays['radius'][idx] = 36.0
                gpu_arrays['active'][idx] = True
                gpu_arrays['ball_color'][idx] = color
                current_active_count += 1
    
    return current_active_count

def try_pop_big_ball(gpu_arrays, method, x, y, small_count=10):
    """Try to pop a big ball at (x,y). If a big ball is under the click, deactivate it and spawn small balls of the same color.
    Returns (popped: bool, active_count_delta: int, small_count_delta: int)
    """
    if method == 'torch':
        import torch
        active = gpu_arrays['active']
        mass = gpu_arrays['mass']
        xpos = gpu_arrays['x']
        ypos = gpu_arrays['y']
        radius = gpu_arrays['radius']
        ball_color = gpu_arrays['ball_color']

        big_mask = active & (mass >= 100.0)
        if torch.sum(big_mask) == 0:
            return False, 0, 0

        big_indices = torch.where(big_mask)[0]
        bx = xpos[big_indices]
        by = ypos[big_indices]
        br = radius[big_indices]

        # distances
        dx = bx - x
        dy = by - y
        dist = torch.sqrt(dx * dx + dy * dy)

        # find nearest big ball
        min_idx = torch.argmin(dist)
        if dist[min_idx] <= (br[min_idx] + 8.0):
            target_global = int(big_indices[int(min_idx)])
            color = ball_color[target_global]

            # deactivate big ball
            active[target_global] = False
            mass[target_global] = 0.0
            radius[target_global] = 0.0
            xpos[target_global] = 0.0
            ypos[target_global] = 0.0
            gpu_arrays['vx'][target_global] = 0.0
            gpu_arrays['vy'][target_global] = 0.0

            # spawn small balls from inactive pool
            inactive = torch.where(~active)[0]
            spawn = min(len(inactive), small_count)
            for i in range(spawn):
                child = int(inactive[i])
                xpos[child] = x + float((torch.randn(1) * 8.0).item())
                ypos[child] = y + float((torch.randn(1) * 8.0).item())
                angle = float((torch.rand(1) * 2.0 * 3.14159).item())
                gpu_arrays['vx'][child] = float(torch.cos(torch.tensor(angle)).item()) * 200.0
                gpu_arrays['vy'][child] = float(torch.sin(torch.tensor(angle)).item()) * 200.0
                gpu_arrays['mass'][child] = 1.0
                gpu_arrays['radius'][child] = 8.0
                gpu_arrays['active'][child] = True
                gpu_arrays['split_cooldown'][child] = 5.0
                gpu_arrays['ball_color'][child] = color

            # active count delta = spawned smalls - 1 (big removed)
            return True, (spawn - 1), spawn

    # Cupy and other methods not implemented
    return False, 0, 0

