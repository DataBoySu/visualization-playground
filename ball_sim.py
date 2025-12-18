#!/usr/bin/env python3
"""GPU-accelerated particle physics simulation."""

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / 'simulation'))

from simulation import physics_torch, particle_utils, gpu_setup
from simulation import visualizer, event_handler, metrics_sampler, config


def _hsv_to_rgb(h, s, v):
    # h in [0,1], s in [0,1], v in [0,1]
    i = int(h * 6.0)
    f = (h * 6.0) - i
    p = v * (1.0 - s)
    q = v * (1.0 - f * s)
    t = v * (1.0 - (1.0 - f) * s)
    i = i % 6
    if i == 0:
        r, g, b = v, t, p
    elif i == 1:
        r, g, b = q, v, p
    elif i == 2:
        r, g, b = p, v, t
    elif i == 3:
        r, g, b = p, q, v
    elif i == 4:
        r, g, b = t, p, v
    else:
        r, g, b = v, p, q
    return int(r * 255), int(g * 255), int(b * 255)


class BallSimulation:
    """Main simulation controller."""
    
    def __init__(self, particle_count=100000):
        self.particle_count = particle_count
        self.gpu_arrays = None
        self.counters = None
        self.metrics_sampler = None
        self.running = False
        self.iterations = 0
        self._terminal_stats_error_printed = False
        
        self.gravity_strength = 500.0
        self.small_ball_speed = 200.0
        self.initial_balls = 20
        self.max_balls_cap = 100000
        self.split_enabled = False
        self.drop_timer = 0.0
        
    def initialize(self):
        import torch

        # Enforce CUDA 12.x compatibility
        cuda_version = getattr(torch.version, 'cuda', None) or torch.version.cuda
        if not torch.cuda.is_available() or not cuda_version:
            raise RuntimeError(
                f"CUDA GPU not available or PyTorch not built with CUDA. "
                "This simulation requires CUDA 12.x and PyTorch/CuPy built for CUDA 12.x."
            )
        try:
            major = int(str(cuda_version).split('.')[0])
        except Exception:
            major = None
        if major != 12:
            raise RuntimeError(
                f"Incompatible CUDA version detected: {cuda_version!s}. "
                "This project requires CUDA major version 12.x. Please install PyTorch/CuPy built for CUDA 12.x."
            )

        self.gpu_arrays, self.counters = gpu_setup.setup_torch_arrays(self.particle_count, torch)
        self.metrics_sampler = metrics_sampler.GPUMetricsSampler()
        self.metrics_sampler.start()
    
    def run(self, duration=None, show_visualization=True):
        import torch
        import pygame
        
        self.running = True
        start_time = time.time()

        if show_visualization:
            # Use the pygame visualizer (keep renderer simple and stable)
            viz = visualizer.ParticleVisualizer(window_size=config.WINDOW_SIZE)

            clock = pygame.time.Clock()
            frame_times = []
            max_frame_history = 10
            
            try:
                while self.running:
                    frame_start = time.time()

                    events = pygame.event.get()
                    # Use the return value from the event handler to decide
                    # whether to continue running. This prevents double-consumption
                    # of events and ensures a single close-click exits the app.
                    if not event_handler.handle_events(viz, events):
                        self.running = False
                        break

                    elapsed = time.time() - start_time
                    if duration and elapsed >= duration:
                        self.running = False
                        break

                    slider_values = viz.get_slider_values()
                    self.gravity_strength = slider_values['gravity']
                    self.small_ball_speed = slider_values['small_ball_speed']
                    self.initial_balls = int(slider_values['initial_balls'])

                    try:
                        max_cap_text = viz.get_max_balls_cap()
                        self.max_balls_cap = int(max_cap_text) if max_cap_text.isdigit() else 100000
                    except:
                        self.max_balls_cap = 100000

                    self.split_enabled = viz.get_split_enabled()

                    spawn_requests = viz.get_spawn_requests()
                    for item in spawn_requests:
                        # spawn requests are tuples (x, y, count)
                        try:
                            sim_x, sim_y, count = item
                        except Exception:
                            # ignore malformed requests
                            continue

                        # If a big ball exists at the click location, pop it instead of spawning
                        popped, delta_active, delta_small = particle_utils.try_pop_big_ball(self.gpu_arrays, 'torch', sim_x, sim_y, small_count=10)
                        if popped:
                            # adjust counters: active_count decreased by 1, small balls increased by delta_small
                            self.counters['active_count'] = max(0, self.counters.get('active_count', 0) + delta_active)
                            self.counters['small_ball_count'] = max(0, self.counters.get('small_ball_count', 0) + delta_small)
                        else:
                            # regular spawn
                            self.spawn_big_balls(sim_x, sim_y, count)

                    params = {
                        'gravity_strength': self.gravity_strength,
                        'small_ball_speed': self.small_ball_speed,
                        'initial_balls': self.initial_balls,
                        'max_balls_cap': self.max_balls_cap,
                        'split_enabled': self.split_enabled,
                        'active_count': self.counters['active_count'],
                        'small_ball_count': self.counters['small_ball_count'],
                        'drop_timer': self.drop_timer,
                    }

                    result = physics_torch.run_particle_physics_torch(
                        self.gpu_arrays, params, torch
                    )

                    self.counters['active_count'] = result['active_count']
                    self.counters['small_ball_count'] = result['small_ball_count']
                    self.drop_timer = result['drop_timer']
                    self.split_enabled = result['split_enabled']

                    # If physics reports a winning color (only one color remains among big balls),
                    # display it on-screen for 5 seconds then exit.
                    winner = result.get('winner_color', None)
                    if winner is not None:
                        try:
                            r, g, b = [int(max(0, min(255, c * 255))) for c in winner]
                        except Exception:
                            r, g, b = (255, 255, 255)

                        # Draw full-screen overlay with the color and text
                        try:
                            viz.screen.fill((r, g, b))
                            text = viz.font.render('Winner color', True, (255 - r, 255 - g, 255 - b))
                            tw, th = text.get_size()
                            viz.screen.blit(text, ((viz.window_size[0] - tw) // 2, (viz.window_size[1] - th) // 2))
                            viz.pygame.display.flip()
                        except Exception:
                            pass

                        # Wait up to 5 seconds while processing events so window stays responsive
                        wait_start = time.time()
                        while time.time() - wait_start < 5.0:
                            for e in viz.pygame.event.get():
                                if e.type == viz.pygame.QUIT:
                                    break
                            time.sleep(0.05)

                        self.running = False
                        break

                    torch.cuda.synchronize()
                    self.iterations += 1

                    render_fps = 0
                    gpu_util = 0

                    positions, masses, colors, glows = self.get_particle_sample(max_samples=2000)

                    if positions is not None:
                        frame_time = time.time() - frame_start
                        frame_times.append(frame_time)
                        if len(frame_times) > max_frame_history:
                            frame_times.pop(0)
                        avg_frame_time = sum(frame_times) / len(frame_times)
                        render_fps = 1.0 / avg_frame_time if avg_frame_time > 0 else 0

                        gpu_util = self.metrics_sampler.get_current_util()

                        influence_boundaries = self.get_influence_boundaries(self.gravity_strength)

                        viz.render_frame(
                            positions=positions,
                            masses=masses,
                            colors=colors,
                            glows=glows,
                            influence_boundaries=influence_boundaries,
                            total_particles=self.particle_count,
                            active_particles=self.counters['active_count'],
                            fps=render_fps,
                            gpu_util=gpu_util,
                            elapsed_time=elapsed
                        )

                    # Print colorful terminal stats periodically to aid monitoring
                    if self.iterations % 10 == 0:
                        try:
                            iter_per_sec = (self.iterations / elapsed) if elapsed > 0 else 0
                            active_count = self.counters.get('active_count', 0)
                            small = self.counters.get('small_ball_count', 0)
                            big = max(0, active_count - small)

                            # Try to find color of the big ball with the most health
                            r = g = b = None
                            try:
                                health = self.gpu_arrays.get('health', None)
                                ball_color = self.gpu_arrays.get('ball_color', None)
                                mass = self.gpu_arrays.get('mass', None)
                                active_mask = self.gpu_arrays.get('active', None)
                                if health is not None and ball_color is not None and mass is not None and active_mask is not None:
                                    big_mask = (mass >= 100.0) & active_mask
                                    if int(torch.sum(big_mask)) > 0:
                                        big_indices = torch.where(big_mask)[0]
                                        h_vals = health[big_indices].cpu().numpy()
                                        if h_vals.size > 0:
                                            argmax_local = int(h_vals.argmax())
                                            idx = int(big_indices[argmax_local])
                                            col = ball_color[idx].cpu().numpy()
                                            r, g, b = [int(max(0, min(255, c * 255))) for c in col]
                            except Exception as e:
                                # compute fallback color and record debug once
                                r = g = b = None
                                if not self._terminal_stats_error_printed:
                                    print(f"[DEBUG] terminal color selection failed: {e}", file=sys.stderr)
                                    self._terminal_stats_error_printed = True

                            if r is None:
                                hue = (time.time() * 0.18) % 1.0
                                r, g, b = _hsv_to_rgb(hue, 0.72, 0.95)

                            # ANSI 24-bit color sequence
                            color_seq = f"\x1b[38;2;{r};{g};{b}m"
                            reset_seq = "\x1b[0m"

                            spinner_frames = ['◐', '◓', '◑', '◒']
                            spinner = spinner_frames[self.iterations % len(spinner_frames)]

                            line = (
                                f"Iter: {self.iterations:>6,} | {render_fps:>5.1f} FPS | "
                                f"GPU: {gpu_util:>3.0f}% | Active: {active_count:>6,} | Small: {small:>6,} | Big: {big:>6,} "
                                f"{spinner}"
                            )

                            # Overwrite the same terminal line with color
                            print(f"\r{color_seq}{line}{reset_seq}", end='', flush=True)
                        except Exception as e:
                            # Ensure printing errors don't stop the simulation; print debug once
                            if not self._terminal_stats_error_printed:
                                print(f"[DEBUG] terminal printing failed: {e}", file=sys.stderr)
                                self._terminal_stats_error_printed = True

                    clock.tick()
            except KeyboardInterrupt:
                # User pressed Ctrl+C — stop cleanly
                self.running = False
                print("\nInterrupted by user (KeyboardInterrupt)")
            except Exception as e:
                # Ensure visualizer and metrics are cleaned up on unexpected errors
                try:
                    viz.close()
                except Exception:
                    pass
                try:
                    self.metrics_sampler.stop()
                except Exception:
                    pass
                print(f"\nError during visualization loop: {e}")
                sys.exit(1)
            finally:
                try:
                    viz.close()
                except Exception:
                    pass
        else:
            import torch

            try:
                while self.running:
                    elapsed = time.time() - start_time
                    if duration and elapsed >= duration:
                        break

                    params = {
                        'gravity_strength': self.gravity_strength,
                        'small_ball_speed': self.small_ball_speed,
                        'initial_balls': self.initial_balls,
                        'max_balls_cap': self.max_balls_cap,
                        'split_enabled': self.split_enabled,
                        'active_count': self.counters['active_count'],
                        'small_ball_count': self.counters['small_ball_count'],
                        'drop_timer': self.drop_timer,
                    }

                    result = physics_torch.run_particle_physics_torch(
                        self.gpu_arrays, params, torch
                    )

                    self.counters['active_count'] = result['active_count']
                    self.counters['small_ball_count'] = result['small_ball_count']
                    self.drop_timer = result['drop_timer']

                    torch.cuda.synchronize()
                    self.iterations += 1

                    if self.iterations % 50 == 0:
                        gpu_util = self.metrics_sampler.get_current_util()
                        iter_per_sec = self.iterations / elapsed
                        active = self.counters['active_count']
                        small = self.counters['small_ball_count']
                        big = active - small
                        print(f"\rIter: {self.iterations:>6,} | {iter_per_sec:>6.1f} it/s | GPU: {gpu_util:>3.0f}% | Active: {active:>6,} | Small: {small:>6,} | Big: {big:>3}  ", end='', flush=True)
            except KeyboardInterrupt:
                self.running = False
                print("\nInterrupted by user (KeyboardInterrupt)")
            except Exception as e:
                try:
                    self.metrics_sampler.stop()
                except Exception:
                    pass
                print(f"\nError during headless loop: {e}")
                sys.exit(1)
        
        elapsed = time.time() - start_time
        self.metrics_sampler.stop()
        print(f"\n\nCompleted: {self.iterations:,} iterations in {elapsed:.1f}s ({self.iterations/elapsed:.1f} it/s)")
    
    def spawn_big_balls(self, x, y, count=1):
        self.counters['active_count'] = particle_utils.spawn_big_balls(
            self.gpu_arrays,
            'torch',
            x, y, count,
            self.counters['active_count']
        )
    
    def get_influence_boundaries(self, gravity_strength):
        return particle_utils.get_influence_boundaries(
            self.gpu_arrays,
            'torch',
            gravity_strength
        )
    
    def get_particle_sample(self, max_samples=2000):
        return particle_utils.get_particle_sample(
            self.gpu_arrays,
            'torch',
            max_samples
        )


def main():
    parser = argparse.ArgumentParser(
        description='GPU-accelerated particle physics simulation',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument('--particles', type=int, default=100000,
                        help='particle count (default: 100000)')
    parser.add_argument('--duration', type=int, default=None,
                        help='duration in seconds (default: infinite)')
    parser.add_argument('--visualize', action='store_true',
                        help='enable visualization')
    
    args = parser.parse_args()
    
    try:
        sim = BallSimulation(particle_count=args.particles)
        sim.initialize()
        sim.run(duration=args.duration, show_visualization=args.visualize)
    except KeyboardInterrupt:
        print("\n\nInterrupted")
        sys.exit(0)
    except Exception as e:
        print(f"\nError: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()
