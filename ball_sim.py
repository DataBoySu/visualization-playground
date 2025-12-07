#!/usr/bin/env python3
"""GPU-accelerated particle physics simulation."""

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / 'simulation'))

from simulation import physics_torch, particle_utils, gpu_setup
from simulation import visualizer, event_handler, metrics_sampler


class BallSimulation:
    """Main simulation controller."""
    
    def __init__(self, particle_count=100000):
        self.particle_count = particle_count
        self.gpu_arrays = None
        self.counters = None
        self.metrics_sampler = None
        self.running = False
        self.iterations = 0
        
        self.gravity_strength = 500.0
        self.small_ball_speed = 200.0
        self.initial_balls = 20
        self.max_balls_cap = 100000
        self.split_enabled = False
        self.drop_timer = 0.0
        
    def initialize(self):
        """Initialize GPU arrays and simulation."""
        import torch
        
        self.gpu_arrays, self.counters = gpu_setup.setup_torch_arrays(self.particle_count, torch)
        self.metrics_sampler = metrics_sampler.GPUMetricsSampler()
        self.metrics_sampler.start()
    
    def run(self, duration=None, show_visualization=True):
        """Run the simulation."""
        import torch
        import pygame
        
        self.running = True
        start_time = time.time()
        
        if show_visualization:
            viz = visualizer.ParticleVisualizer(window_size=(1000, 800))
            clock = pygame.time.Clock()
            frame_times = []
            max_frame_history = 10
            
            while self.running:
                frame_start = time.time()
                
                # Handle events
                events = pygame.event.get()
                event_handler.handle_events(viz, events)
                
                # Check for quit
                for event in events:
                    if event.type == pygame.QUIT:
                        self.running = False
                
                # Check duration
                elapsed = time.time() - start_time
                if duration and elapsed >= duration:
                    self.running = False
                    break
                
                # Update physics parameters from UI
                slider_values = viz.get_slider_values()
                self.gravity_strength = slider_values['gravity']
                self.small_ball_speed = slider_values['small_ball_speed']
                self.initial_balls = int(slider_values['initial_balls'])
                
                # Get max_balls_cap from text box
                try:
                    max_cap_text = viz.get_max_balls_cap()
                    self.max_balls_cap = int(max_cap_text) if max_cap_text.isdigit() else 100000
                except:
                    self.max_balls_cap = 100000
                
                self.split_enabled = viz.get_split_enabled()
                
                # Process spawn requests
                spawn_requests = viz.get_spawn_requests()
                for sim_x, sim_y, count in spawn_requests:
                    self.spawn_big_balls(sim_x, sim_y, count)
                
                # Run physics
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
                
                # Update counters
                self.counters['active_count'] = result['active_count']
                self.counters['small_ball_count'] = result['small_ball_count']
                self.drop_timer = result['drop_timer']
                self.split_enabled = result['split_enabled']
                
                torch.cuda.synchronize()
                self.iterations += 1
                
                # Get particle data for rendering
                positions, masses, colors, glows = self.get_particle_sample(max_samples=2000)
                
                if positions is not None:
                    # Calculate FPS
                    frame_time = time.time() - frame_start
                    frame_times.append(frame_time)
                    if len(frame_times) > max_frame_history:
                        frame_times.pop(0)
                    avg_frame_time = sum(frame_times) / len(frame_times)
                    render_fps = 1.0 / avg_frame_time if avg_frame_time > 0 else 0
                    
                    # Get GPU utilization
                    gpu_util = self.metrics_sampler.get_current_util()
                    
                    # Get influence boundaries for big balls
                    influence_boundaries = self.get_influence_boundaries(self.gravity_strength)
                    
                    # Render
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
                
                clock.tick()
            
            viz.close()
        else:
            import torch
            
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
        
        elapsed = time.time() - start_time
        self.metrics_sampler.stop()
        print(f"\n\nCompleted: {self.iterations:,} iterations in {elapsed:.1f}s ({self.iterations/elapsed:.1f} it/s)")
    
    def spawn_big_balls(self, x, y, count=1):
        """Spawn big balls at specified position."""
        self.counters['active_count'] = particle_utils.spawn_big_balls(
            self.gpu_arrays,
            'torch',
            x, y, count,
            self.counters['active_count']
        )
    
    def get_influence_boundaries(self, gravity_strength):
        """Get influence boundaries for big balls."""
        return particle_utils.get_influence_boundaries(
            self.gpu_arrays,
            'torch',
            gravity_strength
        )
    
    def get_particle_sample(self, max_samples=2000):
        """Get particle data for rendering."""
        return particle_utils.get_particle_sample(
            self.gpu_arrays,
            'torch',
            max_samples
        )


def main():
    """Main entry point."""
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
