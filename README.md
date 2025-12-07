# GPU Particle Physics Simulation

Real-time GPU-accelerated particle physics with interactive visualization.

## Installation

```bash
pip install -r requirements.txt
```

Requires: Python 3.10+, NVIDIA GPU with CUDA

## Usage

```bash
# Headless mode (benchmark)
python ball_sim.py --particles 100000 --duration 60

# Interactive visualization
python ball_sim.py --visualize --particles 50000

# Custom particle count
python ball_sim.py --visualize --particles 200000 --duration 120
```

## Controls

- **Left Click**: Spawn big balls
- **Sliders**: Gravity, speed, particle count
- **Text Box**: Max particle cap
- **ESC**: Exit

## Features

- N-body gravitational physics
- Elastic collision detection
- Real-time parameter adjustment
- Particle splitting mechanics
- Click-to-spawn interaction
- GPU metrics monitoring

## Performance

NVIDIA RTX 3060:

- 100K particles: 80-100 FPS (visualization)
- 100K particles: 150-200 it/s (headless)
- GPU utilization: 15-25%

## Future Improvements

### Physics

- [ ] Soft-body dynamics
- [ ] Fluid simulation integration
- [ ] Custom force fields
- [ ] Magnetic/electric field simulation
- [ ] Heat transfer and thermodynamics

### Visualization

- [ ] WebGL/Three.js web interface
- [ ] VR support
- [ ] Trail effects for particles
- [ ] Heatmap overlays
- [ ] Record/replay functionality
- [ ] Multiple camera angles

### Analysis

- [ ] Energy conservation metrics
- [ ] Statistical analysis tools
- [ ] Export data to CSV/HDF5
- [ ] Real-time plotting
- [ ] Benchmark comparison suite

### Integration

- [ ] REST API for remote control
- [ ] Python API for scripting
- [ ] Plugin system
- [ ] External renderer support (Blender, Unity)

## License

MIT
