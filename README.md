# GPU Particle Physics Simulation

py-game based GPU-accelerated particle physics with interactive visualization.

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

## License

MIT
