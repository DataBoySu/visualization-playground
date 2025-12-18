<p align="center">
  <img alt="Bouncing Balls" src="simulation/assets/textanim.gif"/>
</p>

![Status: WIP](https://img.shields.io/badge/status-WIP-orange.svg) ![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg) ![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)

This is my personal playground where I implement various bonker things to get better at visualizations and physics simulation. Expect experiments, small games, and visualization-first simulations.

<p align="center">
  <img alt="Bouncing Balls" src="simulation/assets/bouncingballs.gif"/>
</p>

## Controls

- **Left Click**: Spawn big balls
- **Sliders**: Gravity, speed, particle count
- **Text Box**: Max particle cap
- **ESC**: Exit

---

## Physics & Math

This project uses a mix of simple and pragmatic physics models to balance performance and visual richness:

- **Gravity / attraction:** pairwise influence approximating inverse-square behavior
  - Force from mass j on i:  `F_ij = G * m_j / (r_ij^2 + ε)`
  - Implementation notes: forces are accumulated using vectorized GPU ops (PyTorch tensors on CUDA). A softening term `ε` avoids singularities at short range.

- **Collision response (impulses):**
  - Compute normal `n = (x_j - x_i) / |x_j - x_i|`
  - Relative velocity along `n` determines impulse magnitude; big masses use a near-elastic restitution.

- **Time integration:** semi-implicit Euler (velocity updated by acceleration, positions incremented by velocity)

- **Gameplay mechanics:** big balls have health, same-color small balls heal & get absorbed; non-own-color hits damage and can trigger explosions into small balls.

---

## Where things are used

- **PyTorch (CUDA)** — physics & vector math on the GPU (`simulation/physics_torch.py`, `simulation/gpu_setup.py`).
- **Pygame** — visualization, UI, event handling (`simulation/visualizer.py`, `simulation/ui_components.py`).
- **CuPy** (optional) — alternate GPU compute backend.
- **numpy** — utility conversions and CPU fallbacks.

---

## System Requirements

| Component | Requirement |
|---|---|
| OS | Windows / Linux / macOS (dev-tested on Windows) |
| Python | **3.10+** |
| GPU | NVIDIA GPU with CUDA (recommended) |
| CUDA | **12.x** (matching PyTorch/CuPy build) |
| RAM | 8 GB+ |

> A CUDA-capable GPU is strongly recommended for interactive experiments at high particle counts.

---

## Quick Start

## Installation

```bash
pip install -r requirements.txt
```

Requires: Python 3.10+, NVIDIA GPU with CUDA

Note: This simulation requires CUDA 12.x (CUDA 12.0+ builds). If your system has a different CUDA major version installed, install CUDA 12.x and matching CuPy/PyTorch wheels. The project enforces CUDA 12.x only.

Install (inside a virtual env):

```bash
python -m venv .venv
# Windows PowerShell
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Run headless benchmark:

```bash
python ball_sim.py --particles 100000 --duration 60
```

Run interactive visualization:

```bash
python ball_sim.py --visualize --particles 50000
```

---

## Future Improvements

- 3D visualization backend
- More physically-accurate integrators & collision handling
- Mini-games and interactive scenarios
- Improved UI and in-app presets

---

## Contributing

See [`CONTRIBUTING.md`](CONTRIBUTING.md)  give ideas, suggest more good things, submit experiments, or collaborate.

---

## License

This repository is available under the [MIT License](LICENSE).

---

*Have fun experimenting, this is a sandbox for trying computationally-interesting visual ideas.*
