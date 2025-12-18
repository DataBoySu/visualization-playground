<p align="center">
  <img alt="Bouncing Balls" src="simulation/assets/textanim.gif"/>
</p>

![Status: WIP](https://img.shields.io/badge/status-WIP-orange.svg) ![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg) ![Cross-platform](https://img.shields.io/badge/platform-cross--platform-brightgreen) ![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)

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

This project uses a set of pragmatic, GPU-friendly approximations to produce visually interesting behavior while remaining performant at large particle counts. Below are the main formulas and the implementation notes showing how they map to the code.

### 1) Gravity / Pairwise Attraction

We use a softened inverse-square style influence computed pairwise and accumulated using vectorized GPU ops.

Vector form (conceptual):

$$\mathbf{F}_{ij} = G\; m_j \; \frac{\mathbf{r}_j - \mathbf{r}_i}{\|\mathbf{r}_j - \mathbf{r}_i\|^3 + \varepsilon}$$

Practical scalar form used in code (softening in denominator):

$$F_{ij} = G\;\frac{m_j}{\|\mathbf{r}_j - \mathbf{r}_i\|^2 + \varepsilon}$$

Notes:

- A small softening term $\varepsilon$ prevents singularities at short ranges.
- Forces are accumulated per-particle using GPU tensors (see `simulation/physics_torch.py`).

---

### 2) Collision response (impulse-based)

Collisions are resolved pairwise using the collision normal and the relative velocity along that normal.

Unit normal:

$$\mathbf{n} = \frac{\mathbf{r}_j - \mathbf{r}_i}{\|\mathbf{r}_j - \mathbf{r}_i\|}$$

Relative velocity:

$$v_{rel} = (\mathbf{v}_j - \mathbf{v}_i) \cdot \mathbf{n}$$

Classical instantaneous impulse magnitude (elastic model):

$$J = -\frac{(1+e)\,v_{rel}}{\dfrac{1}{m_i} + \dfrac{1}{m_j}}$$

Velocity updates (impulse application):

$$\Delta\mathbf{v}_i = \frac{J}{m_i}\mathbf{n},\quad \Delta\mathbf{v}_j = -\frac{J}{m_j}\mathbf{n}$$

Implementation notes:

- The code uses a mass-weighted impulse factor and an adjustable restitution (near-elastic for large masses) to produce lively collisions while remaining stable on the GPU.
- After impulses, overlapping pairs are separated to prevent sticking:

$$\text{separation} = \alpha\times\text{overlap},\quad \alpha\approx 0.6$$

---

### 3) Time integration

We use semi-implicit (symplectic) Euler integration:

1. $\mathbf{v} \leftarrow \mathbf{v} + \mathbf{a}\,\Delta t$
2. $\mathbf{x} \leftarrow \mathbf{x} + \mathbf{v}\,\Delta t$

With additional safeguards:

- Small-ball speeds are clamped to a configured maximum to keep visuals coherent.
- Bounce responses apply velocity sign flips with damping at boundaries.

---

### 4) Numerical stability and performance

- Add small epsilons to denominators (e.g., $\|\mathbf{r}\|^2 + \varepsilon$) to avoid divisions by zero.
- Use vectorized tensor operations and accumulation (e.g., `scatter_add`/index-add patterns) to handle multiple collisions affecting the same particle in one timestep.
- Defer mutations to the global active mask until after per-frame accumulation so array shapes remain consistent during GPU writes.

---

### 5) Game mechanics & spawning rules

- Big balls carry a scalar `health` and a `consec_non_own` counter.
- Same-color small-ball collisions:
  - Heal the big ball by +1
  - Absorb and deactivate the small ball
- Non-own-color collisions:
  - Decrease big ball health by 1
  - Increment `consec_non_own`
  - If `consec_non_own >= 50` the big ball "explodes" into several small balls (spawned from the inactive pool)

Implementation mapping:

- Health, consec counters and spawn logic are implemented in `simulation/physics_torch.py` (collision loop and post-collision spawn), and helper spawn routines in `simulation/particle_utils.py`.

---

For a concise code-to-formula guide, see the comments near the collision and attraction loops in `simulation/physics_torch.py`. If you want, I can add a short inline table mapping each equation above to the exact line ranges in that file. Would you like that?

---

## Where things are used

- **PyTorch (CUDA)** : physics & vector math on the GPU (`simulation/physics_torch.py`, `simulation/gpu_setup.py`).
- **Pygame** : visualization, UI, event handling (`simulation/visualizer.py`, `simulation/ui_components.py`).
- **CuPy** (optional) : alternate GPU compute backend.
- **numpy** : utility conversions and CPU fallbacks.

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
