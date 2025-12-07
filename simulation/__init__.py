"""GPU-accelerated particle physics simulation."""

__version__ = '1.0.0'

from . import physics_torch
from . import particle_utils
from . import gpu_setup
from . import visualizer
from . import event_handler
from . import backend_stress
from . import metrics_sampler
from . import ui_components

__all__ = [
    'physics_torch',
    'particle_utils',
    'gpu_setup',
    'visualizer',
    'event_handler',
    'backend_stress',
    'metrics_sampler',
    'ui_components',
]
