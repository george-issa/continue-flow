"""poleflow -- analytic continuation of imaginary-time Green's functions.

Maps a noisy DQMC :math:`G(\\tau)` to real-frequency spectral functions
:math:`A(\\omega)` written as sums of poles, with no frequency grid.

Two models share the encoder and the pole representation:

* the flow model (:mod:`poleflow.models.flow`) with its posterior report
  (:mod:`poleflow.posterior`): many spectra that fit the data, with intervals
  on the spectral weight;
* the earlier variational autoencoder (:mod:`poleflow.models.vae`): one pole
  set per input.

See :mod:`poleflow.physics` for the forward map and the constraint algebra, and
``docs/DESIGN.md`` for the design and the literature.
"""

__version__ = "0.1.0"

from . import physics  # noqa: F401
from .diagnostics import AnsatzFloor, ansatz_chi2_floor, tail_weight_bound  # noqa: E402
from .io import load_checkpoint, load_model, save_model  # noqa: E402
from .refine import RefineConfig, RefineResult, refine_latent  # noqa: E402

__all__ = [
    "physics",
    "ansatz_chi2_floor",
    "AnsatzFloor",
    "tail_weight_bound",
    "load_model",
    "save_model",
    "load_checkpoint",
    "refine_latent",
    "RefineConfig",
    "RefineResult",
    "__version__",
]
