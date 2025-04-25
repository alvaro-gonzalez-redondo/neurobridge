from __future__ import annotations

from typing import Optional

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .engine import SimulatorEngine
    import logging


engine: Optional[SimulatorEngine] = None
logger: Optional[logging.Logger] = None
