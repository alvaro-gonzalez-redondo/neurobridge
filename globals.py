from __future__ import annotations

from typing import Optional

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .engine import Simulator
    import logging


simulator: Optional[Simulator] = None
logger: Optional[logging.Logger] = None
