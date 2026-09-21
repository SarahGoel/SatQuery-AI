"""Abstract BaseTool class for SatQuery AI remote sensing specialist tools."""

from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from typing import Any, Dict


class BaseTool(ABC):
    """Modular tool interface for Earth Observation & remote sensing specialists."""

    name: str
    description: str
    parameters: Dict[str, Any]

    @abstractmethod
    async def execute(self, scratchpad: Dict[str, Any], **kwargs: Any) -> Dict[str, Any]:
        """Execute the tool asynchronously, reading and mutating the shared agent scratchpad."""
        pass

    def run(self, scratchpad: Dict[str, Any], **kwargs: Any) -> Dict[str, Any]:
        """Synchronous convenience wrapper to execute the tool."""
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = None

        if loop is not None and loop.is_running():
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                return pool.submit(asyncio.run, self.execute(scratchpad, **kwargs)).result()
        return asyncio.run(self.execute(scratchpad, **kwargs))
