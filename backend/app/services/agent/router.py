"""Direct module re-export for app.services.agent.router.

Provides multi-modal routing and input inspection compatibility.
"""

from __future__ import annotations

import sys
from app.agents.router import *
from app.agents.router import (
    InputInspectorNode,
    STANDARDIZED_TASK_MAP,
    INTERNAL_TASK_MAP,
    TEMPORAL_PHRASES,
    TEMPORAL_WORDS,
    GROUNDING_TRIGGERS,
    LANDCOVER_TRIGGERS,
    TASK_SINGLE_GROUNDING,
    TASK_SINGLE_VQA,
    TASK_BITEMPORAL_CHANGE,
    TASK_CROSS_MODAL,
    TASK_DOMAIN_KNOWLEDGE_QA,
    INTERNAL_SINGLE_GROUNDING,
    INTERNAL_SINGLE_VQA,
    INTERNAL_BITEMPORAL_CHANGE,
    INTERNAL_CROSS_MODAL,
    INTERNAL_DOMAIN_KNOWLEDGE_QA,
)

# Register aliases in sys.modules
sys.modules.setdefault("app.services.agent.router", sys.modules[__name__])
sys.modules.setdefault("backend.app.services.agent.router", sys.modules[__name__])
