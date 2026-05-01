"""Context prompts for LLM-based scene generation.

This module contains prompt templates for various stages of scene generation:
- semantic_plan: Generate semantic plans with explicit placement instructions
- constraints: Generate constraint-based placement plans (legacy)
- place: Generate simple position-based placement (legacy)
- objects: Object selection and disambiguation
- model: Model database queries
- scale: Object scaling
- expand: Scene expansion
- disambiguation: Resolve ambiguous object references
"""

from creator.contexts_prompts.semantic_plan import fmt_semantic_plan_tmpl
from creator.contexts_prompts.constraints import (
    fmt_constraints_plan_tmpl,
    fmt_seating_plan_tmpl,
)
from creator.contexts_prompts.place import fmt_place_qa_tmpl

__all__ = [
    "fmt_semantic_plan_tmpl",
    "fmt_constraints_plan_tmpl",
    "fmt_seating_plan_tmpl",
    "fmt_place_qa_tmpl",
]
