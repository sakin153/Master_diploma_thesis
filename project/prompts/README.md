# LLM Prompts Directory

This directory contains all LLM prompt templates used by the world-creator project.

## Prompt Files

### Scene Planning Prompts (scene_prompts.py)

1. **classify_hierarchy.txt** - Builds scene graph hierarchy from object list
2. **anchor_relations.txt** - Defines relationships between top-level (anchor) objects
3. **group_relations.txt** - Defines spatial relationships inside one anchor's group
4. **place_node.txt** - Places a single object instance
5. **place_node_batch.txt** - Places multiple identical objects at once
6. **coplace.txt** - Places mixed-type objects in coordinated layout
7. **place_batch_row.txt** - Places objects in a straight row pattern
8. **place_batch_grid.txt** - Places objects in a grid pattern
9. **place_batch_fix.txt** - Fixes placement violations from previous attempts
10. **surface_arrangement_plan.txt** - Decides horizontal arrangement for objects on surfaces
11. **layering_plan.txt** - Decides vertical stacking for objects in containers

### Model Selection Prompts (model_picker.py)

12. **model_disambiguation.txt** - Picks the best 3D model from candidate list

### Scene Expansion Prompts (prompt_expander.py)

13. **expand_prompt.txt** - Expands user query into detailed scene specification

### Physics Classification Prompts

14. **physics_classify.txt** - Classifies objects as static or dynamic for physics simulation

## Usage

All prompts are loaded automatically by their respective modules using the `_load_prompt()` helper function. The prompts support Python string formatting with `{variable}` placeholders for dynamic content.

## Editing Prompts

When editing prompts:
1. Keep the format consistent with existing prompts
2. Preserve all `{variable}` placeholders - they are replaced at runtime
3. Test changes by running the relevant module
4. Do not change the filename without updating the corresponding Python module

## File Organization

- Prompts are plain text files with `.txt` extension
- Each prompt is self-contained and includes its own instructions
- Related prompts (e.g., placement variants) share similar structure
- Dynamic prompts (with many runtime variables) are kept separate from static ones
