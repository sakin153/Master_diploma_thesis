# Bugfix Requirements Document

## Introduction

This document specifies the requirements for fixing two critical bugs in the Universal Placement System:

1. **Orientation Bug**: Objects with `region:edge` constraint face walls instead of the room center, resulting in poor scene composition (e.g., sofas facing walls in a living room).

2. **Physics Bug**: Small objects are incorrectly marked as static (`is_static=true`) instead of dynamic, causing them to float in mid-air rather than respond to physics (e.g., apples in boxes remain frozen in space).

These bugs affect scene realism and usability. The fixes have been implemented in `floor_solver.py` and `universal_system.py`, but verification is needed to ensure the Universal System correctly integrates these fixes.

## Bug Analysis

### Current Behavior (Defect)

**Bug 1: Edge Object Orientation**

1.1 WHEN an object has a `region:edge` constraint and is placed near a wall THEN the system assigns random orientation causing the object to face the wall

1.2 WHEN generating a living room scene (prompt: "гостиная") with sofas THEN the sofas face walls instead of the room interior

**Bug 2: Small Object Physics**

1.3 WHEN an object has volume < 0.1 m³ (e.g., apples, books, small items) THEN the system marks it as `is_static=true`

1.4 WHEN generating a scene with small objects on surfaces (prompt: "Стол и две коробки на нем в каждой коробке по 3 яблока") THEN the apples float in mid-air and remain fixed in space

### Expected Behavior (Correct)

**Bug 1: Edge Object Orientation**

2.1 WHEN an object has a `region:edge` constraint and is placed near a wall THEN the system SHALL calculate yaw angle to face toward the room center (0, 0)

2.2 WHEN generating a living room scene (prompt: "гостиная") with sofas THEN the sofas SHALL face toward the room center with yaw calculated using `_compute_face_center_yaw(x, y)`

2.3 WHEN an edge object is positioned at coordinates (x, y) THEN the system SHALL use `_get_yaw_candidates_for_position()` to provide face-center yaw as the primary candidate

**Bug 2: Small Object Physics**

2.4 WHEN an object has volume < 0.1 m³ THEN the system SHALL set `is_static=false` to enable dynamic physics

2.5 WHEN generating a scene with small objects on surfaces (prompt: "Стол и две коробки на нем в каждой коробке по 3 яблока") THEN the apples SHALL be marked as dynamic and respond to physics simulation

2.6 WHEN the Universal System processes object metadata THEN it SHALL auto-detect `is_static` based on object volume using the threshold of 0.1 m³

### Unchanged Behavior (Regression Prevention)

**General Placement**

3.1 WHEN an object does NOT have a `region:edge` constraint THEN the system SHALL CONTINUE TO use default yaw candidates without face-center calculation

3.2 WHEN an object is placed in the room interior (not near walls) THEN the system SHALL CONTINUE TO use existing orientation logic

**Large Object Physics**

3.3 WHEN an object has volume ≥ 0.1 m³ (furniture, large items) THEN the system SHALL CONTINUE TO mark it as `is_static=true`

3.4 WHEN generating scenes with furniture (tables, chairs, sofas) THEN the furniture SHALL CONTINUE TO remain static and not respond to physics

**Scene Generation Pipeline**

3.5 WHEN the Universal System generates a scene THEN it SHALL CONTINUE TO use the existing pipeline stages (semantic planning, scene graph, layout solving, export)

3.6 WHEN exporting to MuJoCo XML format THEN the system SHALL CONTINUE TO include all object properties (pose, size, is_static) in the output

**Constraint Processing**

3.7 WHEN processing constraints other than `region:edge` (e.g., `near`, `on`, `inside`) THEN the system SHALL CONTINUE TO handle them with existing logic

3.8 WHEN multiple constraints are applied to an object THEN the system SHALL CONTINUE TO resolve them using the existing constraint engine
