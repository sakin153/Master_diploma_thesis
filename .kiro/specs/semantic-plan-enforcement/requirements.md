# Requirements Document: Semantic Plan Enforcement

## Introduction

Система World Creator генерирует 3D сцены MuJoCo из текстовых описаний на русском и английском языках. 

**Новая архитектура**: LLM является главным композитором и архитектором сцены. LLM полностью контролирует размещение объектов, их ориентацию относительно друг друга (смотрит на объект, стоит спиной, сбоку слева/справа, на расстоянии X метров и т.д.), и все пространственные взаимосвязи.

**Текущая проблема**: Семантический план, созданный LLM с детальными инструкциями по размещению, не соблюдается точно при финальной сборке сцены. Например, запрос "Простая столовая где есть 4 стола и 16 стульев у каждого стола по 4 стула" должен генерировать сцену где:
- Ровно 4 стола и 16 стульев
- Каждый стул стоит лицом к своему столу
- Каждый стул расположен по середине бока стола на правильном расстоянии
- Все объекты имеют правильную ориентацию согласно логике сцены

**Цель функции**: Обеспечить, чтобы детальный семантический план LLM (количество объектов, точное расположение, ориентация, расстояния) строго соблюдался на всех этапах генерации сцены: от создания плана LLM до финальной сборки MuJoCo XML. Система должна быть "тупым исполнителем" инструкций LLM, точно реализуя все указания по размещению и ориентации объектов.

## Glossary

- **LLM_Scene_Composer**: Главный архитектор сцены - LLM, который создает детальный семантический план с точными инструкциями по размещению и ориентации каждого объекта
- **Semantic_Plan**: Детальный структурированный план от LLM, содержащий для каждого объекта: точное количество, абсолютные/относительные координаты, ориентацию (yaw/pitch/roll или относительную: "лицом к столу", "спиной к двери"), расстояния до других объектов, и все пространственные взаимосвязи
- **Placement_Executor**: "Тупой исполнитель" - система, которая буквально выполняет инструкции из Semantic_Plan без собственных решений или оптимизаций
- **Orientation_Resolver**: Компонент, который преобразует относительные ориентации ("лицом к столу", "спиной к окну") в абсолютные углы (yaw/pitch/roll в градусах)
- **Distance_Resolver**: Компонент, который преобразует относительные расстояния ("близко к столу", "на расстоянии 0.5м от стула") в абсолютные координаты
- **Scene_Assembly**: Процесс создания финального MuJoCo XML файла из размещенных объектов с сохранением всех параметров из Semantic_Plan
- **Constraint_Validator**: Компонент для проверки, что финальная сцена точно соответствует всем инструкциям из Semantic_Plan
- **MuJoCo_Interface**: Интерфейс для работы с физическим движком MuJoCo
- **Placement_Solution**: Результат работы Placement_Executor, содержащий абсолютные координаты и ориентацию всех объектов согласно Semantic_Plan

## Requirements

### Requirement 1: LLM Generates Detailed Placement Instructions

**User Story:** Как LLM-композитор сцены, я хочу генерировать детальные инструкции по размещению каждого объекта, чтобы полностью контролировать итоговую сцену.

#### Acceptance Criteria

1. WHEN THE LLM_Scene_Composer SHALL create a Semantic_Plan, THE Semantic_Plan SHALL contain exact count for each object type
2. FOR EACH object in Semantic_Plan, THE LLM_Scene_Composer SHALL specify either absolute position (x, y, z coordinates) OR relative position (relative_to object_id, direction, distance)
3. FOR EACH object in Semantic_Plan, THE LLM_Scene_Composer SHALL specify orientation as either absolute (yaw_deg, pitch_deg, roll_deg) OR relative (facing object_id, facing_direction: "front", "back", "left_side", "right_side")
4. WHEN THE user query SHALL specify relationships ("у каждого стола по 4 стула"), THE LLM_Scene_Composer SHALL generate explicit placement instructions for each chair relative to its table
5. THE Semantic_Plan SHALL include for each object: id, Model, type, position (absolute or relative), orientation (absolute or relative), size, is_static, model_loc
6. THE LLM_Scene_Composer SHALL specify distance constraints with numeric values (e.g., "0.5 meters from table", "touching the wall")
7. WHEN THE object SHALL be placed relative to another object, THE Semantic_Plan SHALL specify the reference point on target object (center, front_edge, back_edge, left_edge, right_edge, top_surface)

### Requirement 2: Resolve Relative Positions to Absolute Coordinates

**User Story:** Как Placement_Executor, я хочу преобразовывать относительные позиции из Semantic_Plan в абсолютные координаты, чтобы точно разместить объекты согласно инструкциям LLM.

#### Acceptance Criteria

1. WHEN THE Semantic_Plan SHALL contain relative position (relative_to: "table_1", direction: "front", distance: 0.5), THE Distance_Resolver SHALL calculate absolute (x, y, z) coordinates
2. THE Distance_Resolver SHALL support directions: "front", "back", "left", "right", "front_left", "front_right", "back_left", "back_right", "above", "below"
3. WHEN THE direction SHALL be "front", THE Distance_Resolver SHALL place object in front of target's front face at specified distance
4. WHEN THE direction SHALL be "left", THE Distance_Resolver SHALL place object to the left of target's center at specified distance
5. THE Distance_Resolver SHALL use target object's size and orientation to calculate correct placement
6. WHEN THE reference_point SHALL be specified (e.g., "front_edge"), THE Distance_Resolver SHALL calculate position relative to that specific point on target object
7. THE Distance_Resolver SHALL preserve the z-coordinate (height) based on object type: floor objects at z=0, table-top objects at table surface height

### Requirement 3: Resolve Relative Orientations to Absolute Angles

**User Story:** Как Placement_Executor, я хочу преобразовывать относительные ориентации из Semantic_Plan в абсолютные углы, чтобы объекты смотрели в правильном направлении согласно инструкциям LLM.

#### Acceptance Criteria

1. WHEN THE Semantic_Plan SHALL contain relative orientation (facing: "table_1", facing_direction: "front"), THE Orientation_Resolver SHALL calculate absolute yaw angle in degrees
2. WHEN THE facing_direction SHALL be "front", THE Orientation_Resolver SHALL orient object to face the front of target object
3. WHEN THE facing_direction SHALL be "back", THE Orientation_Resolver SHALL orient object to face the back of target object
4. WHEN THE facing_direction SHALL be "left_side", THE Orientation_Resolver SHALL orient object to face the left side of target object
5. WHEN THE facing_direction SHALL be "right_side", THE Orientation_Resolver SHALL orient object to face the right side of target object
6. THE Orientation_Resolver SHALL calculate yaw angle based on target object's position and orientation
7. WHEN THE Semantic_Plan SHALL specify absolute orientation (yaw_deg: 90), THE Placement_Executor SHALL use it directly without modification
8. THE Orientation_Resolver SHALL support "facing_away" mode where object faces away from target instead of towards it

### Requirement 4: Execute Placement Without Modifications

**User Story:** Как Placement_Executor, я хочу размещать объекты точно согласно Semantic_Plan без собственных оптимизаций или изменений, чтобы LLM имел полный контроль над сценой.

#### Acceptance Criteria

1. WHEN THE Placement_Executor SHALL process Semantic_Plan, THE Placement_Executor SHALL place ALL objects from the plan without omissions
2. THE Placement_Executor SHALL NOT modify positions calculated from Semantic_Plan instructions
3. THE Placement_Executor SHALL NOT modify orientations calculated from Semantic_Plan instructions
4. THE Placement_Executor SHALL NOT apply automatic collision avoidance that changes LLM-specified positions
5. THE Placement_Executor SHALL NOT apply automatic spacing optimization that changes LLM-specified distances
6. IF THE Placement_Executor SHALL detect potential collisions, THE Placement_Executor SHALL log a warning but still place objects as instructed
7. THE Placement_Executor SHALL preserve all object metadata (uuid, model_loc, size, is_static) from Semantic_Plan to Placement_Solution
8. THE Placement_Solution SHALL contain exactly the same number of objects as Semantic_Plan with matching ids

### Requirement 5: Assemble Scene Exactly as Specified

**User Story:** Как Scene_Assembly, я хочу создавать MuJoCo XML точно по Placement_Solution без изменений, чтобы финальная сцена соответствовала инструкциям LLM.

#### Acceptance Criteria

1. WHEN THE Scene_Assembly SHALL create MuJoCo XML, THE Scene_Assembly SHALL include ALL objects from Placement_Solution without omissions
2. FOR EACH object in Placement_Solution, THE Scene_Assembly SHALL set position in XML to exact (x, y, z) coordinates from Placement_Solution
3. FOR EACH object in Placement_Solution, THE Scene_Assembly SHALL set orientation in XML to exact yaw_deg from Placement_Solution
4. FOR EACH object in Placement_Solution, THE Scene_Assembly SHALL use model_loc path to include correct 3D model
5. FOR EACH object in Placement_Solution, THE Scene_Assembly SHALL set is_static flag correctly in MuJoCo XML
6. THE Scene_Assembly SHALL verify that model_loc file exists before adding object to XML
7. IF THE model_loc file SHALL NOT exist, THE Scene_Assembly SHALL log error and skip that object
8. THE Scene_Assembly SHALL preserve object id from Placement_Solution as XML element name or attribute for traceability

### Requirement 6: Validate Semantic Plan Schema

**User Story:** Как система, я хочу валидировать схему Semantic_Plan перед обработкой, чтобы обнаруживать ошибки в инструкциях LLM до начала размещения.

#### Acceptance Criteria

1. THE Semantic_Plan SHALL follow a defined JSON schema with required fields: schema_version, objects, room_size
2. FOR EACH object in Semantic_Plan, THE System SHALL verify required fields: id, Model, type, size, is_static, model_loc
3. FOR EACH object in Semantic_Plan, THE System SHALL verify that position is specified as either absolute (x, y, z) OR relative (relative_to, direction, distance)
4. FOR EACH object in Semantic_Plan, THE System SHALL verify that orientation is specified as either absolute (yaw_deg) OR relative (facing, facing_direction)
5. WHEN THE position SHALL be relative, THE System SHALL verify that relative_to references an existing object id in the plan
6. WHEN THE orientation SHALL be relative, THE System SHALL verify that facing references an existing object id in the plan
7. IF THE schema validation SHALL fail, THE System SHALL return detailed error message with field name and validation rule that failed
8. THE System SHALL support schema_version field for backward compatibility with future schema changes

### Requirement 7: Track Objects Through Pipeline Stages

**User Story:** Как разработчик системы, я хочу отслеживать объекты на каждом этапе pipeline, чтобы диагностировать где теряются или искажаются данные.

#### Acceptance Criteria

1. THE System SHALL log object count and list of object ids at these stages: Semantic_Plan input, Distance_Resolver output, Orientation_Resolver output, Placement_Solution output, Scene_Assembly output
2. WHEN THE object count SHALL change between stages, THE System SHALL log which objects were added or removed with their ids
3. FOR EACH stage, THE System SHALL log object metadata: id, Model, position (absolute), orientation (absolute), size
4. THE System SHALL generate a pipeline trace file in JSON format containing full object data at each stage
5. THE pipeline trace file SHALL include timestamps, stage names, object counts, and complete object data
6. THE System SHALL log warnings when object metadata changes unexpectedly between stages (e.g., position shifts, orientation changes)
7. THE pipeline trace file SHALL be saved to a configurable location with timestamp in filename

### Requirement 8: Validate Final Scene Against Semantic Plan

**User Story:** Как пользователь системы, я хочу получать отчет о соответствии финальной сцены Semantic_Plan, чтобы понимать насколько точно выполнены инструкции LLM.

#### Acceptance Criteria

1. WHEN THE Scene_Assembly SHALL complete, THE Constraint_Validator SHALL compare final MuJoCo XML with original Semantic_Plan
2. THE Constraint_Validator SHALL verify that object count in XML matches object count in Semantic_Plan
3. FOR EACH object in Semantic_Plan, THE Constraint_Validator SHALL verify that corresponding object exists in XML with matching id
4. FOR EACH object, THE Constraint_Validator SHALL calculate position error: distance between planned and actual position in meters
5. FOR EACH object, THE Constraint_Validator SHALL calculate orientation error: angular difference between planned and actual yaw in degrees
6. THE Constraint_Validator SHALL generate a validation report with: total objects, missing objects, position errors, orientation errors
7. WHEN THE position error SHALL exceed 0.1 meters, THE Constraint_Validator SHALL mark object as "position mismatch"
8. WHEN THE orientation error SHALL exceed 5 degrees, THE Constraint_Validator SHALL mark object as "orientation mismatch"
9. THE validation report SHALL include overall compliance score (percentage of objects placed correctly)

### Requirement 9: Handle Placement Dependencies Correctly

**User Story:** Как Placement_Executor, я хочу правильно обрабатывать зависимости между объектами, чтобы объекты с относительным размещением располагались корректно.

#### Acceptance Criteria

1. WHEN THE Semantic_Plan SHALL contain objects with relative positions, THE Placement_Executor SHALL resolve dependencies in correct order
2. THE Placement_Executor SHALL place objects with absolute positions first, before objects that depend on them
3. WHEN THE object A SHALL have relative position to object B, THE Placement_Executor SHALL ensure object B is placed before object A
4. THE Placement_Executor SHALL detect circular dependencies (A depends on B, B depends on A) and report error
5. WHEN THE circular dependency SHALL be detected, THE System SHALL log error with list of objects in the cycle
6. THE Placement_Executor SHALL support multi-level dependencies (A depends on B, B depends on C, C has absolute position)
7. THE Placement_Executor SHALL build a dependency graph and use topological sort to determine placement order
8. IF THE dependency graph SHALL contain cycles, THE System SHALL fail with clear error message listing the cycle

### Requirement 10: Generate Detailed Placement Report

**User Story:** Как пользователь системы, я хочу получать детальный отчет о размещении объектов, чтобы понимать как была построена сцена и диагностировать проблемы.

#### Acceptance Criteria

1. THE System SHALL generate a placement report in JSON format after scene assembly completes
2. THE placement report SHALL include: Semantic_Plan summary (object count, room size), Placement_Solution summary (placed objects, failed objects), Scene_Assembly summary (XML objects, missing objects)
3. FOR EACH object, THE report SHALL include: id, Model, planned position, actual position, position error, planned orientation, actual orientation, orientation error
4. THE report SHALL include list of objects that failed to place with reason for failure
5. THE report SHALL include list of objects missing from final XML with reason
6. THE report SHALL include overall statistics: total objects planned, total objects placed, total objects in XML, average position error, average orientation error
7. THE report SHALL be saved to a configurable location with timestamp in filename
8. THE System SHALL provide a command-line tool to generate HTML visualization of the placement report

## Special Requirements Guidance

### LLM Prompt Engineering Requirements

Для того чтобы LLM мог эффективно выполнять роль композитора сцены, необходимо:

1. **Structured Output Format**: LLM должен генерировать Semantic_Plan в строго определенном JSON формате
2. **Explicit Instructions**: Промпт для LLM должен содержать четкие инструкции по заполнению всех полей (position, orientation, distances)
3. **Examples**: Промпт должен содержать примеры правильных Semantic_Plan для типичных сценариев (стол со стульями, комната с мебелью)
4. **Validation Feedback Loop**: Если Semantic_Plan не проходит валидацию схемы, система должна отправить ошибку обратно LLM для исправления

### Example Semantic Plan Structure

```json
{
  "schema_version": "1.0",
  "room_size": {"width": 10.0, "length": 10.0, "height": 3.0},
  "objects": [
    {
      "id": "table_1",
      "Model": "dining_table",
      "type": "furniture",
      "size": {"width": 1.5, "length": 0.8, "height": 0.75},
      "is_static": true,
      "model_loc": "models/tables/dining_table_01.xml",
      "position": {"absolute": {"x": 5.0, "y": 5.0, "z": 0.0}},
      "orientation": {"absolute": {"yaw_deg": 0.0}}
    },
    {
      "id": "chair_1",
      "Model": "dining_chair",
      "type": "furniture",
      "size": {"width": 0.5, "length": 0.5, "height": 0.9},
      "is_static": false,
      "model_loc": "models/chairs/dining_chair_01.xml",
      "position": {
        "relative": {
          "relative_to": "table_1",
          "direction": "front",
          "distance": 0.5,
          "reference_point": "front_edge"
        }
      },
      "orientation": {
        "relative": {
          "facing": "table_1",
          "facing_direction": "front"
        }
      }
    }
  ]
}
```

**Note**: Этот подход универсален и работает для любых комбинаций объектов (стол-стул, диван-столик, кровать-тумбочка, лампа-стол и т.д.). LLM просто указывает относительные позиции и ориентации для каждого объекта, а система их разрешает. Нет специальных паттернов или special-case логики - только универсальный алгоритм относительного размещения.

## Iteration and Feedback Rules

- Модель ДОЛЖНА вносить изменения, если пользователь запрашивает изменения
- Модель ДОЛЖНА учитывать всю обратную связь пользователя перед переходом к следующему этапу
- Модель ДОЛЖНА предложить вернуться к предыдущим этапам, если обнаружены пробелы

## Phase Completion

После завершения документа для этого этапа модель ДОЛЖНА остановиться. Пользователь нажмет кнопку в UI для перехода к следующему этапу.
