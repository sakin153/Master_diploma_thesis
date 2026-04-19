import sys
import re
import json
import subprocess
import tempfile
import os
import urllib.request
import urllib.error

OLLAMA_URL = "http://localhost:11434/api/chat"
OLLAMA_MODEL = "qwen3.5:cloud"  # замени на свою модель

MUJOCO_PYTHON = "/home/sakin153/mujoco-env/bin/python"

SYSTEM_PROMPT = """You are an expert MuJoCo 3.3.7 scene designer.
Generate valid MJCF XML scenes based on user descriptions.

STRICT OUTPUT RULES:
- Output ONLY the XML. No explanations, no markdown, no ```xml blocks.
- Start directly with <mujoco model="scene">
- Always include: <option gravity="0 0 -9.81"/>, floor plane geom, at least one light

COLORS & MATERIALS:
- Do NOT use <asset>, <material>, or <texture> tags
- Do NOT use material="..." attribute on geoms
- Assign color directly on each geom using rgba="r g b a" (values 0.0–1.0)
- Do NOT use <visual><global> or any <global> tag

FORBIDDEN ELEMENTS (MuJoCo 3.3.7 does not support):
- No <geom type="cone"> — use cylinder or capsule instead
- No <accelerometer>, <gyro>, or <sensor> tags unless explicitly requested
- No <global> tags inside <visual>

NAMING (critical — MuJoCo rejects duplicates):
- Every name="..." must be globally unique across ALL elements (bodies, geoms, joints, actuators)
- Use numeric suffixes: geom1, geom2 — never repeat the same name
- Do NOT reuse the same name for a body and its child geom

DYNAMIC vs STATIC objects:
- Dynamic objects (fruits, balls, items that can move): place directly in <worldbody> with <freejoint/>
- Static objects (table, walls, floor): no joint
- NEVER nest a <body> with <freejoint/> inside another <body> — always in <worldbody>

BODY HIERARCHY (critical for editor compatibility):
- Prefer placing static geoms directly in <worldbody> without a <body> wrapper when possible
- If you use a <body> for a static object, put its world position in the <body pos="..."> and keep child geom positions as small local offsets from the body origin
- AVOID deeply nested body chains (body inside body inside body) — use at most 1 level of nesting for static objects
- Good: <body name="table" pos="0 0 0.44"><geom pos="0 0 0" .../></body>
- Bad:  <body name="table"><body name="top" pos="0 0 0.84"><geom pos="0 0 0" .../></body></body>

GEOMETRY sizes (all half-sizes):
- box:     size="hx hy hz"
- sphere:  size="radius"
- cylinder: size="radius half-height"
- capsule: fromto="x1 y1 z1  x2 y2 z2" size="radius"

OBJECT PLACEMENT (critical — prevents objects clipping into surfaces):
- Table surface is at z=0.88
- Place dynamic objects ABOVE their resting surface with extra clearance:
  z = surface_z + object_half_size + 0.05
  Example: sphere radius=0.04 on table → z = 0.88 + 0.04 + 0.05 = 0.97
- Objects will fall under gravity and land correctly — never place them exactly at contact height
- Keep x,y positions inside container boundaries (check box inner dimensions)

STRUCTURE TEMPLATE:
<mujoco model="scene">
  <option gravity="0 0 -9.81"/>
  <worldbody>
    <geom name="floor" type="plane" size="5 5 0.1" rgba="0.75 0.75 0.75 1"/>
    <light name="light1" pos="0 0 4" dir="0 0 -1"/>
    <body name="table1">
      <geom name="table_top1" type="box" size="0.5 0.3 0.04" pos="0 0 0.84" rgba="0.5 0.3 0.1 1"/>
    </body>
    <body name="apple1" pos="0.1 0 0.96">
      <freejoint/>
      <geom name="apple1_geom" type="sphere" size="0.04" rgba="0.9 0.1 0.1 1"/>
    </body>
  </worldbody>
</mujoco>
"""


class ChatSession:
    """Keeps conversation history for multi-turn Ollama /api/chat requests."""

    def __init__(self, model: str = OLLAMA_MODEL):
        self.model = model
        self.messages: list[dict] = [
            {"role": "system", "content": SYSTEM_PROMPT}
        ]

    def send(self, user_text: str) -> str:
        self.messages.append({"role": "user", "content": user_text})

        payload = json.dumps({
            "model": self.model,
            "messages": self.messages,
            "stream": False,
            "options": {"temperature": 0.3, "num_predict": 4096}
        }).encode()

        req = urllib.request.Request(
            OLLAMA_URL,
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST"
        )

        # Стриминг — читаем чанки, соединение не обрывается при долгой генерации
        payload_stream = json.dumps({
            "model": self.model,
            "messages": self.messages,
            "stream": True,
            "options": {"temperature": 0.3, "num_predict": 4096}
        }).encode()
        req_stream = urllib.request.Request(
            OLLAMA_URL,
            data=payload_stream,
            headers={"Content-Type": "application/json"},
            method="POST"
        )

        print(f"  Запрос к Ollama ({self.model}) [история: {len(self.messages)} сообщ.]...")
        try:
            reply_parts = []
            with urllib.request.urlopen(req_stream, timeout=600) as resp:
                for raw_line in resp:
                    line = raw_line.decode("utf-8").strip()
                    if not line:
                        continue
                    try:
                        chunk = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    token = chunk.get("message", {}).get("content", "")
                    reply_parts.append(token)
                    if chunk.get("done"):
                        break
            reply = "".join(reply_parts)
            self.messages.append({"role": "assistant", "content": reply})
            return reply
        except urllib.error.URLError as e:
            print(f"  Ошибка соединения с Ollama: {e}")
            print("  Убедись что Ollama запущена: ollama serve")
            sys.exit(1)


def extract_xml(text: str) -> str:
    # Убираем markdown-блоки если модель их добавила
    text = re.sub(r"```xml\s*", "", text)
    text = re.sub(r"```\s*", "", text)
    text = text.strip()

    # Вырезаем только XML-блок
    match = re.search(r"(<mujoco[\s\S]*</mujoco>)", text)
    if match:
        return match.group(1).strip()
    return text


def fix_duplicate_names(xml: str) -> str:
    """Auto-rename duplicate name attributes to make them unique."""
    import xml.etree.ElementTree as ET
    try:
        root = ET.fromstring(xml)
    except ET.ParseError:
        # Fallback: simple regex rename
        seen: dict = {}
        def replacer(m):
            name = m.group(1)
            if name in seen:
                seen[name] += 1
                return f'name="{name}_{seen[name]}"'
            seen[name] = 0
            return f'name="{name}"'
        return re.sub(r'name="([^"]+)"', replacer, xml)

    seen: dict = {}
    for elem in root.iter():
        name = elem.get("name")
        if name is None:
            continue
        if name in seen:
            seen[name] += 1
            elem.set("name", f"{name}_{seen[name]}")
        else:
            seen[name] = 0

    ET.indent(root, space="  ")
    return ET.tostring(root, encoding="unicode", xml_declaration=False)


def fix_nested_freejoints(xml: str) -> str:
    """Move freejoint bodies to worldbody, converting local coords to world coords."""
    import xml.etree.ElementTree as ET

    try:
        root = ET.fromstring(xml)
    except ET.ParseError:
        return xml  # невалидный XML — пусть валидатор выдаст ошибку

    worldbody = root.find("worldbody")
    if worldbody is None:
        return xml

    def get_pos(elem) -> tuple[float, float, float]:
        pos = elem.get("pos", "0 0 0").split()
        return tuple(float(v) for v in pos[:3])

    def accumulate_world_pos(body, parent_world):
        lx, ly, lz = get_pos(body)
        px, py, pz = parent_world
        return px + lx, py + ly, pz + lz

    def collect_freejoints(parent, parent_world_pos, to_move, to_remove):
        for child in list(parent):
            if child.tag != "body":
                continue
            world_pos = accumulate_world_pos(child, parent_world_pos)
            if child.find("freejoint") is not None and parent is not worldbody:
                child.set("pos", f"{world_pos[0]:.4f} {world_pos[1]:.4f} {world_pos[2]:.4f}")
                to_move.append(child)
                to_remove.append((parent, child))
            else:
                collect_freejoints(child, world_pos, to_move, to_remove)

    to_move, to_remove = [], []
    collect_freejoints(worldbody, (0.0, 0.0, 0.0), to_move, to_remove)

    for parent, child in to_remove:
        parent.remove(child)
    for body in to_move:
        worldbody.append(body)

    ET.indent(root, space="  ")
    return ET.tostring(root, encoding="unicode", xml_declaration=False)


def validate_xml(xml: str) -> tuple[bool, str]:
    with tempfile.NamedTemporaryFile(mode="w", suffix=".xml", delete=False) as f:
        f.write(xml)
        tmp_path = f.name

    try:
        result = subprocess.run(
            [MUJOCO_PYTHON, "-c",
             f"import mujoco; m = mujoco.MjModel.from_xml_path('{tmp_path}'); "
             f"print(f'bodies={{m.nbody}} geoms={{m.ngeom}} joints={{m.njnt}} actuators={{m.nu}}')"],
            capture_output=True, text=True, timeout=15
        )
        if result.returncode == 0:
            return True, result.stdout.strip()
        else:
            # Extract last meaningful error line + line number context
            lines = result.stderr.strip().split("\n")
            err = next((l for l in reversed(lines) if "Error" in l or "error" in l or "line" in l.lower()), lines[-1])
            return False, err
    finally:
        os.unlink(tmp_path)


def get_error_context(xml: str, error: str) -> str:
    """Extract lines around the error line number from the XML."""
    match = re.search(r'line (\d+)', error)
    if not match:
        return ""
    line_no = int(match.group(1))
    lines = xml.split("\n")
    start = max(0, line_no - 4)
    end = min(len(lines), line_no + 3)
    context = "\n".join(
        f"{'>>>' if i + 1 == line_no else '   '} {i+1}: {lines[i]}"
        for i in range(start, end)
    )
    return f"\nProblematic area (line {line_no}):\n{context}"


def review_positions(session: ChatSession, xml: str) -> str:
    """Ask the model to verify physical correctness of its own output."""
    review_prompt = f"""Review this MuJoCo XML for physical correctness. Check:
1. Are all objects (fruits, balls) positioned ABOVE their container surfaces, not below or inside geometry?
2. Are fruits inside their boxes (check x,y,z ranges match the box interior)?
3. Are any two objects overlapping (same pos)?

If everything is correct, return the XML unchanged.
If there are positioning errors, fix them and return ONLY the corrected XML.

XML:
{xml}"""
    raw = session.send(review_prompt)
    fixed = extract_xml(raw)
    return fixed if fixed.strip().startswith("<mujoco") else xml


def fix_xml_with_session(session: ChatSession, xml: str, error: str) -> str:
    context = get_error_context(xml, error)
    fix_prompt = f"""The XML you generated has a MuJoCo validation error. Fix it and return ONLY the corrected XML.

VALIDATION ERROR: {error}{context}

Common causes:
- Duplicate name="..." — every name must be globally unique
- <body> with <freejoint/> inside another <body> — move it to <worldbody>
- Wrong geom size format"""
    return session.send(fix_prompt)


def launch_viewer(xml_path: str):
    print(f"\n  Запускаю MuJoCo viewer... (закрой окно для выхода)")
    subprocess.run(
        [MUJOCO_PYTHON, "-c", f"""
import mujoco, mujoco.viewer, time
m = mujoco.MjModel.from_xml_path('{xml_path}')
d = mujoco.MjData(m)
print(f"bodies: {{m.nbody}} | geoms: {{m.ngeom}} | timestep: {{m.opt.timestep}}")
with mujoco.viewer.launch_passive(m, d) as v:
    real_start = time.time()
    sim_start  = d.time
    while v.is_running():
        mujoco.mj_step(m, d)
        v.sync()
        # Синхронизация: ждём если симуляция обгоняет реальное время
        elapsed_real = time.time() - real_start
        elapsed_sim  = d.time - sim_start
        if elapsed_sim > elapsed_real:
            time.sleep(elapsed_sim - elapsed_real)
"""],
        env={**os.environ, "DISPLAY": ":0"}
    )


def main():
    if len(sys.argv) > 1:
        user_prompt = " ".join(sys.argv[1:])
    else:
        print("Описание сцены (Enter для подтверждения):")
        user_prompt = input("> ").strip()
        if not user_prompt:
            print("Нет описания. Выход.")
            sys.exit(0)

    print(f"\n{'='*55}")
    print(f"  Сцена: {user_prompt}")
    print(f"  Модель: {OLLAMA_MODEL}")
    print(f"{'='*55}\n")

    # Создаём сессию — история сохраняется между запросами
    session = ChatSession(OLLAMA_MODEL)

    # Генерация
    raw = session.send(user_prompt)
    xml = extract_xml(raw)

    # Автофиксы структурных проблем
    xml = fix_nested_freejoints(xml)   # конвертирует локальные coords → мировые
    xml = fix_duplicate_names(xml)

    # Review-пасс: модель проверяет физическую корректность своего результата
    print("  Review позиций объектов...")
    xml = review_positions(session, xml)
    xml = fix_nested_freejoints(xml)
    xml = fix_duplicate_names(xml)

    # Валидация + до 2 попыток автофикса (модель помнит свой предыдущий ответ)
    for attempt in range(3):
        ok, info = validate_xml(xml)
        if ok:
            print(f"  Валидация OK — {info}")
            break
        print(f"  Попытка {attempt+1}: ошибка — {info}")
        if attempt < 2:
            print("  Прошу Ollama исправить (с контекстом истории)...")
            raw = fix_xml_with_session(session, xml, info)
            xml = extract_xml(raw)
            xml = fix_nested_freejoints(xml)
            xml = fix_duplicate_names(xml)
    else:
        print("\n  Не удалось получить валидный XML за 3 попытки.")
        print("  Сохраняю черновик в: scene_draft.xml")
        with open("scene_draft.xml", "w") as f:
            f.write(xml)
        sys.exit(1)

    # Сохраняем
    out_name = "scene_" + "_".join(user_prompt.split()[:3]).lower() + ".xml"
    out_name = re.sub(r"[^\w_.]", "", out_name)
    out_path = os.path.join(os.path.dirname(__file__), out_name)

    with open(out_path, "w") as f:
        f.write(xml)
    print(f"  Сохранено: {out_path}")

    # Запуск viewer (блокирующий — ждёт закрытия окна)
    launch_viewer(out_path)
    print("  Viewer закрыт.")


if __name__ == "__main__":
    main()
