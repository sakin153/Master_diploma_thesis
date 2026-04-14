#!/usr/bin/env python3
"""Render scene_graph_latest.json into publication-friendly graph images.

Usage:
  python visualize_scene_graph.py
  python visualize_scene_graph.py --input scene_graph_latest.json --output scene_graph_latest

Outputs:
  - <output>.dot (always)
  - <output>.png (if Graphviz `dot` is available)
  - <output>.svg (if Graphviz `dot` is available)
"""

from __future__ import annotations

import argparse
import json
import math
import re
import shutil
import subprocess
from pathlib import Path
from typing import Dict, List, Tuple


def _sanitize_id(raw: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_]", "_", raw or "") or "node"


def _label_for_node(node: Dict[str, object]) -> str:
    node_id = str(node.get("id", ""))
    name = str(node.get("name", "Unknown"))
    constraints_count = int(node.get("constraints_count", 0) or 0)
    return f"{name}\\n({node_id})\\nconstraints={constraints_count}"


def _color_for_name(name: str) -> str:
    palette = [
        "#d6eaf8",
        "#d5f5e3",
        "#fdebd0",
        "#f5eef8",
        "#f9ebea",
        "#e8f8f5",
        "#fef9e7",
    ]
    idx = sum(ord(ch) for ch in name) % len(palette)
    return palette[idx]


def _edge_label(edge: Dict[str, object]) -> str:
    e_type = str(edge.get("type", "relation"))
    w = edge.get("weight", None)
    value = edge.get("value", None)
    distance = edge.get("distance", None)

    parts: List[str] = [e_type]
    if isinstance(w, (int, float)):
        parts.append(f"w={w:.2f}")
    if value not in (None, "", "None"):
        parts.append(f"v={value}")
    if isinstance(distance, list) and len(distance) == 2:
        parts.append(f"d=[{distance[0]}, {distance[1]}]")
    return " | ".join(parts)


def build_dot(graph: Dict[str, object]) -> str:
    nodes = graph.get("nodes", [])
    edges = graph.get("edges", [])
    query = str(graph.get("query", ""))

    lines: List[str] = []
    lines.append("digraph SceneGraph {")
    lines.append('  graph [rankdir=LR, fontsize=12, fontname="Helvetica", labelloc=t, pad=0.2];')
    lines.append('  node [shape=box, style="rounded,filled", color="#455a64", penwidth=1.1, fontname="Helvetica", fontsize=10];')
    lines.append('  edge [color="#546e7a", penwidth=1.0, arrowsize=0.7, fontname="Helvetica", fontsize=9];')
    lines.append(f'  label="Scene Graph: {query.replace(chr(34), chr(39))}";')

    known_ids = set()
    for node in nodes:
        if not isinstance(node, dict):
            continue
        node_id = str(node.get("id", "")).strip()
        if not node_id:
            continue
        known_ids.add(node_id)
        name = str(node.get("name", "Unknown"))
        node_key = _sanitize_id(node_id)
        label = _label_for_node(node).replace('"', "'")
        fill = _color_for_name(name)
        lines.append(f'  "{node_key}" [label="{label}", fillcolor="{fill}"];')

    virtual_idx = 0
    for edge in edges:
        if not isinstance(edge, dict):
            continue
        src = str(edge.get("from", "")).strip()
        if not src:
            continue

        dst = str(edge.get("to", "")).strip()
        src_key = _sanitize_id(src)
        label = _edge_label(edge).replace('"', "'")
        hard = bool(edge.get("hard", False))
        style = "solid" if hard else "dashed"

        if dst and dst in known_ids:
            dst_key = _sanitize_id(dst)
            lines.append(
                f'  "{src_key}" -> "{dst_key}" [label="{label}", style="{style}"];'
            )
        else:
            # Constraints without explicit target become virtual annotation nodes.
            virtual_idx += 1
            to_name = str(edge.get("to_name", "")).strip()
            v_label = label if not to_name else f"{label}\\n(target={to_name})"
            v_key = f"virtual_{virtual_idx}"
            lines.append(
                f'  "{v_key}" [shape=note, fillcolor="#eceff1", label="{v_label}", color="#90a4ae"];'
            )
            lines.append(
                f'  "{src_key}" -> "{v_key}" [style="{style}", color="#78909c"];'
            )

    lines.append("}")
    return "\n".join(lines) + "\n"


def render_with_graphviz(dot_path: Path, output_base: Path) -> Tuple[bool, List[Path]]:
    dot_bin = shutil.which("dot")
    if dot_bin is None:
        return False, []

    created: List[Path] = []
    for fmt in ("png", "svg"):
        out_path = output_base.with_suffix(f".{fmt}")
        subprocess.run(
            [dot_bin, f"-T{fmt}", str(dot_path), "-o", str(out_path)],
            check=True,
        )
        created.append(out_path)
    return True, created


def render_simple_svg(graph: Dict[str, object], output_svg: Path) -> None:
    nodes_raw = [n for n in graph.get("nodes", []) if isinstance(n, dict)]
    edges_raw = [e for e in graph.get("edges", []) if isinstance(e, dict)]

    width = 1600
    height = 1100
    cx = width / 2.0
    cy = height / 2.0
    radius = min(width, height) * 0.36

    node_ids: List[str] = []
    node_map: Dict[str, Dict[str, object]] = {}
    for node in nodes_raw:
        node_id = str(node.get("id", "")).strip()
        if not node_id:
            continue
        if node_id in node_map:
            continue
        node_map[node_id] = node
        node_ids.append(node_id)

    if not node_ids:
        output_svg.write_text(
            "<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"1200\" height=\"300\">"
            "<text x=\"40\" y=\"80\" font-size=\"32\" font-family=\"Helvetica\">"
            "Scene graph is empty"
            "</text></svg>",
            encoding="utf-8",
        )
        return

    positions: Dict[str, Tuple[float, float]] = {}
    for i, node_id in enumerate(node_ids):
        a = (2.0 * math.pi * i) / max(1, len(node_ids))
        x = cx + radius * math.cos(a)
        y = cy + radius * math.sin(a)
        positions[node_id] = (x, y)

    pieces: List[str] = []
    pieces.append(
        f"<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"{width}\" height=\"{height}\">"
    )
    pieces.append(
        "<rect x=\"0\" y=\"0\" width=\"100%\" height=\"100%\" fill=\"#ffffff\"/>"
    )
    query = str(graph.get("query", "")).replace("&", "&amp;").replace("<", "&lt;")
    pieces.append(
        "<text x=\"40\" y=\"50\" font-size=\"28\" font-family=\"Helvetica\" fill=\"#263238\">"
        f"Scene Graph: {query}</text>"
    )

    virtual_offsets: Dict[str, int] = {}
    for edge in edges_raw:
        src = str(edge.get("from", "")).strip()
        dst = str(edge.get("to", "")).strip()
        if src not in positions:
            continue
        x1, y1 = positions[src]
        label = _edge_label(edge).replace("&", "&amp;").replace("<", "&lt;")
        hard = bool(edge.get("hard", False))
        dash = "" if hard else " stroke-dasharray=\"7 5\""

        if dst in positions:
            x2, y2 = positions[dst]
        else:
            # Keep no-target constraints visible as virtual annotations.
            k = virtual_offsets.get(src, 0)
            virtual_offsets[src] = k + 1
            x2 = x1 + 130.0
            y2 = y1 + (k * 22.0) - 30.0
            pieces.append(
                f"<rect x=\"{(x2 - 68):.1f}\" y=\"{(y2 - 13):.1f}\" width=\"136\" height=\"24\" "
                f"rx=\"6\" ry=\"6\" fill=\"#eceff1\" stroke=\"#90a4ae\" stroke-width=\"1.0\"/>"
            )
            pieces.append(
                f"<text x=\"{x2:.1f}\" y=\"{(y2 + 4):.1f}\" font-size=\"10\" font-family=\"Helvetica\" "
                f"fill=\"#455a64\" text-anchor=\"middle\">{label}</text>"
            )

        pieces.append(
            f"<line x1=\"{x1:.1f}\" y1=\"{y1:.1f}\" x2=\"{x2:.1f}\" y2=\"{y2:.1f}\" "
            f"stroke=\"#607d8b\" stroke-width=\"1.8\"{dash}/>")
        mx = (x1 + x2) / 2.0
        my = (y1 + y2) / 2.0
        if dst in positions:
            pieces.append(
                f"<text x=\"{mx:.1f}\" y=\"{my:.1f}\" font-size=\"12\" font-family=\"Helvetica\" "
                f"fill=\"#455a64\" text-anchor=\"middle\">{label}</text>"
            )

    for node_id in node_ids:
        node = node_map[node_id]
        name = str(node.get("name", "Unknown"))
        fill = _color_for_name(name)
        x, y = positions[node_id]
        width_box = 190
        height_box = 62
        rx = x - width_box / 2.0
        ry = y - height_box / 2.0
        label_1 = name.replace("&", "&amp;").replace("<", "&lt;")
        label_2 = str(node_id).replace("&", "&amp;").replace("<", "&lt;")
        pieces.append(
            f"<rect x=\"{rx:.1f}\" y=\"{ry:.1f}\" width=\"{width_box}\" height=\"{height_box}\" "
            f"rx=\"10\" ry=\"10\" fill=\"{fill}\" stroke=\"#455a64\" stroke-width=\"1.2\"/>"
        )
        pieces.append(
            f"<text x=\"{x:.1f}\" y=\"{(y - 4):.1f}\" font-size=\"13\" font-family=\"Helvetica\" "
            f"fill=\"#263238\" text-anchor=\"middle\">{label_1}</text>"
        )
        pieces.append(
            f"<text x=\"{x:.1f}\" y=\"{(y + 14):.1f}\" font-size=\"11\" font-family=\"Helvetica\" "
            f"fill=\"#37474f\" text-anchor=\"middle\">{label_2}</text>"
        )

    pieces.append("</svg>")
    output_svg.write_text("\n".join(pieces) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Visualize scene graph JSON as image")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("scene_graph_latest.json"),
        help="Path to scene graph JSON",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("scene_graph_latest_viz"),
        help="Output file base name (without extension)",
    )
    args = parser.parse_args()

    with args.input.open("r", encoding="utf-8") as f:
        graph = json.load(f)

    dot_text = build_dot(graph)
    dot_path = args.output.with_suffix(".dot")
    dot_path.write_text(dot_text, encoding="utf-8")
    print(f"DOT saved: {dot_path}")

    rendered, created = render_with_graphviz(dot_path, args.output)
    if rendered:
        for p in created:
            print(f"Image saved: {p}")
    else:
        fallback_svg = args.output.with_name(args.output.name + "_simple").with_suffix(".svg")
        render_simple_svg(graph, fallback_svg)
        print(f"Image saved (fallback SVG): {fallback_svg}")
        print("Graphviz `dot` was not found. For high-quality PNG/SVG export install graphviz.")
        print("On Fedora: sudo dnf install graphviz")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
