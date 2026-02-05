from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Tuple, Any, List

import pandas as pd
import networkx as nx
from pyvis.network import Network

from graph_build import compute_radial_positions, node_winrate_color
from config import EMBED_ASSETS, FORMATS, RANGE_OPTIONS


def _apply_visual_overrides(net: Network, G: nx.DiGraph, archetypes_df: pd.DataFrame) -> None:
    positions = compute_radial_positions(archetypes_df)

    for n in net.nodes:
        node_id = n["id"]
        attrs = G.nodes[node_id]

        size_val = float(attrs.get("size", 25))
        matches_val = int(attrs.get("matches", 0))
        ow = attrs.get("overall_winrate")

        wr_txt = f"{ow * 100:.1f}%" if isinstance(ow, float) else "n/a"
        label = f"Winrate: {wr_txt}\nMatches: {matches_val:,}"
        # Keep text inside large nodes and readable inside small ones.
        font_size = max(8, min(18, size_val * 0.33))

        n["size"] = size_val
        n["display_label"] = node_id
        n["label"] = label
        n["title"] = attrs.get("title", node_id)
        n["shape"] = "circle"
        n["font"] = {
            "color": "#ffffff",
            "size": font_size,
            "strokeWidth": 3,
            "strokeColor": "#101010",
            "align": "center",
            "vadjust": 0,
        }
        n["matches"] = matches_val
        n["overall_winrate"] = float(ow) if isinstance(ow, float) else None

        if isinstance(ow, float):
            n["color"] = node_winrate_color(ow)

        if attrs.get("url"):
            n["url"] = attrs["url"]

        if node_id in positions:
            x, y = positions[node_id]
            n["x"] = x
            n["y"] = y
            n["fixed"] = True

    for e in net.edges:
        src = e["from"]
        dst = e["to"]
        attrs = G.edges[(src, dst)]
        e["width"] = float(attrs.get("width", 2))
        e["color"] = attrs.get("color", "#888888")
        e["label"] = ""
        e["title"] = attrs.get("title", "")
        e["matches"] = int(attrs.get("matches", 0))
        e["winrate"] = float(attrs.get("winrate", 0.5))
        e["winrate_from"] = float(attrs.get("winrate_from", attrs.get("winrate", 0.5)))
        e["neutral"] = bool(attrs.get("neutral", False))
        arrows = attrs.get("arrows", "to")
        e["arrows"] = arrows

        if attrs.get("neutral", False):
            e["arrows"] = {"to": {"enabled": False}}


def _build_dataset(G: nx.DiGraph, archetypes_df: pd.DataFrame) -> Dict[str, List[Dict[str, Any]]]:
    positions = compute_radial_positions(archetypes_df)

    nodes: List[Dict[str, Any]] = []
    label_nodes: List[Dict[str, Any]] = []
    for node_id, attrs in G.nodes(data=True):
        size_val = float(attrs.get("size", 25))
        matches_val = int(attrs.get("matches", 0))
        ow = attrs.get("overall_winrate")
        wr_txt = f"{ow * 100:.1f}%" if isinstance(ow, float) else "n/a"
        label = f"Winrate: {wr_txt}\nMatches: {matches_val:,}"
        font_size = max(8, min(18, size_val * 0.33))

        node: Dict[str, Any] = {
            "id": node_id,
            "display_label": node_id,
            "label": label,
            "size": size_val,
            "title": attrs.get("title", node_id),
            "shape": "circle",
            "font": {
                "color": "#ffffff",
                "size": font_size,
                "strokeWidth": 3,
                "strokeColor": "#101010",
                "align": "center",
                "vadjust": 0,
            },
            "matches": matches_val,
            "overall_winrate": float(ow) if isinstance(ow, float) else None,
        }

        if isinstance(ow, float):
            node["color"] = node_winrate_color(ow)

        if attrs.get("url"):
            node["url"] = attrs["url"]

        if node_id in positions:
            x, y = positions[node_id]
            node["x"] = x
            node["y"] = y
            node["fixed"] = True

            # External label node (archetype name) with uniform size.
            label_nodes.append(
                {
                    "id": f"label::{node_id}",
                    "base_id": node_id,
                    "is_label_node": True,
                    "label": node_id,
                    "shape": "text",
                    "physics": False,
                    "x": x,
                    "y": y - (size_val * 1.15 + 28),
                    "fixed": True,
                    "font": {
                        "color": "#ffffff",
                        "size": 16,
                        "strokeWidth": 3,
                        "strokeColor": "#101010",
                        "align": "center",
                        "vadjust": 0,
                    },
                }
            )

        nodes.append(node)

    edges: List[Dict[str, Any]] = []
    for idx, (src, dst, attrs) in enumerate(G.edges(data=True)):
        edge: Dict[str, Any] = {
            "id": f"e{idx}_{src}__{dst}",
            "from": src,
            "to": dst,
            "width": float(attrs.get("width", 2)),
            "color": attrs.get("color", "#888888"),
            "label": "",
            "title": attrs.get("title", ""),
            "matches": int(attrs.get("matches", 0)),
            "winrate": float(attrs.get("winrate", 0.5)),
            "winrate_from": float(attrs.get("winrate_from", attrs.get("winrate", 0.5))),
            "neutral": bool(attrs.get("neutral", False)),
            "arrows": attrs.get("arrows", "to"),
        }

        if edge["neutral"]:
            edge["arrows"] = {"to": {"enabled": False}}

        edges.append(edge)

    return {"nodes": nodes + label_nodes, "edges": edges}


def render_pyvis(
    graphs_by_format: Dict[str, Dict[str, Tuple[nx.DiGraph, pd.DataFrame]]],
    out_html: str,
    default_format_key: str,
    default_range_key: str,
) -> None:
    if default_format_key not in graphs_by_format:
        raise KeyError(f"default_format_key '{default_format_key}' not found in graphs_by_format")
    if default_range_key not in graphs_by_format[default_format_key]:
        raise KeyError(
            f"default_range_key '{default_range_key}' not found in graphs_by_format[{default_format_key!r}]"
        )

    default_G, default_df = graphs_by_format[default_format_key][default_range_key]

    net = Network(height="800px", width="100%", directed=True, bgcolor="#151a21", font_color="#e6e8ec")
    net.from_nx(default_G)
    _apply_visual_overrides(net, default_G, default_df)

    net.set_options(
        """
    var options = {
      "interaction": {
        "hover": true,
        "multiselect": true,
        "navigationButtons": false,
        "zoomView": true,
        "dragView": true
      },
      "physics": { "enabled": false },
      "nodes": { "shape": "dot" },
      "edges": {
        "smooth": { "type": "dynamic" },
        "font": { "align": "top" }
      }
    }
    """)

    net.write_html(out_html, open_browser=False, notebook=False)

    datasets_by_format: Dict[str, Dict[str, Any]] = {}
    for format_key, ranges_dict in graphs_by_format.items():
        format_meta = FORMATS.get(format_key, {})
        ranges_payload: Dict[str, Dict[str, Any]] = {}
        for range_key, (G, df) in ranges_dict.items():
            range_meta = RANGE_OPTIONS.get(range_key, {})
            ranges_payload[range_key] = {
                "key": range_key,
                "label": range_meta.get("label", range_key),
                **_build_dataset(G, df),
            }

        if not ranges_payload:
            continue

        datasets_by_format[format_key] = {
            "key": format_key,
            "label": format_meta.get("label", format_key),
            "ranges": ranges_payload,
        }

    inject_filter_ui(
        out_html,
        datasets_by_format=datasets_by_format,
        default_format_key=default_format_key,
        default_range_key=default_range_key,
    )
    if EMBED_ASSETS:
        inline_assets(out_html)


def inject_filter_ui(
    out_html: str,
    datasets_by_format: Dict[str, Dict[str, Any]],
    default_format_key: str,
    default_range_key: str,
) -> None:
    html_path = Path(out_html)
    html = html_path.read_text(encoding="utf-8")

    if "matchupBody" in html:
        return

    if default_format_key not in datasets_by_format:
        raise KeyError(f"default_format_key '{default_format_key}' not found in datasets_by_format")
    if default_range_key not in datasets_by_format[default_format_key].get("ranges", {}):
        raise KeyError(
            f"default_range_key '{default_range_key}' not found in datasets_by_format[{default_format_key!r}]['ranges']"
        )

    updated_at_utc = datetime.now(timezone.utc).strftime("%B %d, %Y at %H:%M UTC")
    datasets_json = json.dumps(datasets_by_format, ensure_ascii=True)

    format_options_html_parts: List[str] = []
    for key in FORMATS.keys():
        if key not in datasets_by_format:
            continue
        label = datasets_by_format[key].get("label", key)
        selected = " selected" if key == default_format_key else ""
        format_options_html_parts.append(f'<option value="{key}"{selected}>{label}</option>')
    format_options_html = "\n                    ".join(format_options_html_parts)

    default_ranges = datasets_by_format[default_format_key].get("ranges", {})
    range_options_html_parts: List[str] = []
    for key in RANGE_OPTIONS.keys():
        if key not in default_ranges:
            continue
        label = default_ranges[key].get("label", key)
        selected = " selected" if key == default_range_key else ""
        range_options_html_parts.append(f'<option value="{key}"{selected}>{label}</option>')
    range_options_html = "\n                    ".join(range_options_html_parts)

    extra_head = """
        <link rel="stylesheet" href="lib/vis-9.1.2/vis-network.css" />
        <script src="lib/vis-9.1.2/vis-network.min.js"></script>
        <link rel="stylesheet" href="lib/tom-select/tom-select.css" />
        <script src="lib/tom-select/tom-select.complete.min.js"></script>
    """

    extra_css = """
             @import url('https://fonts.googleapis.com/css2?family=Sora:wght@400;500;600;700&display=swap');

             :root {
                 --bg: #0f1216;
                 --panel: #151a21;
                 --panel-2: #1b212a;
                 --panel-3: #222a35;
                 --border: #2b3441;
                 --border-strong: #3a4657;
                 --text: #e6e8ec;
                 --muted: #9aa4b2;
                 --accent: #3b82f6;
                 --accent-2: #22c55e;
             }

             body {
                 background: var(--bg);
                 margin: 0;
                 color: var(--text);
                 font-family: 'Sora', sans-serif;
             }
             html, body {
                 height: 100%;
             }
             body {
                 overflow: hidden;
             }
             .card {
                 background: var(--panel);
                 border: 1px solid var(--border);
                 height: 100vh;
                 display: flex;
                 flex-direction: column;
                 border-radius: 0;
                 overflow: hidden;
             }
             #mynetwork {
                 background-color: var(--panel-2);
                 border: 1px solid var(--border);
                 border-radius: 12px;
             }
             /* Hide vis-network navigation buttons (arrows + zoom) */
             div.vis-network div.vis-navigation {
                 display: none !important;
             }
             .graph-controls {
                 display: flex;
                 align-items: stretch;
                 justify-content: space-between;
                 gap: 16px;
                 padding: 12px 14px;
                 border-bottom: 1px solid var(--border-strong);
                 background: var(--panel-3);
                 font-family: 'Sora', sans-serif;
                 color: var(--text);
                 border-radius: 0;
             }
             .graph-controls label {
                 font-size: 12px;
                 color: var(--muted);
                 margin: 0;
                 text-transform: uppercase;
                 letter-spacing: 0.06em;
             }
             .graph-controls .filters {
                 display: grid;
                 grid-template-columns: repeat(4, var(--filter-width));
                 gap: 14px 18px;
                 align-items: flex-start;
             }
             .graph-controls .filters {
                 --filter-width: 160px;
             }
             .graph-controls select {
                 color-scheme: dark;
             }
             .graph-controls .filter-col {
                 display: flex;
                 flex-direction: column;
                 gap: 6px;
                 min-width: 0;
             }
             .graph-controls .filter-col select {
                 width: 100%;
                 min-width: 0;
                 height: 30px;
                 border-radius: 0;
                 border: 1px solid var(--border);
                 background: #0e131a;
                 color: var(--text);
                 box-sizing: border-box;
             }
             .graph-controls .filter-col .ts-wrapper {
                 width: 100%;
                 min-width: 0;
             }
             .graph-controls .ts-wrapper,
             .graph-controls .ts-wrapper.single,
             .graph-controls .ts-control {
                 width: 100% !important;
             }
             .graph-controls .ts-control .item {
                 max-width: 100%;
                 overflow: hidden;
                 text-overflow: ellipsis;
                 white-space: nowrap;
             }
             .graph-controls .ts-control {
                 overflow: hidden;
                 gap: 6px;
             }
             .graph-controls .ts-control .item,
             .graph-controls .ts-control input {
                 flex: 1 1 auto;
                 min-width: 0;
             }
             .graph-controls .ts-control {
                 min-height: 30px;
                 height: 30px;
                 padding: 3px 8px;
                 border-radius: 0;
                 border: 1px solid var(--border);
                 background: #0e131a;
                 color: var(--text);
                 box-sizing: border-box;
                 line-height: 22px;
             }
             .graph-controls .ts-control input {
                 height: 22px;
                 color: var(--text);
                 background: transparent;
                 line-height: 22px;
             }
             .graph-controls .ts-control .item {
                 color: var(--text);
             }
             .graph-controls .ts-wrapper,
             .graph-controls .ts-wrapper.single {
                 background: #0e131a;
             }
             .graph-controls .ts-wrapper.single {
                 height: 30px;
                 min-height: 30px;
             }
             .graph-controls .ts-wrapper.single .ts-control {
                 background: #0e131a;
             }
             .graph-controls .ts-wrapper.single .ts-control,
             .graph-controls .ts-wrapper.single .ts-control > input {
                 height: 30px;
                 min-height: 30px;
                 max-height: 30px;
             }
             .graph-controls .ts-wrapper.single .ts-control .item {
                 line-height: 22px;
             }
             .graph-controls .ts-dropdown {
                 background: #0e131a;
                 border: 1px solid var(--border);
                 color: var(--text);
             }
             .graph-controls .ts-dropdown .ts-dropdown-content {
                 max-height: 220px;
                 overflow: auto;
                 scrollbar-color: #2b3441 #0b0f14;
                 scrollbar-width: thin;
             }
             .graph-controls .ts-dropdown .ts-dropdown-content::-webkit-scrollbar {
                 width: 10px;
             }
             .graph-controls .ts-dropdown .ts-dropdown-content::-webkit-scrollbar-track {
                 background: #0b0f14;
             }
             .graph-controls .ts-dropdown .ts-dropdown-content::-webkit-scrollbar-thumb {
                 background-color: #2b3441;
                 border: 2px solid #0b0f14;
                 border-radius: 8px;
             }
             .graph-controls .ts-wrapper {
                 width: 100%;
             }
             .graph-controls .ts-control {
                 flex-wrap: nowrap;
             }
             .graph-controls .ts-control .item {
                 color: var(--text);
                 background: transparent;
             }
             .graph-controls .ts-control input,
             .graph-controls .ts-control input:focus {
                 color: var(--text);
             }
             .graph-controls .ts-control::after {
                 border-color: var(--muted) transparent transparent transparent;
             }
             .graph-controls .ts-dropdown .active {
                 background: #121823;
                 color: var(--text);
             }
             .graph-controls .ts-dropdown .option {
                 background: #0e131a;
                 color: var(--text);
             }
             .graph-controls .ts-dropdown .option:hover,
             .graph-controls .ts-dropdown .option.active {
                 background: #182131;
                 color: var(--text);
             }
             .graph-controls .ts-wrapper.single.input-active .ts-control,
             .graph-controls .ts-wrapper.single.input-active .ts-control input {
                 background: #0e131a;
                 color: var(--text);
             }
             .graph-controls .ts-wrapper.single.input-active .ts-control {
                 box-shadow: none;
             }
             .graph-controls .actions {
                 display: flex;
                 flex-direction: column;
                 gap: 4px;
                 align-items: flex-end;
                 flex: 0 0 auto;
             }
             .graph-controls .spacer {
                 flex: 1 1 auto;
             }
             .graph-controls .hint-line {
                 font-size: 12px;
                 color: #dbe3ef;
                 text-decoration: none;
                 font-weight: 600;
                 white-space: nowrap;
                 align-self: flex-end;
                 text-align: right;
                 background: rgba(15, 23, 42, 0.8);
                 border: 1px solid #2f3b52;
                 border-radius: 10px;
                 padding: 4px 10px;
             }
             .sidepanel.hidden {
                 display: none;
             }
             .panel-header {
                 display: flex;
                 align-items: flex-start;
                 justify-content: space-between;
                 gap: 12px;
             }
             .panel-summary {
                 text-align: right;
                 min-width: 120px;
                 font-size: 12px;
                 line-height: 1.25;
             }
             .panel-summary .label {
                 color: var(--muted);
                 display: block;
             }
             .panel-summary .value {
                 font-weight: 700;
             }
             .node-summary-title {
                 font-size: 13px;
                 font-weight: 700;
                 margin-bottom: 4px;
                 color: var(--text);
             }
             .node-summary-row {
                 display: flex;
                 justify-content: space-between;
                 gap: 8px;
             }
             .node-summary-row strong {
                 color: var(--text);
             }
             .graph-controls .btn-mini {
                 border: 1px solid var(--border);
                 background: #111827;
                 color: var(--text);
                 padding: 4px 10px;
                 font-size: 12px;
                 cursor: pointer;
                 border-radius: 999px;
             }
             .help-btn {
                 border: 1px solid #3766a9;
                 background: #1d4ed8;
                 color: #eaf2ff;
                 padding: 4px 10px;
                 font-size: 12px;
                 font-weight: 600;
                 cursor: pointer;
                 border-radius: 999px;
             }
             .help-overlay {
                 display: none;
                 position: fixed;
                 inset: 0;
                 background: rgba(0,0,0,0.6);
                 z-index: 9999;
                 align-items: center;
                 justify-content: center;
             }
             .help-modal {
                 width: min(680px, 92vw);
                 background: var(--panel);
                 border: 1px solid var(--border);
                 border-radius: 8px;
                 padding: 14px 16px;
                 color: var(--text);
                 font-family: 'Sora', sans-serif;
                 font-size: 13px;
                 line-height: 1.4;
                 box-shadow: 0 10px 30px rgba(0,0,0,0.45);
             }
             .help-modal h4 {
                 margin: 0 0 6px 0;
                 font-size: 16px;
             }
             .help-modal ul {
                 margin: 6px 0 0 18px;
                 padding: 0;
             }
             .help-modal li {
                 margin: 3px 0;
             }
             .help-close {
                 float: right;
                 border: none;
                 background: transparent;
                 color: var(--muted);
                 font-size: 16px;
                 cursor: pointer;
             }
             .graph-layout {
                 display: flex;
                 gap: 0;
                 align-items: stretch;
                 position: relative;
                 flex: 1 1 auto;
                 min-height: 0;
                 overflow: hidden;
             }
             .graph-canvas {
                 position: relative;
                 flex: 1 1 auto;
                 min-width: 0;
                 display: flex;
             }
             #mynetwork {
                 flex: 1 1 auto;
                 min-width: 0;
                 height: 100% !important;
                 min-height: 0;
             }
             .sidepanel {
                 width: 360px;
                 padding: 10px 12px;
                 border-left: 1px solid var(--border-strong);
                 background: var(--panel-3);
                 font-family: 'Sora', sans-serif;
                 display: flex;
                 flex-direction: column;
                 min-height: 0;
                 border-radius: 0;
             }
             .sidepanel h3 {
                 font-size: 16px;
                 margin: 0 0 6px 0;
                 color: var(--text);
             }
             .sidepanel .sub {
                 font-size: 12px;
                 color: var(--muted);
                 margin-bottom: 8px;
             }
             .sidepanel table {
                 width: 100%;
                 border-collapse: collapse;
                 font-size: 12px;
                 background: #0f141b;
                 border-radius: 0;
                 table-layout: fixed;
             }
             .sidepanel th:nth-child(1),
             .sidepanel td:nth-child(1) { width: 56%; }
             .sidepanel th:nth-child(2),
             .sidepanel td:nth-child(2) { width: 22%; }
             .sidepanel th:nth-child(3),
             .sidepanel td:nth-child(3) { width: 22%; }
             .table-scroll {
                 flex: 1 1 auto;
                 min-height: 0;
                 overflow: auto;
                 border: 1px solid var(--border);
                 background: #0b0f14;
                 border-radius: 0;
             }
             .table-scroll {
                 scrollbar-color: #2b3441 #0b0f14;
                 scrollbar-width: thin;
             }
             .table-scroll::-webkit-scrollbar {
                 width: 10px;
             }
             .table-scroll::-webkit-scrollbar-track {
                 background: #0b0f14;
             }
             .table-scroll::-webkit-scrollbar-thumb {
                 background-color: #2b3441;
                 border: 2px solid #0b0f14;
                 border-radius: 8px;
             }
             .sidepanel th, .sidepanel td {
                 border-bottom: 1px solid #1f2833;
                 padding: 4px 6px;
                 text-align: left;
                 color: var(--text);
             }
             .sidepanel th {
                 background: #121823;
             }
             .sidepanel tbody tr {
                 background: #0c1118;
             }
             .sidepanel tbody tr:nth-child(even) {
                 background: #0f151e;
             }
             .sidepanel tbody td {
                 color: var(--text);
             }
             .sidepanel tbody td:nth-child(2) {
                 text-shadow: 0 1px 2px rgba(0,0,0,0.65);
                 font-weight: 600;
             }
             .graph-footer {
                 margin-top: 0;
                 padding: 8px 12px;
                 border-top: 1px solid var(--border);
                 color: var(--muted);
                 font-family: 'Sora', sans-serif;
                 font-size: 12px;
                 display: flex;
                 justify-content: space-between;
                 gap: 12px;
                 flex-wrap: wrap;
                 flex: 0 0 auto;
                 background: var(--panel);
                 border-radius: 0;
             }
             .graph-actions {
                 position: absolute;
                 left: 16px;
                 top: 12px;
                 display: flex;
                 gap: 10px;
                 z-index: 5;
                 pointer-events: auto;
             }
             .graph-action-btn {
                 border-radius: 999px;
                 padding: 8px 16px;
                 font-size: 12px;
                 font-weight: 600;
                 border: 1px solid var(--border);
                 background: rgba(17, 24, 39, 0.9);
                 color: var(--text);
                 cursor: pointer;
                 box-shadow: 0 6px 16px rgba(0,0,0,0.35);
             }
             .graph-action-btn.help {
                 border-color: #1d4ed8;
                 background: rgba(29, 78, 216, 0.95);
                 color: #eaf2ff;
             }
             .graph-footer a {
                 color: var(--accent);
                 text-decoration: none;
             }
             .graph-footer a:hover,
             .graph-footer a:focus {
                 text-decoration: underline;
             }
             @media (max-width: 980px) {
                 .graph-controls {
                     flex-wrap: wrap;
                     gap: 8px;
                 }
                 .graph-controls .filters {
                     grid-template-columns: repeat(2, minmax(160px, 1fr));
                     width: 100%;
                 }
                 .graph-controls .actions {
                     width: 100%;
                     align-items: flex-start;
                 }
                 .graph-controls select,
                 .graph-controls .btn-mini,
                 .graph-controls .help-btn {
                     min-height: 36px;
                     font-size: 13px;
                 }
                .graph-controls .hint-line {
                     width: 100%;
                     order: 99;
                     font-size: 11px;
                     white-space: normal;
                 }
                 .node-summary {
                     width: 100%;
                     order: 98;
                 }
                 .graph-layout {
                     flex-direction: column;
                 }
                 .sidepanel {
                     width: 100%;
                     border-left: none;
                     border-top: 1px solid var(--border);
                     max-height: 45vh;
                     overflow: auto;
                 }
                 #mynetwork {
                     height: 62vh !important;
                 }
             }
             @media (max-width: 640px) {
                 .graph-controls {
                     flex-direction: column;
                     align-items: stretch;
                 }
                 .graph-controls .filter-col {
                     width: 100%;
                 }
                 .graph-controls .filter-col select {
                     width: 100%;
                 }
                 .graph-controls .filters {
                     grid-template-columns: 1fr;
                 }
                 .graph-controls .actions {
                     width: 100%;
                 }
                 .graph-controls .btn-mini,
                 .graph-controls .help-btn {
                     width: 100%;
                 }
                .graph-controls .hint-line {
                    font-size: 11px;
                }
                 .sidepanel table {
                     font-size: 11px;
                 }
             }
    """

    controls_html = f"""
            <div class="graph-controls">
                <div class="filters">
                    <div class="filter-col">
                        <label for="formatFilter">Format</label>
                        <select id="formatFilter">
                            {format_options_html}
                        </select>
                    </div>
                    <div class="filter-col">
                        <label for="rangeFilter">Range</label>
                        <select id="rangeFilter">
                            {range_options_html}
                        </select>
                    </div>
                    <div class="filter-col">
                        <label for="nodeLimit">Nodes</label>
                        <select id="nodeLimit">
                            <option value="30">30</option>
                            <option value="20" selected>20</option>
                            <option value="10">10</option>
                        </select>
                    </div>
                    <div class="filter-col">
                        <label for="archetypeFilter">Archetype</label>
                        <select id="archetypeFilter" placeholder="All">
                            <option value="__all__">All</option>
                        </select>
                    </div>
                </div>
                <div class="spacer"></div>
                <div class="actions">
                    <span class="hint-line">To filter, click a node in the graph.</span>
                    <span class="hint-line">Zoom in to see details.</span>
                </div>
            </div>
            <div id="helpOverlay" class="help-overlay">
                <div class="help-modal" role="dialog" aria-modal="true">
                    <button class="help-close" id="helpClose" aria-label="Close">×</button>
                    <h4>How to read this graph</h4>
                    <ul>
                        <li><strong>Node size</strong> = overall matches played by that deck.</li>
                        <li><strong>Node color</strong> = overall winrate (red → yellow → green).</li>
                        <li><strong>Edge direction</strong> = which deck wins the matchup.</li>
                        <li><strong>Edge color</strong> = winrate (greener = better for the winner, redder = worse).</li>
                        <li><strong>Edge width</strong> = number of matches (thicker = more data).</li>
                        <li><strong>Format</strong> lets you switch between Modern, Standard, Legacy, Premodern, and Pauper.</li>
                        <li>When a deck is selected, it moves to the center and all colors/arrows are shown from its point of view.</li>
                    </ul>
                </div>
            </div>
    """

    sidepanel_html = """
            <div class="graph-layout">
                <div class="graph-canvas">
                    <div id="mynetwork" class="card-body"></div>
                    <div class="graph-actions">
                        <button id="resetFilter" class="graph-action-btn" type="button">Reset</button>
                        <button id="matrixBtn" class="graph-action-btn" type="button">Matrix</button>
                        <button id="helpBtn" class="graph-action-btn help" type="button">Help</button>
                    </div>
                </div>
                <div class="sidepanel hidden" id="sidepanel">
                    <div class="panel-header">
                        <div>
                            <h3 id="panelTitle">Select an archetype</h3>
                            <div class="sub" id="panelSubtitle">Matchups will be shown (sortable).</div>
                        </div>
                        <div class="panel-summary">
                            <span class="label">Winrate</span>
                            <span class="value" id="panelWinrate">—</span>
                            <span class="label" style="margin-top:4px;">Matches</span>
                            <span class="value" id="panelMatches">—</span>
                        </div>
                    </div>
                    <div class="table-scroll">
                        <table>
                            <thead>
                                <tr>
                                    <th id="thDeck" data-key="deck" title="Click to sort by Deck">Deck</th>
                                    <th id="thWinrate" data-key="winrate" title="Click to sort by Winrate">Winrate</th>
                                    <th id="thMatches" data-key="matches" title="Click to sort by Matches">Matches</th>
                                </tr>
                            </thead>
                            <tbody id="matchupBody"></tbody>
                        </table>
                    </div>
                </div>
            </div>
    """

    footer_html = f"""
            <div class="graph-footer">
                <span>Data source: <a href="https://mtgdecks.net/" target="_blank" rel="noopener noreferrer">mtgdecks.net</a></span>
                <span>Last updated: {updated_at_utc}</span>
            </div>
    """

    extra_js = (
        f"                  var datasetsByFormat = {datasets_json};\n"
        f"                  var defaultFormatKey = '{default_format_key}';\n"
        f"                  var defaultRangeKey = '{default_range_key}';\n"
        "                  var currentFormatKey = defaultFormatKey;\n"
        "                  var currentRangeKey = defaultRangeKey;\n"
        "                  var formatSelect = document.getElementById('formatFilter');\n"
        "                  var rangeSelect = document.getElementById('rangeFilter');\n"
        "                  var nodeLimitSelect = document.getElementById('nodeLimit');\n"
        """
                  var filterSelect = document.getElementById('archetypeFilter');
                  var resetBtn = document.getElementById('resetFilter');
                  var matrixBtn = document.getElementById('matrixBtn');
                  var tomSelectRef = null;
                  var panelTitle = document.getElementById('panelTitle');
                  var panelSubtitle = document.getElementById('panelSubtitle');
                  var matchupBody = document.getElementById('matchupBody');
                  var thDeck = document.getElementById('thDeck');
                  var thWinrate = document.getElementById('thWinrate');
                  var thMatches = document.getElementById('thMatches');
                  var helpBtn = document.getElementById('helpBtn');
                  var helpOverlay = document.getElementById('helpOverlay');
                  var helpClose = document.getElementById('helpClose');
                  var sidepanel = document.getElementById('sidepanel');
                  var panelWinrate = document.getElementById('panelWinrate');
                  var panelMatches = document.getElementById('panelMatches');

                  var baseNodeState = {};
                  var baseEdgeState = {};
                  var baseSizes = {};
                  var currentFocusId = null;

                  function visibleDataNodes() {
                      return nodes.get().filter(function(n) { return !n.is_label_node && !n.hidden; });
                  }

                  function allLabelNodes() {
                      return nodes.get().filter(function(n) { return n.is_label_node; });
                  }

                  function clearPanelSummary() {
                      if (panelWinrate) panelWinrate.textContent = '—';
                      if (panelMatches) panelMatches.textContent = '—';
                  }

                  function winrateColor(wr) {
                      var x = Math.max(0, Math.min(1, wr));
                      var k = 8.5;
                      var gamma = 0.45;

                      var RED = [255, 0, 0];
                      var YEL = [255, 255, 0];
                      var GRN = [0, 255, 0];

                      var r, g, b, t, u;
                      if (x >= 0.5) {
                          u = (x - 0.5) / 0.5;
                          t = Math.tanh(k * u);
                          t = Math.max(0, Math.min(1, t));
                          t = Math.pow(t, gamma);
                          r = Math.round(YEL[0] + (GRN[0] - YEL[0]) * t);
                          g = Math.round(YEL[1] + (GRN[1] - YEL[1]) * t);
                          b = Math.round(YEL[2] + (GRN[2] - YEL[2]) * t);
                      } else {
                          u = (0.5 - x) / 0.5;
                          t = Math.tanh(k * u);
                          t = Math.max(0, Math.min(1, t));
                          t = Math.pow(t, gamma);
                          r = Math.round(YEL[0] + (RED[0] - YEL[0]) * t);
                          g = Math.round(YEL[1] + (RED[1] - YEL[1]) * t);
                          b = Math.round(YEL[2] + (RED[2] - YEL[2]) * t);
                      }
                      return 'rgb(' + r + ',' + g + ',' + b + ')';
                  }

                  function edgeWidthScale(matches, minM, maxM) {
                      var OUT_MIN = 0.6;
                      var OUT_MAX = 10.0;

                      var m = matches || 0;
                      if (!isFinite(m) || m <= 0) return OUT_MIN;
                      if (!isFinite(minM) || !isFinite(maxM) || maxM <= minM) return (OUT_MIN + OUT_MAX) / 2;

                      var x = (Math.log10(m) - Math.log10(minM)) / (Math.log10(maxM) - Math.log10(minM));
                      x = Math.max(0, Math.min(1, x));
                      return OUT_MIN + x * (OUT_MAX - OUT_MIN);
                  }

                  function applyNodeLimit(limit) {
                      return buildNodeLimitSet(limit, null);
                  }

                  function buildNodeLimitSet(limit, selectedId) {
                      var nodesArray = nodes.get();
                      var dataNodes = nodesArray.filter(function(n) { return !n.is_label_node; });
                      dataNodes.sort(function(a, b) { return (b.matches || 0) - (a.matches || 0); });
                      var keep = {};
                      for (var i = 0; i < dataNodes.length && i < limit; i++) {
                          keep[dataNodes[i].id] = true;
                      }
                      if (selectedId) {
                          keep[selectedId] = true;
                      }
                      return keep;
                  }

                  function showAll() {
                      currentFocusId = null;
                      var nodesArray = nodes.get();
                      var edgesArray = edges.get();
                      var dataNodeIds = {};
                      var basePos = {};
                      var limit = nodeLimitSelect ? parseInt(nodeLimitSelect.value || '20', 10) : 0;
                      var limitKeep = limit ? buildNodeLimitSet(limit, null) : null;
                      for (var i0 = 0; i0 < nodesArray.length; i0++) {
                          var nn0 = nodesArray[i0];
                          if (!nn0.is_label_node) {
                              dataNodeIds[nn0.id] = true;
                          }
                      }
                      for (var i = 0; i < nodesArray.length; i++) {
                          var n0 = nodesArray[i];
                          if (n0.is_label_node) {
                              var keepLabel = dataNodeIds[n0.base_id];
                              if (limitKeep) {
                                  keepLabel = keepLabel && !!limitKeep[n0.base_id];
                              }
                              n0.hidden = !keepLabel;
                              continue;
                          }
                          var keepNode = true;
                          if (limitKeep) {
                              keepNode = !!limitKeep[n0.id];
                          }
                          n0.hidden = !keepNode;
                          var original = baseNodeState[n0.id];
                          if (original) {
                              n0.fixed = original.fixed;
                              n0.x = original.x;
                              n0.y = original.y;
                          }
                          if (!n0.hidden) {
                              basePos[n0.id] = { x: n0.x || 0, y: n0.y || 0 };
                          }
                      }
                      for (var i2 = 0; i2 < nodesArray.length; i2++) {
                          var lbl = nodesArray[i2];
                          if (!lbl.is_label_node) continue;
                          var bp = basePos[lbl.base_id];
                          lbl.hidden = !bp;
                          if (bp) {
                              var off = ((baseSizes[lbl.base_id] || 20) * 1.2 + 28);
                              lbl.x = bp.x;
                              lbl.y = bp.y - off;
                              lbl.fixed = true;
                          }
                      }
                      for (var j = 0; j < edgesArray.length; j++) {
                          if (limitKeep) {
                              edgesArray[j].hidden = !(limitKeep[edgesArray[j].from] && limitKeep[edgesArray[j].to]);
                          } else {
                              edgesArray[j].hidden = false;
                          }
                          var ebase = baseEdgeState[edgesArray[j].id];
                          if (ebase) {
                              edgesArray[j].from = ebase.from;
                              edgesArray[j].to = ebase.to;
                              edgesArray[j].color = ebase.color;
                              edgesArray[j].arrows = ebase.arrows;
                              edgesArray[j].width = ebase.width;
                          }
                      }
                      nodes.update(nodesArray);
                      edges.update(edgesArray);
                      network.setOptions({
                          physics: { enabled: false },
                          edges: { smooth: { type: "dynamic" } },
                          interaction: { dragView: true }
                      });
                      network.fit({ animation: false });
                      if (sidepanel) sidepanel.classList.add('hidden');
                      panelTitle.textContent = 'Select an archetype';
                      panelSubtitle.textContent = 'Matchups will be shown (click headers to sort).';
                      matchupBody.innerHTML = '';
                      clearPanelSummary();
                  }

                  function showRelations(nodeId) {
                      currentFocusId = nodeId;
                      var limit = nodeLimitSelect ? parseInt(nodeLimitSelect.value || '20', 10) : 0;
                      var limitKeep = limit ? buildNodeLimitSet(limit, nodeId) : null;
                      var connectedNodes = network.getConnectedNodes(nodeId);
                      var connectedEdges = network.getConnectedEdges(nodeId);
                      var nodesArray = nodes.get();
                      var edgesArray = edges.get();
                      var keepIds = {};
                      keepIds[nodeId] = true;
                      for (var ci = 0; ci < connectedNodes.length; ci++) {
                          var cid = connectedNodes[ci];
                          if (!limitKeep || limitKeep[cid]) {
                              keepIds[cid] = true;
                          }
                      }

                      for (var i = 0; i < nodesArray.length; i++) {
                          var n1 = nodesArray[i];
                          if (n1.is_label_node) {
                              var keepLabel = !!keepIds[n1.base_id];
                              if (limitKeep) {
                                  keepLabel = keepLabel && !!limitKeep[n1.base_id];
                              }
                              n1.hidden = !keepLabel;
                              continue;
                          }
                          var keepNode = !!keepIds[n1.id];
                          if (limitKeep) {
                              keepNode = keepNode && !!limitKeep[n1.id];
                          }
                          n1.hidden = !keepNode;
                      }

                      for (var j = 0; j < edgesArray.length; j++) {
                          var keepEdge = connectedEdges.indexOf(edgesArray[j].id) !== -1;
                          if (limitKeep) {
                              keepEdge = keepEdge && !!limitKeep[edgesArray[j].from] && !!limitKeep[edgesArray[j].to];
                          }
                          edgesArray[j].hidden = !keepEdge;
                      }

                      nodes.update(nodesArray);
                      edges.update(edgesArray);
                      network.setOptions({
                          physics: { enabled: false },
                          edges: { smooth: false },
                          interaction: { dragView: true }
                      });
                      layoutVisibleNodesCentered(nodeId);
                      applyCenterEdgeEncoding(nodeId);
                      network.fit({ animation: false });
                      renderMatchups(nodeId);
                  }

                  function rebuildArchetypeOptions() {
                      var nodeList = visibleDataNodes();
                      nodeList.sort(function(a, b) {
                          return String(a.display_label || a.id).localeCompare(String(b.display_label || b.id));
                      });

                      var options = [{ value: '__all__', text: 'All' }];
                      for (var i = 0; i < nodeList.length; i++) {
                          options.push({ value: nodeList[i].id, text: (nodeList[i].display_label || nodeList[i].id) });
                      }

                      if (tomSelectRef) {
                          tomSelectRef.clear(true);
                          tomSelectRef.clearOptions();
                          tomSelectRef.addOption(options);
                          tomSelectRef.refreshOptions(false);
                          tomSelectRef.setValue('__all__', true);
                          return;
                      }

                      filterSelect.innerHTML = '';
                      for (var j = 0; j < options.length; j++) {
                          var opt = document.createElement('option');
                          opt.value = options[j].value;
                          opt.textContent = options[j].text;
                          filterSelect.appendChild(opt);
                      }

                      if (window.TomSelect) {
                          tomSelectRef = new TomSelect('#archetypeFilter', {
                              allowEmptyOption: true,
                              placeholder: 'All',
                              onChange: function(value) {
                                  if (!value || value === '__all__') {
                                      showAll();
                                      return;
                                  }
                                  showRelations(value);
                              }
                          });
                      }
                  }

                  var rangeOrder = rangeSelect
                      ? Array.from(rangeSelect.options || []).map(function(opt) { return opt.value; })
                      : [];

                  function getFormatEntry(formatKey) {
                      return datasetsByFormat[formatKey] || datasetsByFormat[defaultFormatKey];
                  }

                  function orderedRangeKeys(rangeKeys) {
                      var seen = {};
                      var ordered = [];
                      for (var i = 0; i < rangeOrder.length; i++) {
                          var rk = rangeOrder[i];
                          if (rangeKeys.indexOf(rk) !== -1 && !seen[rk]) {
                              ordered.push(rk);
                              seen[rk] = true;
                          }
                      }
                      for (var j = 0; j < rangeKeys.length; j++) {
                          var rk2 = rangeKeys[j];
                          if (!seen[rk2]) {
                              ordered.push(rk2);
                              seen[rk2] = true;
                          }
                      }
                      return ordered;
                  }

                  function chooseRangeKey(formatKey, preferredRangeKey) {
                      var entry = getFormatEntry(formatKey);
                      var ranges = (entry && entry.ranges) ? entry.ranges : {};
                      if (ranges[preferredRangeKey]) {
                          return preferredRangeKey;
                      }
                      if (ranges[defaultRangeKey]) {
                          return defaultRangeKey;
                      }
                      var keys = Object.keys(ranges);
                      return keys.length > 0 ? keys[0] : preferredRangeKey;
                  }

                  function rebuildRangeOptions(formatKey, preferredRangeKey) {
                      if (!rangeSelect) {
                          return preferredRangeKey;
                      }
                      var entry = getFormatEntry(formatKey);
                      var ranges = (entry && entry.ranges) ? entry.ranges : {};
                      var keys = orderedRangeKeys(Object.keys(ranges));
                      var selectedKey = chooseRangeKey(formatKey, preferredRangeKey);

                      rangeSelect.innerHTML = '';
                      for (var i = 0; i < keys.length; i++) {
                          var k = keys[i];
                          var opt = document.createElement('option');
                          opt.value = k;
                          opt.textContent = (ranges[k] && ranges[k].label) ? ranges[k].label : k;
                          if (k === selectedKey) {
                              opt.selected = true;
                          }
                          rangeSelect.appendChild(opt);
                      }

                      if (rangeSelect.options.length === 0) {
                          var fallback = document.createElement('option');
                          fallback.value = selectedKey;
                          fallback.textContent = selectedKey;
                          fallback.selected = true;
                          rangeSelect.appendChild(fallback);
                      }

                      rangeSelect.value = selectedKey;
                      return selectedKey;
                  }

                  function loadDataset(formatKey, rangeKey) {
                      var entry = getFormatEntry(formatKey);
                      if (!entry) {
                          return;
                      }
                      var selectedRangeKey = chooseRangeKey(formatKey, rangeKey);
                      var dataset = (entry.ranges && entry.ranges[selectedRangeKey]) ? entry.ranges[selectedRangeKey] : null;
                      if (!dataset) {
                          return;
                      }

                      currentFormatKey = entry.key || formatKey;
                      currentRangeKey = dataset.key || selectedRangeKey;

                      nodes = new vis.DataSet(dataset.nodes || []);
                      edges = new vis.DataSet(dataset.edges || []);
                      network.setData({ nodes: nodes, edges: edges });

                      initBaseState();
                      if (nodeLimitSelect) {
                          applyNodeLimit(parseInt(nodeLimitSelect.value || '20', 10));
                      }
                      rebuildArchetypeOptions();
                      showAll();
                      clearPanelSummary();

                      if (formatSelect) {
                          formatSelect.value = currentFormatKey;
                      }
                      if (rangeSelect) {
                          rangeSelect.value = currentRangeKey;
                      }
                  }

                  if (formatSelect) {
                      formatSelect.addEventListener('change', function() {
                          var fmt = formatSelect.value || defaultFormatKey;
                          var rng = rebuildRangeOptions(fmt, defaultRangeKey);
                          loadDataset(fmt, rng);
                      });
                  }

                  if (rangeSelect) {
                      rangeSelect.addEventListener('change', function() {
                          loadDataset(currentFormatKey, rangeSelect.value);
                      });
                  }

                  if (nodeLimitSelect) {
                      nodeLimitSelect.addEventListener('change', function() {
                          var selected = currentFocusId || filterSelect.value;
                          if (!selected || selected === '__all__') {
                              showAll();
                          } else {
                              showRelations(selected);
                          }
                          rebuildArchetypeOptions();
                      });
                  }

                  filterSelect.addEventListener('change', function() {
                      var value = filterSelect.value;
                      if (!value || value === '__all__') {
                          showAll();
                          return;
                      }
                      showRelations(value);
                  });

                  resetBtn.addEventListener('click', function() {
                      if (tomSelectRef) {
                          tomSelectRef.setValue('__all__', true);
                      } else {
                          filterSelect.value = '__all__';
                      }
                      showAll();
                  });

                  function clamp01(x) {
                      return Math.max(0, Math.min(1, x));
                  }

                  function rgbTextColor(rgbStr) {
                      var m = /rgb\\((\\d+),(\\d+),(\\d+)\\)/.exec(rgbStr || '');
                      if (!m) return '#0b1320';
                      var r = Number(m[1]) / 255;
                      var g = Number(m[2]) / 255;
                      var b = Number(m[3]) / 255;
                      var luminance = 0.2126 * r + 0.7152 * g + 0.0722 * b;
                      return luminance > 0.55 ? '#0b1320' : '#f8fafc';
                  }

                  function matchupFor(rowId, colId) {
                      if (rowId === colId) {
                          return { wr: 0.5, matches: 0, diag: true };
                      }
                      var edgeAB = edges.get({
                          filter: function(e) { return e.from === rowId && e.to === colId; }
                      })[0];
                      var edgeBA = edges.get({
                          filter: function(e) { return e.from === colId && e.to === rowId; }
                      })[0];
                      var edge = edgeAB || edgeBA;
                      if (!edge) {
                          return null;
                      }
                      var wrFrom = (edge.winrate_from !== undefined && edge.winrate_from !== null)
                          ? Number(edge.winrate_from)
                          : 0.5;
                      var wrRow = edgeAB ? wrFrom : (1.0 - wrFrom);
                      wrRow = clamp01(wrRow);
                      return {
                          wr: wrRow,
                          matches: Number(edge.matches || 0),
                          diag: false
                      };
                  }

                  function openMatrixView() {
                      var dataNodes = visibleDataNodes();
                      if (!dataNodes.length) {
                          return;
                      }
                      dataNodes.sort(function(a, b) { return (b.matches || 0) - (a.matches || 0); });
                      var ids = dataNodes.map(function(n) { return n.id; });
                      var labels = {};
                      var overallWr = {};
                      var overallMatches = {};
                      for (var i = 0; i < dataNodes.length; i++) {
                          var n = dataNodes[i];
                          labels[n.id] = n.display_label || n.id;
                          overallWr[n.id] = (n.overall_winrate !== undefined && n.overall_winrate !== null)
                              ? Number(n.overall_winrate)
                              : null;
                          overallMatches[n.id] = Number(n.matches || 0);
                      }

                      var win = window.open('', '_blank');
                      if (!win) {
                          return;
                      }

                      var fmtLabel = (datasetsByFormat[currentFormatKey] && datasetsByFormat[currentFormatKey].label)
                          ? datasetsByFormat[currentFormatKey].label
                          : currentFormatKey;
                      var rangeEntry = datasetsByFormat[currentFormatKey] && datasetsByFormat[currentFormatKey].ranges
                          ? datasetsByFormat[currentFormatKey].ranges[currentRangeKey]
                          : null;
                      var rangeLabel = rangeEntry && rangeEntry.label ? rangeEntry.label : currentRangeKey;

                      var htmlParts = [];
                      htmlParts.push('<!doctype html><html><head><meta charset="utf-8" />');
                      htmlParts.push('<title>MTG Winrate Matrix</title>');
                      htmlParts.push('<style>');
                      htmlParts.push("@import url('https://fonts.googleapis.com/css2?family=Sora:wght@400;600;700&display=swap');");
                      htmlParts.push('body{background:#0f1216;color:#e6e8ec;font-family:"Sora",sans-serif;margin:0;padding:12px;}');
                      htmlParts.push('h2{margin:0 0 8px 0;font-size:18px;}');
                      htmlParts.push('.sub{color:#9ca3af;font-size:12px;margin-bottom:10px;}');
                      htmlParts.push('table{border-collapse:collapse;width:100%;table-layout:fixed;font-size:11px;}');
                      htmlParts.push('th,td{border:1px solid #1f2937;padding:4px;vertical-align:top;text-align:center;}');
                      htmlParts.push('th{background:#111827;color:#f3f4f6;position:sticky;top:0;z-index:2;}');
                      htmlParts.push('.rowhdr{position:sticky;left:0;z-index:1;background:#0f172a;text-align:left;font-weight:700;}');
                      htmlParts.push('.cell{min-height:52px;display:flex;flex-direction:column;gap:2px;align-items:center;justify-content:center;}');
                      htmlParts.push('.cell .wr{font-weight:800;font-size:12px;}');
                      htmlParts.push('.cell .matches{font-size:10px;opacity:.9;}');
                      htmlParts.push('.overall{background:#0f172a;}');
                      htmlParts.push('</style></head><body>');
                      htmlParts.push('<h2>Winrate Matrix</h2>');
                      htmlParts.push('<div class="sub">Format: ' + fmtLabel + ' | Range: ' + rangeLabel + '</div>');
                      htmlParts.push('<div style="overflow:auto;max-height:85vh;border:1px solid #1f2937;">');
                      htmlParts.push('<table><thead><tr>');
                      htmlParts.push('<th class="rowhdr">Deck</th>');
                      htmlParts.push('<th>Overall</th>');
                      for (var c = 0; c < ids.length; c++) {
                          htmlParts.push('<th>' + labels[ids[c]] + '</th>');
                      }
                      htmlParts.push('</tr></thead><tbody>');

                      for (var r = 0; r < ids.length; r++) {
                          var rid = ids[r];
                          htmlParts.push('<tr>');
                          htmlParts.push('<td class="rowhdr">' + labels[rid] + '</td>');

                          var owr = overallWr[rid];
                          var oMatches = overallMatches[rid];
                          var oBg = owr !== null ? winrateColor(owr) : 'rgb(20,20,20)';
                          var oFg = rgbTextColor(oBg);
                          htmlParts.push('<td class="overall" style="background:' + oBg + ';color:' + oFg + ';">');
                          htmlParts.push('<div class="cell">');
                          htmlParts.push('<div class="wr">' + (owr !== null ? (owr * 100).toFixed(1) + '%' : 'n/a') + '</div>');
                          htmlParts.push('<div class="matches">' + (oMatches || 0).toLocaleString() + ' matches</div>');
                          htmlParts.push('</div></td>');

                          for (var c2 = 0; c2 < ids.length; c2++) {
                              var cid = ids[c2];
                              var m = matchupFor(rid, cid);
                              if (!m) {
                                  htmlParts.push('<td></td>');
                                  continue;
                              }
                              var isDiag = rid === cid;
                              var bg = isDiag ? '#7dd3fc' : winrateColor(m.wr);
                              var fg = isDiag ? '#0b1320' : rgbTextColor(bg);
                              htmlParts.push('<td style="background:' + bg + ';color:' + fg + ';">');
                              htmlParts.push('<div class="cell">');
                              htmlParts.push('<div class="wr">' + (isDiag ? '—' : (m.wr * 100).toFixed(1) + '%') + '</div>');
                              htmlParts.push('<div class="matches">' + (m.matches || 0).toLocaleString() + ' matches</div>');
                              htmlParts.push('</div></td>');
                          }

                          htmlParts.push('</tr>');
                      }

                      htmlParts.push('</tbody></table></div></body></html>');
                      win.document.open();
                      win.document.write(htmlParts.join(''));
                      win.document.close();
                  }

                  if (matrixBtn) {
                      matrixBtn.addEventListener('click', openMatrixView);
                  }

                  network.on('selectNode', function(params) {
                      if (params.nodes && params.nodes.length > 0) {
                          var nodeId = params.nodes[0];
                          var nodeObj = nodes.get(nodeId) || {};
                          if (nodeObj.is_label_node && nodeObj.base_id) {
                              nodeId = nodeObj.base_id;
                          }
                          if (tomSelectRef) {
                              tomSelectRef.setValue(nodeId, true);
                          } else {
                              filterSelect.value = nodeId;
                          }
                          showRelations(nodeId);
                      }
                  });

                  network.on('deselectNode', function() {
                      if (filterSelect.value === '__all__') {
                          showAll();
                      }
                  });

                  function initBaseState() {
                      baseNodeState = {};
                      baseEdgeState = {};
                      baseSizes = {};
                      var nodeList = nodes.get();
                      for (var i = 0; i < nodeList.length; i++) {
                          var nn = nodeList[i];
                          baseNodeState[nn.id] = {
                              x: nn.x,
                              y: nn.y,
                              fixed: nn.fixed
                          };
                          if (!nn.is_label_node) {
                              baseSizes[nn.id] = nn.size || 10;
                          }
                      }
                      var edgeList = edges.get();
                      for (var j = 0; j < edgeList.length; j++) {
                          baseEdgeState[edgeList[j].id] = {
                              from: edgeList[j].from,
                              to: edgeList[j].to,
                              color: edgeList[j].color,
                              arrows: edgeList[j].arrows,
                              width: edgeList[j].width
                          };
                      }
                  }

                  function layoutVisibleNodesCentered(centerId) {
                      var nodeList = nodes.get();
                      var visible = [];
                      for (var i = 0; i < nodeList.length; i++) {
                          var nn = nodeList[i];
                          if (!nn.hidden && !nn.is_label_node) {
                              visible.push(nn);
                          }
                      }
                      if (visible.length === 0) {
                          return;
                      }

                      var center = null;
                      var others = [];
                      for (var j = 0; j < visible.length; j++) {
                          if (visible[j].id === centerId) {
                              center = visible[j];
                          } else {
                              others.push(visible[j]);
                          }
                      }
                      if (!center) {
                          return;
                      }

                      others.sort(function(a, b) {
                          return (b.matches || 0) - (a.matches || 0);
                      });

                      var total = others.length;
                      var ringCount = 1;
                      var rings = [others];
                      var baseRadius = 760;
                      var step = 0;
                      var updates = [];
                      var labelUpdates = [];
                      updates.push({ id: center.id, x: 0, y: 0, fixed: true });
                      labelUpdates.push({
                          id: 'label::' + center.id,
                          x: 0,
                          y: -((baseSizes[center.id] || 20) * 1.2 + 28),
                          fixed: true,
                          hidden: false
                      });

                      for (var ri = 0; ri < rings.length; ri++) {
                          var ringNodes = rings[ri];
                          if (ringNodes.length === 0) continue;
                          var radius = baseRadius + ri * step;
                          var angleOffset = (ri * Math.PI) / Math.max(1, ringCount);
                          for (var idx = 0; idx < ringNodes.length; idx++) {
                              var angle = angleOffset + (2.0 * Math.PI * idx / ringNodes.length);
                              var nx = Math.cos(angle) * radius;
                              var ny = Math.sin(angle) * radius;
                              var nid = ringNodes[idx].id;
                              updates.push({
                                  id: nid,
                                  x: nx,
                                  y: ny,
                                  fixed: true
                              });
                              labelUpdates.push({
                                  id: 'label::' + nid,
                                  x: nx,
                                  y: ny - ((baseSizes[nid] || 20) * 1.2 + 28),
                                  fixed: true,
                                  hidden: false
                              });
                          }
                      }
                      nodes.update(updates);
                      nodes.update(labelUpdates);
                  }

                  function applyCenterEdgeEncoding(centerId) {
                      var edgesArray = edges.get();

                      var minM = Infinity;
                      var maxM = -Infinity;
                      for (var i = 0; i < edgesArray.length; i++) {
                          var e0 = edgesArray[i];
                          if (e0.hidden) continue;
                          if (e0.from !== centerId && e0.to !== centerId) continue;

                          var mm = (e0.matches !== undefined && e0.matches !== null) ? Number(e0.matches) : 0;
                          if (isFinite(mm) && mm > 0) {
                              if (mm < minM) minM = mm;
                              if (mm > maxM) maxM = mm;
                          }
                      }
                      if (minM === Infinity) { minM = 0; maxM = 1; }

                      for (var j = 0; j < edgesArray.length; j++) {
                          var e = edgesArray[j];
                          if (e.hidden) continue;
                          if (e.from !== centerId && e.to !== centerId) continue;

                          var mm2 = (e.matches !== undefined && e.matches !== null) ? Number(e.matches) : 0;
                          e.width = edgeWidthScale(mm2, minM, maxM);

                          var winrateFrom = (e.winrate_from !== undefined && e.winrate_from !== null)
                              ? e.winrate_from
                              : 0.5;
                          var win = 0.5;
                          if (e.neutral) {
                              win = 0.5;
                          } else if (e.from === centerId) {
                              win = winrateFrom;
                          } else if (e.to === centerId) {
                              win = 1.0 - winrateFrom;
                          }

                          var color = winrateColor(win);
                          if (Math.abs(win - 0.5) <= 0.005) {
                              var otherNeutral = (e.from === centerId) ? e.to : e.from;
                              e.from = centerId;
                              e.to = otherNeutral;
                              e.arrows = "";
                          } else if (win > 0.5) {
                              var otherOut = (e.from === centerId) ? e.to : e.from;
                              e.from = centerId;
                              e.to = otherOut;
                              e.arrows = "to";
                          } else {
                              var otherIn = (e.from === centerId) ? e.to : e.from;
                              e.from = otherIn;
                              e.to = centerId;
                              e.arrows = "to";
                          }

                          e.color = color;
                          e.wr_center = win;
                      }

                      edges.update(edgesArray);
                  }

                  function renderMatchups(nodeId) {
                      var nodeObj = nodes.get(nodeId) || {};
                      var nodeLabel = nodeObj.display_label || nodeObj.id || nodeId;
                      if (sidepanel) sidepanel.classList.remove('hidden');
                      panelTitle.textContent = nodeLabel;
                      updateSortSubtitle();
                      matchupBody.innerHTML = '';

                      var connectedEdges = network.getConnectedEdges(nodeId);
                      var rows = [];
                      for (var i = 0; i < connectedEdges.length; i++) {
                          var e = edges.get(connectedEdges[i]);
                          if (!e || e.hidden) {
                              continue;
                          }
                          var opponent = e.from === nodeId ? e.to : e.from;
                          var winrate = (e.wr_center !== undefined) ? e.wr_center : 0.5;
                          if (e.wr_center === undefined) {
                              if (e.neutral) {
                                  winrate = 0.5;
                              } else if (e.from === nodeId) {
                                  winrate = e.winrate;
                              } else {
                                  winrate = 1.0 - e.winrate;
                              }
                          }
                          rows.push({
                              deck: opponent,
                              opponent: opponent,
                              winrate: winrate,
                              matches: e.matches || 0
                          });
                      }

                      var sortKey = window.__tableSortKey || 'matches';
                      var sortDir = window.__tableSortDir || 'desc';
                      rows.sort(function(a, b) {
                          var av = a[sortKey];
                          var bv = b[sortKey];
                          if (sortKey === 'deck') {
                              av = String(av).toLowerCase();
                              bv = String(bv).toLowerCase();
                              if (av < bv) return sortDir === 'asc' ? -1 : 1;
                              if (av > bv) return sortDir === 'asc' ? 1 : -1;
                              return 0;
                          }
                          return sortDir === 'asc' ? (av - bv) : (bv - av);
                      });

                      var ow = (nodeObj.overall_winrate !== undefined && nodeObj.overall_winrate !== null)
                          ? Number(nodeObj.overall_winrate)
                          : null;
                      var om = (nodeObj.matches !== undefined && nodeObj.matches !== null)
                          ? Number(nodeObj.matches)
                          : 0;
                      if (panelWinrate) {
                          panelWinrate.textContent = (ow !== null && isFinite(ow))
                              ? (ow * 100).toFixed(1) + '%'
                              : 'n/a';
                          panelWinrate.style.color = (ow !== null && isFinite(ow))
                              ? winrateColor(ow)
                              : '#e5e7eb';
                      }
                      if (panelMatches) {
                          panelMatches.textContent = (om || 0).toLocaleString();
                      }

                      for (var k = 0; k < rows.length; k++) {
                          var r = rows[k];
                          var tr = document.createElement('tr');
                          var tdOpponent = document.createElement('td');
                          var tdWr = document.createElement('td');
                          var tdM = document.createElement('td');
                          tdOpponent.textContent = r.opponent;
                          tdWr.textContent = (r.winrate * 100).toFixed(1) + '%';
                          tdM.textContent = r.matches.toLocaleString();
                          tdWr.style.color = winrateColor(r.winrate);
                          tr.appendChild(tdOpponent);
                          tr.appendChild(tdWr);
                          tr.appendChild(tdM);
                          matchupBody.appendChild(tr);
                      }
                  }

                  function setSort(key) {
                      var currentKey = window.__tableSortKey || 'matches';
                      var currentDir = window.__tableSortDir || 'desc';
                      if (key === currentKey) {
                          window.__tableSortDir = currentDir === 'asc' ? 'desc' : 'asc';
                      } else {
                          window.__tableSortKey = key;
                          window.__tableSortDir = (key === 'deck') ? 'asc' : 'desc';
                      }
                      updateSortSubtitle();
                      var selected = filterSelect.value;
                      if (selected && selected !== '__all__') {
                          renderMatchups(selected);
                      }
                  }

                  if (thDeck && thWinrate && thMatches) {
                      thDeck.style.cursor = 'pointer';
                      thWinrate.style.cursor = 'pointer';
                      thMatches.style.cursor = 'pointer';
                      thDeck.addEventListener('click', function() { setSort('deck'); });
                      thWinrate.addEventListener('click', function() { setSort('winrate'); });
                      thMatches.addEventListener('click', function() { setSort('matches'); });
                  }

                  if (helpBtn && helpOverlay && helpClose) {
                      helpBtn.addEventListener('click', function() {
                          helpOverlay.style.display = 'flex';
                      });
                      helpClose.addEventListener('click', function() {
                          helpOverlay.style.display = 'none';
                      });
                      helpOverlay.addEventListener('click', function(ev) {
                          if (ev.target === helpOverlay) {
                              helpOverlay.style.display = 'none';
                          }
                      });
                  }

                  function updateSortSubtitle() {
                      var key = window.__tableSortKey || 'matches';
                      var dir = window.__tableSortDir || 'desc';
                      var label = key === 'deck' ? 'deck' : (key === 'winrate' ? 'winrate' : 'matches');
                      panelSubtitle.textContent = 'Sorted by ' + label + ' (' + dir + ')';
                  }

                  if (formatSelect) {
                      formatSelect.value = defaultFormatKey;
                  }
                  var initialRangeKey = rebuildRangeOptions(defaultFormatKey, defaultRangeKey);
                  loadDataset(defaultFormatKey, initialRangeKey);

                  function applyArrowScale() {
                      var scale = network.getScale();
                      var factor = Math.max(0.3, Math.min(1.2, 1 / Math.pow(scale, 0.7)));
                      network.setOptions({
                          edges: {
                              arrows: { to: { enabled: true, scaleFactor: factor } }
                          }
                      });
                  }

                  network.on('zoom', function() {
                      applyArrowScale();
                  });

                  applyArrowScale();
    """)

    html = html.replace("</head>", f"{extra_head}\n</head>")
    html = html.replace("</style>", f"{extra_css}\n        </style>")
    html = html.replace(
        "<div id=\"mynetwork\" class=\"card-body\"></div>",
        f"{controls_html}\n{sidepanel_html}\n{footer_html}",
    )
    html = html.replace("network = new vis.Network(container, data, options);", "network = new vis.Network(container, data, options);\n" + extra_js)

    html_path.write_text(html, encoding="utf-8")


def inline_assets(out_html: str) -> None:
    html_path = Path(out_html)
    html = html_path.read_text(encoding="utf-8")

    def read_text(path: str) -> str:
        return Path(path).read_text(encoding="utf-8")

    replacements = [
        ("<script src=\"lib/bindings/utils.js\"></script>", "<script>\n" + read_text("lib/bindings/utils.js") + "\n</script>"),
        ("<link rel=\"stylesheet\" href=\"lib/vis-9.1.2/vis-network.css\" />", "<style>\n" + read_text("lib/vis-9.1.2/vis-network.css") + "\n</style>"),
        ("<script src=\"lib/vis-9.1.2/vis-network.min.js\"></script>", "<script>\n" + read_text("lib/vis-9.1.2/vis-network.min.js") + "\n</script>"),
        ("<link rel=\"stylesheet\" href=\"lib/tom-select/tom-select.css\" />", "<style>\n" + read_text("lib/tom-select/tom-select.css") + "\n</style>"),
        ("<script src=\"lib/tom-select/tom-select.complete.min.js\"></script>", "<script>\n" + read_text("lib/tom-select/tom-select.complete.min.js") + "\n</script>"),
        ("<link rel=\"stylesheet\" href=\"https://cdnjs.cloudflare.com/ajax/libs/vis-network/9.1.2/dist/dist/vis-network.min.css\" integrity=\"sha512-WgxfT5LWjfszlPHXRmBWHkV2eceiWTOBvrKCNbdgDYTHrT2AeLCGbF4sZlZw3UMN3WtL0tGUoIAKsu8mllg/XA==\" crossorigin=\"anonymous\" referrerpolicy=\"no-referrer\" />", "<style>\n" + read_text("lib/vis-9.1.2/vis-network.css") + "\n</style>"),
        ("<script src=\"https://cdnjs.cloudflare.com/ajax/libs/vis-network/9.1.2/dist/vis-network.min.js\" integrity=\"sha512-LnvoEWDFrqGHlHmDD2101OrLcbsfkrzoSpvtSQtxK3RMnRV0eOkhhBN2dXHKRrUU8p2DGRTk35n4O8nWSVe1mQ==\" crossorigin=\"anonymous\" referrerpolicy=\"no-referrer\"></script>", "<script>\n" + read_text("lib/vis-9.1.2/vis-network.min.js") + "\n</script>"),
    ]

    for old, new in replacements:
        if old in html:
            html = html.replace(old, new)

    # Remove external bootstrap references to make the HTML standalone
    bootstrap_links = [
        "<link\n          href=\"https://cdn.jsdelivr.net/npm/bootstrap@5.0.0-beta3/dist/css/bootstrap.min.css\"\n          rel=\"stylesheet\"\n          integrity=\"sha384-eOJMYsd53ii+scO/bJGFsiCZc+5NDVN2yr8+0RDqr0Ql0h+rP48ckxlpbzKgwra6\"\n          crossorigin=\"anonymous\"\n        />",
        "<script\n          src=\"https://cdn.jsdelivr.net/npm/bootstrap@5.0.0-beta3/dist/js/bootstrap.bundle.min.js\"\n          integrity=\"sha384-JEW9xMcG8R+pH31jmWH6WWP0WintQrMb4s7ZOdauHnUtxwoG2vI5DkLtS3qm9Ekf\"\n          crossorigin=\"anonymous\"\n        ></script>",
    ]
    for tag in bootstrap_links:
        html = html.replace(tag, "")

    html_path.write_text(html, encoding="utf-8")
