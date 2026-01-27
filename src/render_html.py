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

    net = Network(height="800px", width="100%", directed=True, bgcolor="#1f1f1f", font_color="#e6e6e6")
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

             body {
                 background: #1b1b1b;
             }
             .card {
                 background: #1f1f1f;
                 border: 1px solid #2a2a2a;
             }
             #mynetwork {
                 background-color: #1f1f1f;
                 border: 1px solid #2a2a2a;
             }
             /* Hide vis-network navigation buttons (arrows + zoom) */
             div.vis-network div.vis-navigation {
                 display: none !important;
             }
             .graph-controls {
                 display: flex;
                 gap: 12px;
                 align-items: center;
                 padding: 10px 12px;
                 border-bottom: 1px solid #2a2a2a;
                 background: #222222;
                 font-family: Arial, sans-serif;
             }
             .graph-controls label {
                 font-size: 14px;
                 color: #f0f0f0;
                 margin: 0;
             }
             .graph-controls .hint {
                 font-size: 14px;
                 color: #e5e7eb;
                 text-decoration: underline;
                 font-weight: 600;
             }
             .graph-controls .spacer {
                 flex: 1 1 auto;
             }
             .node-summary {
                 min-width: 220px;
                 padding: 6px 10px;
                 border: 1px solid #2f3b4a;
                 background: #1b2530;
                 color: #e6e6e6;
                 font-family: Arial, sans-serif;
                 font-size: 12px;
                 line-height: 1.25;
                 border-radius: 6px;
             }
             .node-summary-title {
                 font-size: 13px;
                 font-weight: 700;
                 margin-bottom: 4px;
                 color: #f3f4f6;
             }
             .node-summary-row {
                 display: flex;
                 justify-content: space-between;
                 gap: 8px;
             }
             .node-summary-row strong {
                 color: #f9fafb;
             }
             .graph-controls .btn-mini {
                 border: 1px solid #3a3a3a;
                 background: #2a2a2a;
                 color: #e6e6e6;
                 padding: 4px 10px;
                 font-size: 12px;
                 cursor: pointer;
             }
             .help-btn {
                 border: 1px solid #38bdf8;
                 background: #7dd3fc;
                 color: #0f172a;
                 padding: 4px 10px;
                 font-size: 12px;
                 font-weight: 600;
                 cursor: pointer;
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
                 background: #1f1f1f;
                 border: 1px solid #2a2a2a;
                 border-radius: 8px;
                 padding: 14px 16px;
                 color: #e6e6e6;
                 font-family: Arial, sans-serif;
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
                 color: #e6e6e6;
                 font-size: 16px;
                 cursor: pointer;
             }
             .graph-layout {
                 display: flex;
                 gap: 12px;
                 align-items: stretch;
             }
             #mynetwork {
                 flex: 1 1 auto;
                 min-width: 0;
             }
             .sidepanel {
                 width: 360px;
                 padding: 10px 12px;
                 border-left: 1px solid #2a2a2a;
                 background: #1f1f1f;
                 font-family: Arial, sans-serif;
             }
             .sidepanel h3 {
                 font-size: 16px;
                 margin: 0 0 6px 0;
                 color: #f5f5f5;
             }
             .sidepanel .sub {
                 font-size: 12px;
                 color: #b0b0b0;
                 margin-bottom: 8px;
             }
             .sidepanel table {
                 width: 100%;
                 border-collapse: collapse;
                 font-size: 12px;
             }
             .sidepanel th, .sidepanel td {
                 border-bottom: 1px solid #2a2a2a;
                 padding: 4px 6px;
                 text-align: left;
                 color: #f0f0f0;
             }
             .sidepanel th {
                 background: #242424;
             }
             .graph-footer {
                 margin-top: 10px;
                 padding: 8px 12px;
                 border-top: 1px solid #2a2a2a;
                 color: #b8c0cc;
                 font-family: Arial, sans-serif;
                 font-size: 12px;
                 display: flex;
                 justify-content: space-between;
                 gap: 12px;
                 flex-wrap: wrap;
             }
             .graph-footer a {
                 color: #93c5fd;
                 text-decoration: none;
             }
             .graph-footer a:hover,
             .graph-footer a:focus {
                 text-decoration: underline;
             }
    """

    controls_html = f"""
            <div class="graph-controls">
                <label for="formatFilter">Format</label>
                <select id="formatFilter">
                    {format_options_html}
                </select>
                <label for="rangeFilter">Time range</label>
                <select id="rangeFilter">
                    {range_options_html}
                </select>
                <label for="archetypeFilter">Archetype filter</label>
                <select id="archetypeFilter" placeholder="All">
                    <option value="__all__">All</option>
                </select>
                <button id="resetFilter" class="btn-mini" type="button">Reset</button>
                <button id="matrixBtn" class="btn-mini" type="button">Matrix</button>
                <button id="helpBtn" class="help-btn" type="button">Help</button>
                <span class="hint">Select an archetype from the filter OR click a node to focus it, zoom to see details</span>
                <div class="spacer"></div>
                <div id="nodeSummary" class="node-summary">
                    <div id="nodeSummaryTitle" class="node-summary-title">No archetype selected</div>
                    <div class="node-summary-row"><span>Winrate</span><strong id="nodeSummaryWinrate">—</strong></div>
                    <div class="node-summary-row"><span>Matches</span><strong id="nodeSummaryMatches">—</strong></div>
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
                <div id="mynetwork" class="card-body"></div>
                <div class="sidepanel">
                    <h3 id="panelTitle">Select an archetype</h3>
                    <div class="sub" id="panelSubtitle">Matchups will be shown (sortable).</div>
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
                  var nodeSummaryTitle = document.getElementById('nodeSummaryTitle');
                  var nodeSummaryWinrate = document.getElementById('nodeSummaryWinrate');
                  var nodeSummaryMatches = document.getElementById('nodeSummaryMatches');

                  var baseNodeState = {};
                  var baseEdgeState = {};
                  var baseSizes = {};

                  function visibleDataNodes() {
                      return nodes.get().filter(function(n) { return !n.is_label_node; });
                  }

                  function allLabelNodes() {
                      return nodes.get().filter(function(n) { return n.is_label_node; });
                  }

                  function clearNodeSummary() {
                      if (nodeSummaryTitle) nodeSummaryTitle.textContent = 'No archetype selected';
                      if (nodeSummaryWinrate) nodeSummaryWinrate.textContent = '—';
                      if (nodeSummaryMatches) nodeSummaryMatches.textContent = '—';
                  }

                  function updateNodeSummary(nodeId) {
                      var node = nodes.get(nodeId);
                      if (!node) {
                          clearNodeSummary();
                          return;
                      }
                      var label = node.display_label || node.id || nodeId;
                      var wr = (node.overall_winrate !== undefined && node.overall_winrate !== null)
                          ? Number(node.overall_winrate)
                          : null;
                      var matches = (node.matches !== undefined && node.matches !== null)
                          ? Number(node.matches)
                          : 0;

                      if (nodeSummaryTitle) nodeSummaryTitle.textContent = label;
                      if (nodeSummaryWinrate) {
                          nodeSummaryWinrate.textContent = (wr !== null && isFinite(wr))
                              ? (wr * 100).toFixed(1) + '%'
                              : 'n/a';
                          nodeSummaryWinrate.style.color = (wr !== null && isFinite(wr))
                              ? winrateColor(wr)
                              : '#e5e7eb';
                      }
                      if (nodeSummaryMatches) {
                          nodeSummaryMatches.textContent = (matches || 0).toLocaleString();
                      }
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

                  function showAll() {
                      var nodesArray = nodes.get();
                      var edgesArray = edges.get();
                      var dataNodeIds = {};
                      var basePos = {};
                      for (var i0 = 0; i0 < nodesArray.length; i0++) {
                          var nn0 = nodesArray[i0];
                          if (!nn0.is_label_node) {
                              dataNodeIds[nn0.id] = true;
                          }
                      }
                      for (var i = 0; i < nodesArray.length; i++) {
                          var n0 = nodesArray[i];
                          if (n0.is_label_node) {
                              n0.hidden = !dataNodeIds[n0.base_id];
                              continue;
                          }
                          n0.hidden = false;
                          var original = baseNodeState[n0.id];
                          if (original) {
                              n0.fixed = original.fixed;
                              n0.x = original.x;
                              n0.y = original.y;
                          }
                          basePos[n0.id] = { x: n0.x || 0, y: n0.y || 0 };
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
                          edgesArray[j].hidden = false;
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
                          edges: { smooth: { type: "dynamic" } }
                      });
                      network.fit({ animation: false });
                      panelTitle.textContent = 'Select an archetype';
                      panelSubtitle.textContent = 'Matchups will be shown (click headers to sort).';
                      matchupBody.innerHTML = '';
                      clearNodeSummary();
                  }

                  function showRelations(nodeId) {
                      var connectedNodes = network.getConnectedNodes(nodeId);
                      var connectedEdges = network.getConnectedEdges(nodeId);
                      var nodesArray = nodes.get();
                      var edgesArray = edges.get();
                      var keepIds = {};
                      keepIds[nodeId] = true;
                      for (var ci = 0; ci < connectedNodes.length; ci++) {
                          keepIds[connectedNodes[ci]] = true;
                      }

                      for (var i = 0; i < nodesArray.length; i++) {
                          var n1 = nodesArray[i];
                          if (n1.is_label_node) {
                              n1.hidden = !keepIds[n1.base_id];
                              continue;
                          }
                          var keepNode = !!keepIds[n1.id];
                          n1.hidden = !keepNode;
                      }

                      for (var j = 0; j < edgesArray.length; j++) {
                          var keepEdge = connectedEdges.indexOf(edgesArray[j].id) !== -1;
                          edgesArray[j].hidden = !keepEdge;
                      }

                      nodes.update(nodesArray);
                      edges.update(edgesArray);
                      network.setOptions({
                          physics: { enabled: false },
                          edges: { smooth: false }
                      });
                      layoutVisibleNodesCentered(nodeId);
                      applyCenterEdgeEncoding(nodeId);
                      network.fit({ animation: false });
                      renderMatchups(nodeId);
                      updateNodeSummary(nodeId);
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
                      rebuildArchetypeOptions();
                      showAll();
                      clearNodeSummary();

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
                      htmlParts.push('body{background:#0b0f14;color:#e5e7eb;font-family:Arial,sans-serif;margin:0;padding:12px;}');
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
