from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Tuple, Any, List

import pandas as pd
import networkx as nx
from pyvis.network import Network

from graph_build import compute_radial_positions, node_winrate_color
from config import EMBED_ASSETS, RANGE_OPTIONS


def _apply_visual_overrides(net: Network, G: nx.DiGraph, archetypes_df: pd.DataFrame) -> None:
    positions = compute_radial_positions(archetypes_df)

    for n in net.nodes:
        node_id = n["id"]
        attrs = G.nodes[node_id]

        n["size"] = float(attrs.get("size", 25))
        n["title"] = attrs.get("title", node_id)
        n["font"] = {"color": "#ffffff", "size": 16, "strokeWidth": 3, "strokeColor": "#101010"}
        n["matches"] = int(attrs.get("matches", 0))

        ow = attrs.get("overall_winrate")
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
    for node_id, attrs in G.nodes(data=True):
        node: Dict[str, Any] = {
            "id": node_id,
            "label": node_id,
            "size": float(attrs.get("size", 25)),
            "title": attrs.get("title", node_id),
            "font": {"color": "#ffffff", "size": 16, "strokeWidth": 3, "strokeColor": "#101010"},
            "matches": int(attrs.get("matches", 0)),
        }

        ow = attrs.get("overall_winrate")
        if isinstance(ow, float):
            node["color"] = node_winrate_color(ow)

        if attrs.get("url"):
            node["url"] = attrs["url"]

        if node_id in positions:
            x, y = positions[node_id]
            node["x"] = x
            node["y"] = y
            node["fixed"] = True

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

    return {"nodes": nodes, "edges": edges}


def render_pyvis(
    graphs: Dict[str, Tuple[nx.DiGraph, pd.DataFrame]],
    out_html: str,
    default_range_key: str,
) -> None:
    if default_range_key not in graphs:
        raise KeyError(f"default_range_key '{default_range_key}' not found in graphs")

    default_G, default_df = graphs[default_range_key]

    net = Network(height="850px", width="100%", directed=True, bgcolor="#1f1f1f", font_color="#e6e6e6")
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

    datasets_by_range: Dict[str, Dict[str, Any]] = {}
    for range_key, (G, df) in graphs.items():
        meta = RANGE_OPTIONS.get(range_key, {})
        datasets_by_range[range_key] = {
            "key": range_key,
            "label": meta.get("label", range_key),
            **_build_dataset(G, df),
        }

    inject_filter_ui(out_html, datasets_by_range=datasets_by_range, default_range_key=default_range_key)
    if EMBED_ASSETS:
        inline_assets(out_html)


def inject_filter_ui(
    out_html: str,
    datasets_by_range: Dict[str, Dict[str, Any]],
    default_range_key: str,
) -> None:
    html_path = Path(out_html)
    html = html_path.read_text(encoding="utf-8")

    if "matchupBody" in html:
        return

    if default_range_key not in datasets_by_range:
        raise KeyError(f"default_range_key '{default_range_key}' not found in datasets_by_range")

    datasets_json = json.dumps(datasets_by_range, ensure_ascii=True)

    range_options_html_parts: List[str] = []
    for key in RANGE_OPTIONS.keys():
        if key not in datasets_by_range:
            continue
        label = datasets_by_range[key].get("label", key)
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
                 font-size: 12px;
                 color: #b0b0b0;
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
                 border: 1px solid #3a3a3a;
                 background: #2a2a2a;
                 color: #e6e6e6;
                 padding: 4px 10px;
                 font-size: 12px;
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
    """

    controls_html = f"""
            <div class="graph-controls">
                <label for="rangeFilter">Time range</label>
                <select id="rangeFilter">
                    {range_options_html}
                </select>
                <label for="archetypeFilter">Archetype filter</label>
                <select id="archetypeFilter" placeholder="All">
                    <option value="__all__">All</option>
                </select>
                <button id="resetFilter" class="btn-mini" type="button">Reset</button>
                <button id="helpBtn" class="help-btn" type="button">Help</button>
                <span class="hint">Click a node to show only its relations and the table</span>
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

    extra_js = (
        f"                  var datasetsByRange = {datasets_json};\n"
        f"                  var defaultRangeKey = '{default_range_key}';\n"
        "                  var currentRangeKey = defaultRangeKey;\n"
        "                  var rangeSelect = document.getElementById('rangeFilter');\n"
        """
                  var filterSelect = document.getElementById('archetypeFilter');
                  var resetBtn = document.getElementById('resetFilter');
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

                  var baseNodeState = {};
                  var baseEdgeState = {};
                  var baseSizes = {};

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
                      for (var i = 0; i < nodesArray.length; i++) {
                          nodesArray[i].hidden = false;
                          var original = baseNodeState[nodesArray[i].id];
                          if (original) {
                              nodesArray[i].fixed = original.fixed;
                              nodesArray[i].x = original.x;
                              nodesArray[i].y = original.y;
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
                  }

                  function showRelations(nodeId) {
                      var connectedNodes = network.getConnectedNodes(nodeId);
                      var connectedEdges = network.getConnectedEdges(nodeId);
                      var nodesArray = nodes.get();
                      var edgesArray = edges.get();

                      for (var i = 0; i < nodesArray.length; i++) {
                          var keepNode = nodesArray[i].id === nodeId || connectedNodes.indexOf(nodesArray[i].id) !== -1;
                          nodesArray[i].hidden = !keepNode;
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
                  }

                  function rebuildArchetypeOptions() {
                      var nodeList = nodes.get();
                      nodeList.sort(function(a, b) {
                          return String(a.label).localeCompare(String(b.label));
                      });

                      var options = [{ value: '__all__', text: 'All' }];
                      for (var i = 0; i < nodeList.length; i++) {
                          options.push({ value: nodeList[i].id, text: nodeList[i].label });
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

                  function loadRange(rangeKey) {
                      var dataset = datasetsByRange[rangeKey] || datasetsByRange[defaultRangeKey];
                      if (!dataset) {
                          return;
                      }

                      currentRangeKey = dataset.key || rangeKey;
                      nodes = new vis.DataSet(dataset.nodes || []);
                      edges = new vis.DataSet(dataset.edges || []);
                      network.setData({ nodes: nodes, edges: edges });

                      initBaseState();
                      rebuildArchetypeOptions();
                      showAll();
                  }

                  if (rangeSelect) {
                      rangeSelect.addEventListener('change', function() {
                          loadRange(rangeSelect.value);
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

                  network.on('selectNode', function(params) {
                      if (params.nodes && params.nodes.length > 0) {
                          var nodeId = params.nodes[0];
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
                          baseNodeState[nodeList[i].id] = {
                              x: nodeList[i].x,
                              y: nodeList[i].y,
                              fixed: nodeList[i].fixed
                          };
                          baseSizes[nodeList[i].id] = nodeList[i].size || 10;
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
                          if (!nodeList[i].hidden) {
                              visible.push(nodeList[i]);
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
                      var baseRadius = 520;
                      var step = 0;
                      var updates = [];
                      updates.push({ id: center.id, x: 0, y: 0, fixed: true });

                      for (var ri = 0; ri < rings.length; ri++) {
                          var ringNodes = rings[ri];
                          if (ringNodes.length === 0) continue;
                          var radius = baseRadius + ri * step;
                          var angleOffset = (ri * Math.PI) / Math.max(1, ringCount);
                          for (var idx = 0; idx < ringNodes.length; idx++) {
                              var angle = angleOffset + (2.0 * Math.PI * idx / ringNodes.length);
                              updates.push({
                                  id: ringNodes[idx].id,
                                  x: Math.cos(angle) * radius,
                                  y: Math.sin(angle) * radius,
                                  fixed: true
                              });
                          }
                      }
                      nodes.update(updates);
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
                      var nodeLabel = nodes.get(nodeId).label || nodeId;
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

                  if (rangeSelect) {
                      rangeSelect.value = defaultRangeKey;
                  }
                  loadRange(defaultRangeKey);

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
    html = html.replace("<div id=\"mynetwork\" class=\"card-body\"></div>", f"{controls_html}\n{sidepanel_html}")
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
