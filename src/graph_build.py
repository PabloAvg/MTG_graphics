from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, Tuple, List

import pandas as pd
import networkx as nx

from config import (
    EDGE_WIDTH_MAX,
    EDGE_WIDTH_MIN,
    EPS_TIE,
    HIDE_ISOLATED_NODES,
    MIN_MATCHES_EDGE,
    NODE_SIZE_MAX,
    NODE_SIZE_MIN,
    RING_COUNT,
    RING_RADIUS_BASE,
    RING_RADIUS_STEP,
    TOP_N_ARCHETYPES,
)


def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def scale_log(value: float, vmin: float, vmax: float, out_min: float, out_max: float) -> float:
    if value <= 0:
        return out_min
    value = clamp(value, vmin, vmax)
    a = math.log10(vmin) if vmin > 0 else 0.0
    b = math.log10(vmax) if vmax > 0 else 1.0
    x = (math.log10(value) - a) / (b - a) if b != a else 0.0
    return out_min + x * (out_max - out_min)


def scale_sqrt(value: float, vmin: float, vmax: float, out_min: float, out_max: float) -> float:
    if value <= 0:
        return out_min
    value = clamp(value, vmin, vmax)
    a = math.sqrt(vmin) if vmin > 0 else 0.0
    b = math.sqrt(vmax) if vmax > 0 else 1.0
    x = (math.sqrt(value) - a) / (b - a) if b != a else 0.0
    return out_min + x * (out_max - out_min)


def scale_power(
    value: float,
    vmin: float,
    vmax: float,
    out_min: float,
    out_max: float,
    power: float = 2.8,
) -> float:
    """
    Aggressive scaling: values near vmin stay very thin; values near vmax grow very thick.
    power > 1 makes it more drastic (try 2.0–4.0).
    """
    if value <= 0:
        return out_min
    value = clamp(value, vmin, vmax)
    x = (value - vmin) / (vmax - vmin) if vmax != vmin else 0.0  # 0..1
    x = x ** power
    return out_min + x * (out_max - out_min)


def _hex_to_rgb(h: str) -> Tuple[int, int, int]:
    h = h.lstrip("#")
    return int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)


def _rgb_to_hex(rgb: Tuple[int, int, int]) -> str:
    r, g, b = rgb
    return f"#{r:02x}{g:02x}{b:02x}"


def _lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * t


def _lerp_rgb(c1: Tuple[int, int, int], c2: Tuple[int, int, int], t: float) -> Tuple[int, int, int]:
    return (
        int(round(_lerp(c1[0], c2[0], t))),
        int(round(_lerp(c1[1], c2[1], t))),
        int(round(_lerp(c1[2], c2[2], t))),
    )


def winrate_to_color(wr: float, k: float = 5.5, gamma: float = 0.55) -> str:
    """
    Aggressive diverging scale (red <-> yellow <-> green) for dense mid-range data.
    """
    wr = clamp(wr, 0.0, 1.0)

    red = _hex_to_rgb("#FF0000")
    yel = _hex_to_rgb("#FFFF00")
    grn = _hex_to_rgb("#00FF00")

    if wr >= 0.5:
        u = (wr - 0.5) / 0.5  # 0..1
        t = math.tanh(k * u)
        t = clamp(t, 0.0, 1.0)
        t = t ** gamma
        rgb = _lerp_rgb(yel, grn, t)
    else:
        u = (0.5 - wr) / 0.5  # 0..1
        t = math.tanh(k * u)
        t = clamp(t, 0.0, 1.0)
        t = t ** gamma
        rgb = _lerp_rgb(yel, red, t)

    return _rgb_to_hex(rgb)


def edge_winrate_color(wr: float) -> str:
    return winrate_to_color(wr)


def node_winrate_color(wr: float) -> str:
    """
    wr in 0..1. Same scale as edges/table: 0 red, 0.5 yellow, 1 green.
    """
    return winrate_to_color(wr)


def compute_radial_positions(archetypes_df: pd.DataFrame) -> Dict[str, Tuple[float, float]]:
    """
    Rings by overall_matches (log scale). Bigger decks go to outer rings for separation.
    """
    ordered = archetypes_df.sort_values("overall_matches", ascending=False).copy()
    names = ordered["archetype"].tolist()
    matches = ordered["overall_matches"].astype(float).tolist()

    if not names:
        return {}

    vmin = max(1.0, min(matches))
    vmax = max(vmin + 1.0, max(matches))

    def norm_log(m: float) -> float:
        a = math.log10(vmin)
        b = math.log10(vmax)
        if b == a:
            return 0.0
        return (math.log10(max(m, vmin)) - a) / (b - a)

    ring_groups: Dict[int, List[str]] = {i: [] for i in range(RING_COUNT)}
    for name, m in zip(names, matches):
        t = norm_log(m)
        ring_idx = int(round(t * (RING_COUNT - 1)))
        ring_idx = int(clamp(ring_idx, 0, RING_COUNT - 1))
        ring_groups[ring_idx].append(name)

    positions: Dict[str, Tuple[float, float]] = {}
    for ring_idx in range(RING_COUNT):
        ring_nodes = ring_groups.get(ring_idx, [])
        if not ring_nodes:
            continue
        radius = RING_RADIUS_BASE + ring_idx * RING_RADIUS_STEP
        count = len(ring_nodes)
        angle_offset = (ring_idx * math.pi) / max(1, RING_COUNT)
        for i, node in enumerate(ring_nodes):
            angle = angle_offset + (2.0 * math.pi * i / count)
            x = math.cos(angle) * radius
            y = math.sin(angle) * radius
            positions[node] = (x, y)

    return positions


def build_graph(archetypes_df: pd.DataFrame, matchups_df: pd.DataFrame) -> nx.DiGraph:
    arch_sorted = archetypes_df.sort_values("overall_matches", ascending=False).copy()
    if TOP_N_ARCHETYPES and TOP_N_ARCHETYPES > 0:
        arch_sorted = arch_sorted.head(TOP_N_ARCHETYPES).copy()

    G = nx.DiGraph()

    vmin = max(1, int(arch_sorted["overall_matches"].min()))
    vmax = max(vmin + 1, int(arch_sorted["overall_matches"].max()))

    for _, r in arch_sorted.iterrows():
        ms = int(r["overall_matches"])
        size = scale_sqrt(ms, vmin, vmax, NODE_SIZE_MIN, NODE_SIZE_MAX)

        url = r.get("url_relative")
        full_url = (
            f"https://mtgdecks.net{url}" if isinstance(url, str) and url.startswith("/") else None
        )

        ow = r["overall_winrate"]
        title = (
            f"{r['archetype']}\nOverall winrate: {ow:.3f}\nMatches: {ms:,}"
            if pd.notna(ow)
            else f"{r['archetype']}\nMatches: {ms:,}"
        )

        G.add_node(
            r["archetype"],
            size=float(size),
            matches=ms,
            overall_winrate=float(ow) if pd.notna(ow) else None,
            url=full_url,
            title=title,
        )

    top_names = set(arch_sorted["archetype"].tolist())
    m = matchups_df[
        (matchups_df["matches"] >= MIN_MATCHES_EDGE)
        & (matchups_df["a"].isin(top_names))
        & (matchups_df["b"].isin(top_names))
    ].copy()

    m_min = float(max(MIN_MATCHES_EDGE, int(m["matches"].min()))) if not m.empty else float(MIN_MATCHES_EDGE)
    m_max = float(int(m["matches"].max())) if not m.empty else float(MIN_MATCHES_EDGE)
    if m_max <= m_min:
        m_max = m_min + 1.0

    seen_pairs = set()

    for _, r in m.iterrows():
        a, b = r["a"], r["b"]
        if a == b:
            continue

        pair_key = tuple(sorted([a, b]))
        if pair_key in seen_pairs:
            continue
        seen_pairs.add(pair_key)

        ab = m[(m["a"] == a) & (m["b"] == b)]
        ba = m[(m["a"] == b) & (m["b"] == a)]

        if not ab.empty:
            wr_ab = float(ab.iloc[0]["winrate_a_vs_b"])
            matches = int(ab.iloc[0]["matches"])
            ci_low = ab.iloc[0].get("ci_low")
            ci_high = ab.iloc[0].get("ci_high")
        elif not ba.empty:
            wr_ab = 1.0 - float(ba.iloc[0]["winrate_a_vs_b"])
            matches = int(ba.iloc[0]["matches"])
            ci_low = ba.iloc[0].get("ci_low")
            ci_high = ba.iloc[0].get("ci_high")
        else:
            continue

        width = scale_power(
            matches,
            vmin=m_min,
            vmax=m_max,
            out_min=EDGE_WIDTH_MIN,
            out_max=EDGE_WIDTH_MAX,
            power=3.5,
        )
        neutral = abs(wr_ab - 0.5) <= EPS_TIE

        if neutral:
            label = f"{matches:,}"
            title = f"{a} vs {b}\n~ 50%\nMatches: {matches:,}"
            G.add_edge(
                a,
                b,
                matches=matches,
                winrate=wr_ab,
                winrate_from=0.5,
                neutral=True,
                width=float(width),
                color=edge_winrate_color(0.5),
                label=label,
                title=title,
                arrows="",
            )
        else:
            if wr_ab > 0.5:
                src, dst = a, b
                wr = wr_ab
            else:
                src, dst = b, a
                wr = 1.0 - wr_ab

            color = edge_winrate_color(wr)
            label = f"{matches:,}"
            pct = round(wr * 100, 1)

            ci_txt = ""
            if pd.notna(ci_low) and pd.notna(ci_high):
                ci_txt = f"\nCI: {round(ci_low*100,1)}% - {round(ci_high*100,1)}%"

            title = f"{src} -> {dst}\nWinrate: {pct}%\nMatches: {matches:,}{ci_txt}"

            G.add_edge(
                src,
                dst,
                matches=matches,
                winrate=wr,
                winrate_from=wr,
                neutral=False,
                width=float(width),
                color=color,
                label=label,
                title=title,
                arrows="to",
            )

    if HIDE_ISOLATED_NODES:
        isolated = [n for n in G.nodes() if G.degree(n) == 0]
        G.remove_nodes_from(isolated)

    return G
