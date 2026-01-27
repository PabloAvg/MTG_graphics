from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, Tuple

import pandas as pd
import networkx as nx

from config import (
    BASE_WINRATES_URL,
    DEFAULT_RANGE_KEY,
    OUT_HTML,
    RANGE_OPTIONS,
)
from graph_build import build_graph
from render_html import render_pyvis
from scrape import build_range_url, fetch_html, parse_page


def inspect_console(archetypes_df, matchups_df, G) -> None:
    print("\n=== DATA ===")
    print(f"Archetypes: {len(archetypes_df)} | Matchups (raw cells): {len(matchups_df)}")

    print("\nTop 10 archetypes by matches:")
    print(
        archetypes_df.sort_values("overall_matches", ascending=False)
        .head(10)[["archetype", "overall_matches", "overall_winrate"]]
        .to_string(index=False)
    )

    print("\n=== GRAPH ===")
    print(f"Nodes: {G.number_of_nodes()} | Edges: {G.number_of_edges()}")


def _build_range_url(path: str) -> str:
    if not path:
        return BASE_WINRATES_URL
    if path.startswith("range:"):
        range_id = path.split(":", 1)[1]
        return build_range_url(range_id)
    return f"{BASE_WINRATES_URL}/{path}"


def _range_csv_paths(range_key: str) -> Tuple[Path, Path]:
    base_dir = Path(OUT_HTML).parent / "data"
    base_dir.mkdir(parents=True, exist_ok=True)
    return (
        base_dir / f"archetypes_{range_key}.csv",
        base_dir / f"matchups_{range_key}.csv",
    )


def main() -> None:
    graphs: Dict[str, Tuple[nx.DiGraph, pd.DataFrame]] = {}

    in_ci = os.getenv("GITHUB_ACTIONS", "").lower() == "true"
    range_items = (
        [(DEFAULT_RANGE_KEY, RANGE_OPTIONS[DEFAULT_RANGE_KEY])]
        if in_ci and DEFAULT_RANGE_KEY in RANGE_OPTIONS
        else list(RANGE_OPTIONS.items())
    )
    if in_ci:
        print(f"[CI] GITHUB_ACTIONS detected. Scraping only '{DEFAULT_RANGE_KEY}'.")

    for range_key, meta in range_items:
        url = _build_range_url(meta.get("path", ""))
        label = meta.get("label", range_key)
        print(f"\n=== RANGE: {label} ({range_key}) ===")
        print(f"GET {url}")

        html = fetch_html(url)
        if html is None:
            print(f"[SKIP] Skipping range {range_key} (fetch failed)")
            continue
        print(f"HTML downloaded: {len(html):,} chars")

        archetypes_df, matchups_df = parse_page(html)

        out_arch, out_match = _range_csv_paths(range_key)
        archetypes_df.to_csv(out_arch, index=False)
        matchups_df.to_csv(out_match, index=False)
        print(f"Saved: {out_arch}, {out_match}")

        G = build_graph(archetypes_df, matchups_df)
        inspect_console(archetypes_df, matchups_df, G)

        graphs[range_key] = (G, archetypes_df)

    if not graphs:
        raise RuntimeError("No ranges could be fetched; aborting HTML render.")

    default_key = DEFAULT_RANGE_KEY if DEFAULT_RANGE_KEY in graphs else next(iter(graphs.keys()))
    if default_key != DEFAULT_RANGE_KEY:
        print(f"[WARN] Default range '{DEFAULT_RANGE_KEY}' missing; using '{default_key}' instead.")

    render_pyvis(graphs, OUT_HTML, default_range_key=default_key)
    print(f"\nOK: generated HTML -> {OUT_HTML}")
    print("Open it in the browser (double click).")


if __name__ == "__main__":
    main()
