from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, Tuple

import pandas as pd
import networkx as nx

from config import (
    DEFAULT_FORMAT_KEY,
    DEFAULT_RANGE_KEY,
    FORMATS,
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


def _build_range_url(base_url: str, path: str) -> str:
    if not path:
        return base_url
    if path.startswith("range:"):
        range_id = path.split(":", 1)[1]
        return build_range_url(range_id, base_url=base_url)
    return f"{base_url}/{path}"


def _range_csv_paths(format_key: str, range_key: str) -> Tuple[Path, Path]:
    base_dir = Path(OUT_HTML).parent / "data" / format_key
    base_dir.mkdir(parents=True, exist_ok=True)
    return (
        base_dir / f"archetypes_{range_key}.csv",
        base_dir / f"matchups_{range_key}.csv",
    )


def main() -> None:
    graphs_by_format: Dict[str, Dict[str, Tuple[nx.DiGraph, pd.DataFrame]]] = {}

    in_ci = os.getenv("GITHUB_ACTIONS", "").lower() == "true"
    range_items = list(RANGE_OPTIONS.items())
    format_items = list(FORMATS.items())
    if in_ci:
        print(
            "[CI] GITHUB_ACTIONS detected. Scraping only the default "
            f"format/range ({DEFAULT_FORMAT_KEY}/{DEFAULT_RANGE_KEY}) and using cached CSVs for the rest."
        )

    for format_key, format_meta in format_items:
        format_label = format_meta.get("label", format_key)
        base_url = format_meta.get("base_url", "")
        graphs_for_format: Dict[str, Tuple[nx.DiGraph, pd.DataFrame]] = {}

        for range_key, range_meta in range_items:
            url = _build_range_url(base_url, range_meta.get("path", ""))
            range_label = range_meta.get("label", range_key)
            print(f"\n=== FORMAT: {format_label} ({format_key}) | RANGE: {range_label} ({range_key}) ===")
            print(f"GET {url}")

            cache_only = in_ci and (format_key != DEFAULT_FORMAT_KEY or range_key != DEFAULT_RANGE_KEY)
            if cache_only:
                print(
                    f"[CI] Cache-only mode for {format_key}/{range_key}. Skipping network fetch."
                )
                html = None
            else:
                html = fetch_html(url, base_url=base_url)

            out_arch, out_match = _range_csv_paths(format_key, range_key)
            if html is None:
                print(f"[WARN] Fetch failed for {format_key}/{range_key}. Trying cached CSVs...")
                try:
                    archetypes_df = pd.read_csv(out_arch)
                    matchups_df = pd.read_csv(out_match)
                    print(f"[CACHE] Loaded: {out_arch}, {out_match}")
                except FileNotFoundError:
                    print(f"[SKIP] No cached CSVs for {format_key}/{range_key}. Skipping.")
                    continue
            else:
                print(f"HTML downloaded: {len(html):,} chars")
                archetypes_df, matchups_df = parse_page(html)
                archetypes_df.to_csv(out_arch, index=False)
                matchups_df.to_csv(out_match, index=False)
                print(f"Saved: {out_arch}, {out_match}")

            G = build_graph(archetypes_df, matchups_df)
            inspect_console(archetypes_df, matchups_df, G)

            graphs_for_format[range_key] = (G, archetypes_df)

        if graphs_for_format:
            graphs_by_format[format_key] = graphs_for_format

    if not graphs_by_format:
        raise RuntimeError("No format/range combinations could be fetched; aborting HTML render.")

    default_format_key = DEFAULT_FORMAT_KEY if DEFAULT_FORMAT_KEY in graphs_by_format else next(iter(graphs_by_format.keys()))
    if default_format_key != DEFAULT_FORMAT_KEY:
        print(
            f"[WARN] Default format '{DEFAULT_FORMAT_KEY}' missing; using '{default_format_key}' instead."
        )

    default_ranges = graphs_by_format[default_format_key]
    default_range_key = DEFAULT_RANGE_KEY if DEFAULT_RANGE_KEY in default_ranges else next(iter(default_ranges.keys()))
    if default_range_key != DEFAULT_RANGE_KEY:
        print(f"[WARN] Default range '{DEFAULT_RANGE_KEY}' missing; using '{default_range_key}' instead.")

    render_pyvis(
        graphs_by_format,
        OUT_HTML,
        default_format_key=default_format_key,
        default_range_key=default_range_key,
    )
    print(f"\nOK: generated HTML -> {OUT_HTML}")
    print("Open it in the browser (double click).")


if __name__ == "__main__":
    main()
