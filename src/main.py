from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, Tuple, List

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


def _progress_line(current: int, total: int, label: str) -> str:
    bar_len = 24
    filled = int(bar_len * current / total) if total else bar_len
    bar = "#" * filled + "-" * (bar_len - filled)
    return f"[PROGRESS] [{bar}] {current}/{total} {label}"


def main() -> None:
    graphs_by_format: Dict[str, Dict[str, Tuple[nx.DiGraph, pd.DataFrame]]] = {}
    summary_rows: List[Dict[str, str]] = []

    in_ci = os.getenv("GITHUB_ACTIONS", "").lower() == "true"
    ci_mode = os.getenv("MTG_CI_MODE", "cache").strip().lower()
    ci_full_scrape = in_ci and ci_mode in {"full", "all", "network"}
    range_items = list(RANGE_OPTIONS.items())
    format_items = list(FORMATS.items())
    total_jobs = len(range_items) * len(format_items)
    current_job = 0

    if in_ci and not ci_full_scrape:
        print(
            "[CI] GITHUB_ACTIONS detected. Scraping only the default "
            f"format/range ({DEFAULT_FORMAT_KEY}/{DEFAULT_RANGE_KEY}) and using cached CSVs for the rest."
        )
    elif in_ci and ci_full_scrape:
        print("[CI] GITHUB_ACTIONS detected. Full scrape enabled via MTG_CI_MODE.")

    for format_key, format_meta in format_items:
        format_label = format_meta.get("label", format_key)
        base_url = format_meta.get("base_url", "")
        graphs_for_format: Dict[str, Tuple[nx.DiGraph, pd.DataFrame]] = {}

        for range_key, range_meta in range_items:
            current_job += 1
            range_label = range_meta.get("label", range_key)
            url = _build_range_url(base_url, range_meta.get("path", ""))

            print("\n" + _progress_line(current_job, total_jobs, f"{format_label}/{range_label}"))
            print(f"=== FORMAT: {format_label} ({format_key}) | RANGE: {range_label} ({range_key}) ===")

            cache_only = (
                in_ci
                and not ci_full_scrape
                and (format_key != DEFAULT_FORMAT_KEY or range_key != DEFAULT_RANGE_KEY)
            )
            if cache_only:
                print(f"[CI] Cache-only mode for {format_key}/{range_key}. Skipping network fetch.")
                html = None
            else:
                html = fetch_html(url, base_url=base_url)

            out_arch, out_match = _range_csv_paths(format_key, range_key)
            if html is None:
                if cache_only:
                    print(f"[CI] Using cached CSVs for {format_key}/{range_key}.")
                else:
                    print(f"[WARN] Fetch failed for {format_key}/{range_key}. Trying cached CSVs...")
                try:
                    archetypes_df = pd.read_csv(out_arch)
                    matchups_df = pd.read_csv(out_match)
                    print(f"[CACHE] Loaded: {out_arch}, {out_match}")
                    summary_rows.append(
                        {
                            "format": format_key,
                            "range": range_key,
                            "status": "cached",
                            "source": "cache",
                            "details": "used local CSVs",
                        }
                    )
                except FileNotFoundError:
                    print(f"[SKIP] No cached CSVs for {format_key}/{range_key}. Skipping.")
                    summary_rows.append(
                        {
                            "format": format_key,
                            "range": range_key,
                            "status": "skipped",
                            "source": "none",
                            "details": "no cache available",
                        }
                    )
                    continue
            else:
                print(f"HTML downloaded: {len(html):,} chars")
                try:
                    archetypes_df, matchups_df = parse_page(html)
                except Exception as e:
                    print(f"[FAIL] Parse error for {format_key}/{range_key}: {e}")
                    summary_rows.append(
                        {
                            "format": format_key,
                            "range": range_key,
                            "status": "failed",
                            "source": "network",
                            "details": "parse failed",
                        }
                    )
                    continue

                archetypes_df.to_csv(out_arch, index=False)
                matchups_df.to_csv(out_match, index=False)
                print(f"Saved: {out_arch}, {out_match}")
                summary_rows.append(
                    {
                        "format": format_key,
                        "range": range_key,
                        "status": "updated",
                        "source": "network",
                        "details": "saved CSVs",
                    }
                )

            try:
                G = build_graph(archetypes_df, matchups_df)
                inspect_console(archetypes_df, matchups_df, G)
                graphs_for_format[range_key] = (G, archetypes_df)
            except Exception as e:
                print(f"[FAIL] Graph build failed for {format_key}/{range_key}: {e}")
                summary_rows.append(
                    {
                        "format": format_key,
                        "range": range_key,
                        "status": "failed",
                        "source": "graph",
                        "details": "graph build failed",
                    }
                )

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

    if summary_rows:
        print("\n=== SUMMARY ===")
        df = pd.DataFrame(summary_rows)
        print(df.to_string(index=False))


if __name__ == "__main__":
    main()
