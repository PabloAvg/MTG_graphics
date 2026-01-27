from __future__ import annotations

from pathlib import Path

from config import OUT_ARCHETYPES_CSV, OUT_MATCHUPS_CSV, OUT_HTML, URL
from graph_build import build_graph
from render_html import render_pyvis
from scrape import fetch_html, parse_page


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


def main() -> None:
    print(f"GET {URL}")
    html = fetch_html(URL)
    print(f"HTML downloaded: {len(html):,} chars")

    archetypes_df, matchups_df = parse_page(html)

    # Ensure output folders exist after the restructure.
    Path(OUT_ARCHETYPES_CSV).parent.mkdir(parents=True, exist_ok=True)
    Path(OUT_MATCHUPS_CSV).parent.mkdir(parents=True, exist_ok=True)

    archetypes_df.to_csv(OUT_ARCHETYPES_CSV, index=False)
    matchups_df.to_csv(OUT_MATCHUPS_CSV, index=False)
    print(f"Saved: {OUT_ARCHETYPES_CSV}, {OUT_MATCHUPS_CSV}")

    G = build_graph(archetypes_df, matchups_df)
    inspect_console(archetypes_df, matchups_df, G)

    render_pyvis(G, archetypes_df, OUT_HTML)
    print(f"\nOK: generated HTML -> {OUT_HTML}")
    print("Open it in the browser (double click).")


if __name__ == "__main__":
    main()