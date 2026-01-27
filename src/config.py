from pathlib import Path

# CONFIG
URL = "https://mtgdecks.net/Modern/winrates"

# Project root (one level above this src/ folder)
BASE_DIR = Path(__file__).resolve().parent.parent

# Outputs
OUT_ARCHETYPES_CSV = str(BASE_DIR / "data" / "archetypes.csv")
OUT_MATCHUPS_CSV = str(BASE_DIR / "data" / "matchups.csv")
OUT_HTML = str(BASE_DIR / "modern_meta_graph.html")

# Graph coverage / filtering
MIN_MATCHES_EDGE = 1           # include more edges (can be noisy)
EPS_TIE = 0.005                # tie threshold around 50% (+/- EPS)
HIDE_ISOLATED_NODES = False    # set True to remove isolated nodes
TOP_N_ARCHETYPES = 35          # keep only the top N most represented decks
EMBED_ASSETS = True            # inline JS/CSS so the HTML works standalone

# Visual scaling
NODE_SIZE_MIN = 6
NODE_SIZE_MAX = 40

EDGE_WIDTH_MIN = 0.2
EDGE_WIDTH_MAX = 16

# Radial layout (by overall_matches)
RING_COUNT = 5                 # number of rings
RING_RADIUS_BASE = 260         # inner ring radius
RING_RADIUS_STEP = 220         # distance between rings