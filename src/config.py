from pathlib import Path

# CONFIG
# Formats to scrape and compare in the UI toolbar
FORMATS = {
    "modern": {
        "label": "Modern",
        "base_url": "https://mtgdecks.net/Modern/winrates",
    },
    "standard": {
        "label": "Standard",
        "base_url": "https://mtgdecks.net/Standard/winrates",
    },
    "legacy": {
        "label": "Legacy",
        "base_url": "https://mtgdecks.net/Legacy/winrates",
    },
    "premodern": {
        "label": "Premodern",
        "base_url": "https://mtgdecks.net/Premodern/winrates",
    },
    "pauper": {
        "label": "Pauper",
        "base_url": "https://mtgdecks.net/Pauper/winrates",
    },
}
DEFAULT_FORMAT_KEY = "modern"

# Backward-compatible default base URL (Modern)
BASE_WINRATES_URL = FORMATS[DEFAULT_FORMAT_KEY]["base_url"]

# Time ranges to compare in the UI toolbar
RANGE_OPTIONS = {
    "last180days": {
        "label": "Last 180 days",
        "path": "",
    },
    "last60days": {
        "label": "Last 60 days",
        "path": "range:last60days",
    },
    "last30days": {
        "label": "Last 30 days",
        "path": "range:last30days",
    },
    "last15days": {
        "label": "Last 15 days",
        "path": "range:last15days",
    },
}
DEFAULT_RANGE_KEY = "last180days"

# Project root (one level above this src/ folder)
BASE_DIR = Path(__file__).resolve().parent.parent

# Outputs
SITE_DIR = BASE_DIR / "site"
OUT_ARCHETYPES_CSV = str(SITE_DIR / "data" / "archetypes.csv")
OUT_MATCHUPS_CSV = str(SITE_DIR / "data" / "matchups.csv")
OUT_HTML = str(SITE_DIR / "index.html")

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
