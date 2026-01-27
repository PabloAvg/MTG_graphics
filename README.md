# MTG_graphics
python --version
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
$env:MTG_INSECURE_SSL="1"
python .\01_fetch_mtgdecks_winrates.py

# Output for GitHub Pages (static site)
# Generates: site/index.html
