from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import Optional, Tuple, List, Dict
from urllib.parse import quote

import pandas as pd
import requests
from bs4 import BeautifulSoup

from config import BASE_WINRATES_URL, DEFAULT_RANGE_KEY, RANGE_OPTIONS


RE_MATCHES = re.compile(r"([\d,]+)\s*matches", re.IGNORECASE)
RE_PCT = re.compile(r"(\d+(?:\.\d+)?)\s*%")

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9,es;q=0.8",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Connection": "keep-alive",
    "Upgrade-Insecure-Requests": "1",
}

SESSION = requests.Session()
SESSION.headers.update(HEADERS)


@dataclass(frozen=True)
class Cell:
    winrate: Optional[float]       # 0..1 (row archetype winrate vs column archetype)
    matches: int
    ci_low: Optional[float]
    ci_high: Optional[float]


def _clean_text(s: str) -> str:
    return " ".join((s or "").replace("\xa0", " ").split()).strip()


def parse_cell(td) -> Cell:
    """
    td expected:
      <td class="winrate-cell" data-winrate="53">
        <div class="data">
          <div class="confidence-interval">52% - 54%</div>
          <b>53</b><span class="percent">%</span>
          <div class="matches-number">7,743 matches</div>
        </div>
      </td>

    If no data:
      <td class="winrate-cell"><div class="data"><b>--</b></div></td>
    """
    wr_attr = td.get("data-winrate")
    if wr_attr is None:
        return Cell(winrate=None, matches=0, ci_low=None, ci_high=None)

    try:
        wr = float(wr_attr) / 100.0
    except ValueError:
        return Cell(winrate=None, matches=0, ci_low=None, ci_high=None)

    matches = 0
    mn = td.find("div", class_="matches-number")
    if mn:
        m = RE_MATCHES.search(_clean_text(mn.get_text(" ", strip=True)))
        if m:
            matches = int(m.group(1).replace(",", ""))

    ci_low = ci_high = None
    ci = td.find("div", class_="confidence-interval")
    if ci:
        pcts = RE_PCT.findall(_clean_text(ci.get_text(" ", strip=True)))
        if len(pcts) >= 2:
            ci_low = float(pcts[0]) / 100.0
            ci_high = float(pcts[1]) / 100.0

    return Cell(winrate=wr, matches=matches, ci_low=ci_low, ci_high=ci_high)


def build_range_url(range_id: str, base_url: str = BASE_WINRATES_URL) -> str:
    """
    Build an mtgdecks range URL using encoded `range:<id>` paths, e.g.:
    https://mtgdecks.net/Modern/winrates/range%3Alast60days
    """
    encoded_path = quote(f"range:{range_id}", safe="")
    return f"{base_url}/{encoded_path}"


def _range_url(range_key: str = DEFAULT_RANGE_KEY, base_url: str = BASE_WINRATES_URL) -> str:
    meta = RANGE_OPTIONS.get(range_key, {})
    path = meta.get("path", "")
    if not path:
        return base_url
    if path.startswith("range:"):
        return build_range_url(path.split(":", 1)[1], base_url=base_url)
    encoded_path = quote(path, safe="")
    return f"{base_url}/{encoded_path}"


def fetch_html(
    url: str | None = None,
    range_key: str = DEFAULT_RANGE_KEY,
    base_url: str = BASE_WINRATES_URL,
) -> str | None:
    if not url:
        url = _range_url(range_key, base_url=base_url)

    # Corporate proxies often inject a custom root CA. Respect common env vars.
    ca_bundle = (
        os.environ.get("MTG_CA_BUNDLE")
        or os.environ.get("REQUESTS_CA_BUNDLE")
        or os.environ.get("CURL_CA_BUNDLE")
    )
    verify: bool | str = True
    if ca_bundle:
        verify = ca_bundle
    elif os.environ.get("MTG_INSECURE_SSL") in {"1", "true", "TRUE", "yes", "YES"}:
        # Last resort: allow opting out of verification explicitly.
        verify = False

    try:
        print(f"GET {url}")
        r = SESSION.get(url, timeout=30, verify=verify)
        if r.status_code == 403:
            print(f"[SKIP] 403 Forbidden for {url}")
            return None
        r.raise_for_status()
        return r.text
    except requests.HTTPError as e:
        print(f"[WARN] Failed to fetch {url}: {e}")
        return None
    except requests.RequestException as e:
        print(f"[WARN] Request error for {url}: {e}")
        return None


def parse_page(html: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    soup = BeautifulSoup(html, "lxml")

    table = soup.find("table", id="winrates")
    if not table:
        raise RuntimeError("Could not find <table id='winrates'>. Possible block/HTML change.")

    thead = table.find("thead")
    if not thead:
        raise RuntimeError("Could not find <thead> in winrates table.")

    ths = thead.find_all("th")
    colnames = [_clean_text(th.get_text(" ", strip=True)) for th in ths]

    if len(colnames) < 3 or colnames[1] != "Overall":
        raise RuntimeError(f"Unexpected headers. First ones: {colnames[:5]}")

    tbody = table.find("tbody")
    if not tbody:
        raise RuntimeError("Could not find <tbody> in winrates table.")

    trs = tbody.find_all("tr", class_="item")
    if not trs:
        raise RuntimeError("Could not find <tr class='item'> rows.")

    archetype_rows: List[Dict] = []
    matchup_rows: List[Dict] = []

    opponent_cols = colnames[2:]  # only top archetypes (>=2% matches per mtgdecks)

    for tr in trs:
        a_name = _clean_text(tr.get("data-name", ""))
        if not a_name:
            continue

        a_overall_wr = tr.get("data-winrate")
        a_overall_matches = tr.get("data-matches")

        overall_wr = float(a_overall_wr) if a_overall_wr else None
        overall_matches = int(a_overall_matches) if a_overall_matches else 0

        a_url = None
        header_td = tr.find("td", class_="header")
        if header_td:
            link = header_td.find("a")
            if link and link.get("href"):
                a_url = link["href"]

        tds = tr.find_all("td")
        if len(tds) < 2:
            continue

        overall_cell = parse_cell(tds[1])

        archetype_rows.append(
            {
                "archetype": a_name,
                "overall_winrate": overall_wr,  # 0..1
                "overall_matches": overall_matches,
                "overall_ci_low": overall_cell.ci_low,
                "overall_ci_high": overall_cell.ci_high,
                "url_relative": a_url,
            }
        )

        for j, b_name in enumerate(opponent_cols):
            td_index = 2 + j
            if td_index >= len(tds):
                break
            cell = parse_cell(tds[td_index])
            if cell.winrate is None or cell.matches == 0:
                continue

            matchup_rows.append(
                {
                    "a": a_name,
                    "b": b_name,
                    "winrate_a_vs_b": cell.winrate,  # 0..1
                    "matches": cell.matches,
                    "ci_low": cell.ci_low,
                    "ci_high": cell.ci_high,
                }
            )

    return pd.DataFrame(archetype_rows), pd.DataFrame(matchup_rows)
