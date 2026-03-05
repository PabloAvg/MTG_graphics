from __future__ import annotations

import csv
import json
import os
import random
import re
import time
from datetime import date, timedelta
from pathlib import Path
from typing import Dict, Iterable, List, Optional
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup

from config import FORMATS, SITE_DIR


MTGO_DECKLISTS_INDEX_URL = "https://www.mtgo.com/decklists"

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9,es;q=0.8",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Connection": "keep-alive",
}


def _env_float(key: str, default: float) -> float:
    try:
        return float(os.environ.get(key, default))
    except Exception:
        return default


def _env_int(key: str, default: int) -> int:
    try:
        return int(os.environ.get(key, default))
    except Exception:
        return default


def _sleep_backoff(attempt: int, base_delay: float, jitter: float) -> None:
    wait = base_delay * (2 ** max(0, attempt - 1)) + random.uniform(0, jitter)
    time.sleep(wait)


def _resolve_verify_setting() -> bool | str:
    ca_bundle = (
        os.environ.get("MTGO_CA_BUNDLE")
        or os.environ.get("MTG_CA_BUNDLE")
        or os.environ.get("REQUESTS_CA_BUNDLE")
        or os.environ.get("CURL_CA_BUNDLE")
    )
    if ca_bundle:
        return ca_bundle
    if os.environ.get("MTGO_INSECURE_SSL") in {"1", "true", "TRUE", "yes", "YES"}:
        return False
    if os.environ.get("MTG_INSECURE_SSL") in {"1", "true", "TRUE", "yes", "YES"}:
        return False
    return True


def _fetch_html(session: requests.Session, url: str) -> Optional[str]:
    timeout = _env_float("MTGO_TIMEOUT", 30.0)
    max_retries = max(1, _env_int("MTGO_MAX_RETRIES", 3))
    base_delay = _env_float("MTGO_RETRY_BASE_DELAY", 1.5)
    jitter = _env_float("MTGO_RETRY_JITTER", 0.5)
    min_interval = _env_float("MTGO_REQUEST_DELAY", 0.9)
    verify = _resolve_verify_setting()
    last_request_ts = getattr(_fetch_html, "_last_request_ts", 0.0)

    for attempt in range(1, max_retries + 1):
        try:
            now = time.time()
            elapsed = now - last_request_ts
            if elapsed < min_interval:
                time.sleep(min_interval - elapsed)

            response = session.get(url, timeout=timeout, verify=verify)
            setattr(_fetch_html, "_last_request_ts", time.time())
            if response.status_code == 404:
                return None
            response.raise_for_status()
            return response.text
        except requests.RequestException as exc:
            print(f"[MTGO][WARN] Request failed ({attempt}/{max_retries}) for {url}: {exc}")
            if attempt < max_retries:
                _sleep_backoff(attempt, base_delay, jitter)
    return None


def _slug_from_url(url: str) -> str:
    path = urlparse(url).path.rstrip("/")
    return path.split("/")[-1] if path else ""


def _date_from_slug(slug: str) -> Optional[date]:
    match = re.search(r"(\d{4}-\d{2}-\d{2})\d*$", slug or "")
    if not match:
        return None
    y, m, d = match.group(1).split("-")
    try:
        return date(int(y), int(m), int(d))
    except Exception:
        return None


def _format_from_slug(slug: str) -> Optional[str]:
    if not slug:
        return None
    prefix = slug.split("-", 1)[0].strip().lower()
    return prefix if prefix in FORMATS else None


def _format_from_data(raw_format: str | None) -> Optional[str]:
    if not raw_format:
        return None
    fmt = str(raw_format).upper()
    for key in FORMATS:
        if key.upper() in fmt:
            return key
    return None


def _extract_js_object(source: str, marker: str) -> Dict:
    marker_idx = source.find(marker)
    if marker_idx < 0:
        raise RuntimeError(f"Marker not found: {marker}")

    start_idx = source.find("{", marker_idx)
    if start_idx < 0:
        raise RuntimeError(f"Could not find JSON object start after marker: {marker}")

    depth = 0
    in_string = False
    escaped = False

    for i in range(start_idx, len(source)):
        ch = source[i]
        if in_string:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_string = False
            continue

        if ch == '"':
            in_string = True
            continue
        if ch == "{":
            depth += 1
            continue
        if ch == "}":
            depth -= 1
            if depth == 0:
                return json.loads(source[start_idx : i + 1])

    raise RuntimeError("Could not extract JSON object (unbalanced braces).")


def _discover_tournament_urls(index_html: str) -> List[str]:
    soup = BeautifulSoup(index_html, "lxml")
    urls: List[str] = []
    seen = set()
    for a in soup.find_all("a", href=True):
        href = a["href"]
        if "/decklist/" not in href:
            continue
        full = urljoin("https://www.mtgo.com", href)
        slug = _slug_from_url(full)
        if not slug or slug in seen:
            continue
        seen.add(slug)
        urls.append(full)
    return urls


def _to_int(value) -> Optional[int]:
    try:
        return int(str(value).strip())
    except Exception:
        return None


def _safe_filename(s: str) -> str:
    bad = '<>:"/\\|?*'
    out = (s or "").strip()
    for ch in bad:
        out = out.replace(ch, "_")
    return out or "unknown"


def _build_tournament_payload(tournament_data: Dict, fallback_slug: str) -> tuple[Optional[str], Dict[str, str], List[Dict[str, str]], Dict[str, List[Dict[str, str]]]]:
    site_name = str(tournament_data.get("site_name") or fallback_slug).strip()
    format_key = _format_from_slug(site_name) or _format_from_data(tournament_data.get("format"))

    event_id = str(tournament_data.get("event_id") or "").strip()
    event_name = str(tournament_data.get("description") or "").strip()
    starttime = str(tournament_data.get("starttime") or "").strip()
    event_date = starttime[:10] if len(starttime) >= 10 else ""
    player_count = str(tournament_data.get("player_count") or "").strip()

    standings = tournament_data.get("standings") or []
    rank_by_loginid: Dict[str, str] = {}
    rank_by_login_name: Dict[str, str] = {}
    for st in standings:
        loginid = str(st.get("loginid") or "").strip()
        login_name = str(st.get("login_name") or "").strip()
        rank = str(st.get("rank") or "").strip()
        if loginid and rank:
            rank_by_loginid[loginid] = rank
        if login_name and rank:
            rank_by_login_name[login_name.casefold()] = rank

    event_row = {
        "event_id": event_id,
        "event_name": event_name,
        "format": format_key or "",
        "date": event_date,
        "site_name": site_name,
        "starttime": starttime,
        "player_count": player_count,
    }

    deck_summary_rows: List[Dict[str, str]] = []
    deck_cards_rows: Dict[str, List[Dict[str, str]]] = {}

    for deck in tournament_data.get("decklists") or []:
        deck_id = str(deck.get("decktournamentid") or "").strip()
        loginid = str(deck.get("loginid") or "").strip()
        player_name = str(deck.get("player") or "").strip()
        rank = rank_by_loginid.get(loginid) or rank_by_login_name.get(player_name.casefold(), "")

        if not deck_id:
            deck_id = f"deck_{len(deck_summary_rows) + 1}"

        deck_summary_rows.append(
            {
                "event_id": event_id,
                "deck_id": deck_id,
                "player_name": player_name,
                "rank": rank,
            }
        )

        card_rows: List[Dict[str, str]] = []
        for raw_key, deck_part in (("main_deck", "mainboard"), ("sideboard_deck", "sideboard")):
            for card in deck.get(raw_key) or []:
                qty = _to_int(card.get("qty"))
                card_name = str((card.get("card_attributes") or {}).get("card_name") or "").strip()
                if not card_name:
                    card_name = str(card.get("docid") or "").strip()
                card_rows.append(
                    {
                        "event_id": event_id,
                        "deck_id": deck_id,
                        "player_name": player_name,
                        "rank": rank,
                        "deck_part": deck_part,
                        "card_name": card_name,
                        "quantity": str(qty if qty is not None else ""),
                    }
                )
        deck_cards_rows[deck_id] = card_rows

    return format_key, event_row, deck_summary_rows, deck_cards_rows


def _write_csv(path: Path, headers: List[str], rows: Iterable[Dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=headers)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def scrape_mtgo_tournaments(base_output_dir: Path, formats: Optional[Iterable[str]] = None) -> Dict[str, int]:
    format_set = set((formats or FORMATS.keys()))
    stats = {
        "discovered": 0,
        "saved": 0,
        "skipped_existing": 0,
        "skipped_format": 0,
        "skipped_old": 0,
        "failed": 0,
    }

    session = requests.Session()
    session.headers.update(HEADERS)

    index_html = _fetch_html(session, MTGO_DECKLISTS_INDEX_URL)
    if not index_html:
        print("[MTGO][WARN] Could not fetch decklists index.")
        return stats

    urls = _discover_tournament_urls(index_html)
    lookback_days = _env_int("MTGO_LOOKBACK_DAYS", 2)
    if lookback_days > 0:
        cutoff = date.today() - timedelta(days=max(0, lookback_days - 1))
        filtered_urls: List[str] = []
        for u in urls:
            slug = _slug_from_url(u)
            slug_date = _date_from_slug(slug)
            if slug_date is None or slug_date >= cutoff:
                filtered_urls.append(u)
            else:
                stats["skipped_old"] += 1
        urls = filtered_urls
        print(f"[MTGO] Applying date window: last {lookback_days} day(s) (cutoff={cutoff.isoformat()}).")

    max_tournaments = _env_int("MTGO_MAX_TOURNAMENTS", 0)
    if max_tournaments > 0:
        urls = urls[:max_tournaments]
    stats["discovered"] = len(urls)
    print(f"[MTGO] Discovered {len(urls)} tournament links in index.")

    for url in urls:
        slug = _slug_from_url(url)
        guessed_format = _format_from_slug(slug)
        if guessed_format not in format_set:
            stats["skipped_format"] += 1
            continue

        tournament_dir = base_output_dir / guessed_format / "mtgo" / slug
        tournament_csv = tournament_dir / "tournament.csv"
        if tournament_csv.exists():
            stats["skipped_existing"] += 1
            continue

        html = _fetch_html(session, url)
        if not html:
            stats["failed"] += 1
            print(f"[MTGO][WARN] Could not fetch tournament page: {url}")
            continue

        try:
            raw = _extract_js_object(html, "window.MTGO.decklists.data")
            parsed_format, event_row, deck_summary_rows, deck_cards_rows = _build_tournament_payload(
                raw, fallback_slug=slug
            )

            if parsed_format and parsed_format in format_set and parsed_format != guessed_format:
                tournament_dir = base_output_dir / parsed_format / "mtgo" / slug
                tournament_csv = tournament_dir / "tournament.csv"

            _write_csv(
                tournament_csv,
                headers=[
                    "event_id",
                    "event_name",
                    "format",
                    "date",
                    "site_name",
                    "starttime",
                    "player_count",
                ],
                rows=[event_row],
            )
            _write_csv(
                tournament_dir / "decks.csv",
                headers=["event_id", "deck_id", "player_name", "rank"],
                rows=deck_summary_rows,
            )

            decks_dir = tournament_dir / "decks"
            for deck_id, card_rows in deck_cards_rows.items():
                deck_file = decks_dir / f"{_safe_filename(deck_id)}.csv"
                _write_csv(
                    deck_file,
                    headers=[
                        "event_id",
                        "deck_id",
                        "player_name",
                        "rank",
                        "deck_part",
                        "card_name",
                        "quantity",
                    ],
                    rows=card_rows,
                )

            stats["saved"] += 1
            print(f"[MTGO] Saved {tournament_dir}")
        except Exception as exc:
            stats["failed"] += 1
            print(f"[MTGO][WARN] Failed parsing {url}: {exc}")

    return stats


def run() -> Dict[str, int]:
    return scrape_mtgo_tournaments(base_output_dir=Path(SITE_DIR) / "data")


if __name__ == "__main__":
    run()
