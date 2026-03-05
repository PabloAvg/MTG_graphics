from __future__ import annotations

from mtgdecks_scrapper import run as run_mtgdecks_scrapper
from mtgo_scrapper import run as run_mtgo_scrapper


def main() -> None:
    print("=== STEP 1/2: MTGDecks scrape + graph render ===")
    run_mtgdecks_scrapper()

    print("\n=== STEP 2/2: MTGO tournaments scrape ===")
    stats = run_mtgo_scrapper()
    print(
        "[MTGO] Summary: "
        f"discovered={stats['discovered']} "
        f"saved={stats['saved']} "
        f"skipped_existing={stats['skipped_existing']} "
        f"skipped_format={stats['skipped_format']} "
        f"skipped_old={stats['skipped_old']} "
        f"failed={stats['failed']}"
    )


if __name__ == "__main__":
    main()
