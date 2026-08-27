"""
Official data-collection path for CGWB groundwater level data.

This project's development container has network egress restricted to a
small allow-list and cannot reach the official portals below, so the
current `data/raw/cgwb_mirror/` snapshot was bootstrapped from a GitHub
mirror instead (see docs/DATA_SOURCES.md for the full provenance and why
that's a stop-gap, not the intended source). Run this script from an
unrestricted machine to pull data directly from the primary sources and
either replace or extend the mirror.

Official sources
-----------------
1. India-WRIS DataDownload portal (interactive, no public REST API found):
   https://indiawris.gov.in/wris/#/DataDownload
   Application: Groundwater -> Report Type: State-wise Station Level Report.
   Exports monthly/annual water levels per station; requires selecting
   state + date range in the UI (see india-wris-manual-steps.md below).

2. India Data Portal (Government of India-affiliated, CKAN-based, has a
   real HTTP API):
   https://ckandev.indiadataportal.com/dataset/cgwb-changes-in-depth-to-water-level
   CKAN datastore_search API, e.g.:
     GET https://ckandev.indiadataportal.com/api/3/action/datastore_search
         ?resource_id=<resource_id>&limit=<n>

3. data.gov.in (official Government of India open data portal, needs a
   free API key from https://data.gov.in/user/register):
     GET https://api.data.gov.in/resource/<resource-id>?api-key=<key>&format=csv

This script implements path (2), the CKAN API, since it needs no
credentials and returns clean tabular data. Paths 1 and 3 are documented
above for completeness -- see the CKAN dataset page for the current
resource_id before running.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import requests

CKAN_BASE = "https://ckandev.indiadataportal.com/api/3/action/datastore_search"
DEFAULT_OUT = Path(__file__).resolve().parents[2] / "data" / "raw" / "cgwb_official"


def fetch_resource(resource_id: str, out_dir: Path, page_size: int = 5000) -> Path:
    """Page through a CKAN datastore_search resource and write it to CSV."""
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{resource_id}.csv"

    offset = 0
    fieldnames: list[str] | None = None
    rows_written = 0

    with out_path.open("w", newline="", encoding="utf-8") as fh:
        writer = None
        while True:
            resp = requests.get(
                CKAN_BASE,
                params={"resource_id": resource_id, "limit": page_size, "offset": offset},
                timeout=60,
            )
            resp.raise_for_status()
            payload = resp.json()
            if not payload.get("success"):
                raise RuntimeError(f"CKAN API returned success=false: {payload}")

            records = payload["result"]["records"]
            if not records:
                break

            if writer is None:
                fieldnames = list(records[0].keys())
                writer = csv.DictWriter(fh, fieldnames=fieldnames)
                writer.writeheader()

            writer.writerows(records)
            rows_written += len(records)
            offset += page_size
            print(f"  fetched {rows_written} rows so far...", file=sys.stderr)

            if len(records) < page_size:
                break

    print(f"Wrote {rows_written} rows to {out_path}")
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "resource_id",
        help=(
            "CKAN resource_id for the target dataset. Look this up from the "
            "dataset page URL on ckandev.indiadataportal.com, e.g. the "
            "'Changes in Depth to Water Level' resource id shown at "
            "https://ckandev.indiadataportal.com/dataset/cgwb-changes-in-depth-to-water-level"
        ),
    )
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    fetch_resource(args.resource_id, args.out_dir)


if __name__ == "__main__":
    main()
