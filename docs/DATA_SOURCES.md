# Data Sources

## Primary dataset: CGWB quarterly groundwater levels (1996–2017)

| | |
|---|---|
| **File** | `data/raw/cgwb_mirror/cgwb_quarterly_1996_2017.csv` |
| **Original owner** | Central Ground Water Board (CGWB), Ministry of Jal Shakti, Government of India |
| **Variables** | `STATE`, `DISTRICT`, `LAT`, `LON`, `SITE_TYPE` (dug well / bore well / tube well / dug-cum-bore well / slim hole), `WLCODE` (unique well identifier), and 84 quarterly depth-to-water-level readings (metres below ground level) from **May 1996 to Jan 2017** |
| **Size** | 28,076 monitoring wells × 84 time points |
| **Geographic coverage** | 24 states, 522 districts across India |
| **Fill rate** | 40.9% of the 28,076 × 84 cell matrix is populated; monitoring frequency and start date vary by well (10,862 wells have ≥40 of 84 readings, i.e. a genuinely usable 10-year+ series) |

### Chain of custody (read this before treating the numbers as gospel)

This is **not** a first-party download from an official CGWB/India-WRIS endpoint. The
provenance, as documented by the person who compiled it, is:

> CGWB → emailed to Tejasvi Hora (University of Waterloo) → published on GitHub by
> [craigdsouza/cgwb](https://github.com/craigdsouza/cgwb)

**Why this mirror is being used instead of the official portal:** the sandboxed
environment this project is currently being developed in has network egress
restricted to a small allow-list (package registries, GitHub) — `data.gov.in`,
`indiawris.gov.in`, `ckandev.indiadataportal.com`, `nwdp.nwic.gov.in` and
`zenodo.org` are all blocked from this container. GitHub was reachable, and this
was the most credible real (non-fabricated) redistribution of CGWB well-level
data found there. `src/data/download_cgwb_data.py` implements the **official**
download path (India-WRIS `DataDownload` portal / data.gov.in / India Data
Portal CKAN API) for use in an unrestricted environment — see that file's
docstring for exact URLs and steps. The GitHub mirror is the bootstrap dataset
so that cleaning, EDA and modelling work can start now instead of blocking on
network access.

**Before this project is presented as final / published widely, re-derive the
same table (or a superset) directly from India-WRIS or the India Data Portal
CKAN resource
(https://ckandev.indiadataportal.com/dataset/cgwb-changes-in-depth-to-water-level)
and cross-check row counts / value ranges against this mirror.** Both are
ultimately republishing the same government dataset, so they should agree
closely; if they don't, trust the official source and say so in the README.

### Known data quality issues (measured, not assumed)

- **59% missingness overall** — monitoring did not run every quarter at every
  well; some wells only came online later in the period, some stopped
  reporting.
- **Uneven site-type mix** — 70% of wells are dug wells, which respond faster
  to seasonal rainfall than bore/tube wells and are noisier for long-term
  trend estimation.
- **No post-2017 data in this table.** CGWB's real-time DWLR telemetry network
  (launched under the National Hydrology Project) covers 2019 onward but was
  not reachable from this sandbox; it's the natural extension once network
  access allows it (see `download_cgwb_data.py`).
- **State coverage gaps** — 24 of India's 28 states/8 UTs appear; a few
  states/UTs are entirely absent from this particular export.

### Is it sufficient for the proposed problem?

Yes, with the above caveats made explicit in the project write-up:
10,862 wells with a decade-plus of quarterly readings is enough to fit
per-well or per-district trend/forecast models and to flag anomalous
acceleration in decline. It is **not** enough, on its own, to make a strong
claim about 2018–2026 conditions — that requires the real-time DWLR
extension.

## Supplementary dataset (not yet integrated)

`3_statewise_stationlevel_wloc/` in the same mirror contains India-WRIS
"State-wise Station Level Report" exports (monthly, May–Oct 2020) for 6
states (Gujarat, Haryana, Karnataka, Madhya Pradesh, Maharashtra, Rajasthan,
Uttar Pradesh) as `.xls` files. Not yet cleaned/merged — flagged here as the
first candidate for extending the time series past 2017.
