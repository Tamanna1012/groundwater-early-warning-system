# Deployment

The dashboard (`dashboard/index.html`, built by `src/output/build_dashboard.py`)
is a single, self-contained static HTML file with zero external
dependencies (Plotly is embedded inline -- see the comments in
`build_dashboard.py` for why a CDN version doesn't work reliably). That
makes deployment as simple as static hosting gets: no server, no build
step, no environment variables, nothing to run.

## GitHub Pages (this project's chosen path)

`.github/workflows/deploy-pages.yml` publishes `dashboard/` to GitHub
Pages automatically on every push to `main` or this project's working
branch, using GitHub's official `actions/deploy-pages` action.

**One manual step is required and cannot be done from inside this
repository or any tool this project has access to:** a repository admin
must go to **Settings -> Pages -> Build and deployment -> Source** and
select **"GitHub Actions"**. This is a one-time toggle; every push after
that deploys automatically, and the workflow reports the live URL in its
run summary (Actions tab -> the latest "Deploy dashboard to GitHub Pages"
run -> the `deploy` job's `page_url` output).

Once enabled, the dashboard will be live at:
`https://tamanna1012.github.io/groundwater-early-warning-system/`

## Why not something else

- **No backend needed**: there's no live data feed (see
  `docs/RISK_ASSESSMENT.md`'s "methodology demonstration" caveat) and no
  user accounts or write-backs, so a static file is the right level of
  complexity -- not a Flask/FastAPI app, not a database.
- **Netlify/Vercel** would work equally well for a single static file, but
  GitHub Pages needs no separate account, deploys from the exact repo a
  recruiter or collaborator is already looking at, and costs nothing.
- **Re-deploying with fresh data**: re-run the pipeline end-to-end (see
  the README's "Reproducing this" section) against newer CGWB/India-WRIS
  data, then push -- the workflow picks up the regenerated
  `dashboard/index.html` automatically.
