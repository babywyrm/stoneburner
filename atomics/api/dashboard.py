"""Optional web dashboard for the atomics API server.

Served at /dashboard when the server is started with --with-dashboard. It is
read-only and purely visual: all data comes from the existing API endpoints.
"""

from __future__ import annotations

from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse

from atomics.api.headers import dashboard_csp, new_nonce

router = APIRouter()

_DASHBOARD_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>atomics dashboard</title>
  <style>
    :root { color-scheme: light dark; }
    body { font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; margin: 0; padding: 2rem; background: #f5f5f5; color: #111; }
    @media (prefers-color-scheme: dark) { body { background: #0d1117; color: #e6edf3; } }
    h1 { margin-top: 0; }
    .grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(320px, 1fr)); gap: 1.5rem; margin-top: 1.5rem; }
    .card { background: #fff; border-radius: 8px; padding: 1.25rem; box-shadow: 0 1px 3px rgba(0,0,0,0.1); }
    @media (prefers-color-scheme: dark) { .card { background: #161b22; } }
    .card h2 { margin-top: 0; font-size: 1.1rem; }
    table { width: 100%; border-collapse: collapse; font-size: 0.9rem; }
    th, td { text-align: left; padding: 0.5rem 0.75rem; border-bottom: 1px solid #ddd; }
    @media (prefers-color-scheme: dark) { th, td { border-color: #30363d; } }
    .status { font-weight: 600; }
    .status.ok { color: #1a7f37; }
    .status.error { color: #cf222e; }
    .status.pending { color: #9a6700; }
    .refresh { float: right; font-size: 0.85rem; color: #666; }
    .empty { color: #666; font-style: italic; }
    .bar { height: 1.25rem; background: #58a6ff; border-radius: 4px; min-width: 2px; }
    .row { display: flex; align-items: center; gap: 0.75rem; margin-bottom: 0.4rem; }
    .row span { width: 8rem; font-size: 0.85rem; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
    button.link { background: none; border: none; padding: 0; color: #0969da; cursor: pointer; font: inherit; }
    @media (prefers-color-scheme: dark) { button.link { color: #58a6ff; } }
    #run-detail { display: none; margin-top: 1.5rem; }
    #run-detail.visible { display: block; }
  </style>
</head>
<body>
  <h1>atomics dashboard <span class="refresh">refreshes every 10s</span></h1>
  <p id="key-warning" style="display:none;">
    <label>API key <input id="key-input" type="password" autocomplete="off" placeholder="X-API-Key"></label>
    <button id="key-save">Connect</button>
  </p>
  <div class="grid">
    <div class="card">
      <h2>Recent runs</h2>
      <div id="recent-runs"><p class="empty">Loading...</p></div>
    </div>
    <div class="card">
      <h2>Distributed jobs</h2>
      <div id="distributed-jobs"><p class="empty">Loading...</p></div>
    </div>
    <div class="card">
      <h2>Workers</h2>
      <div id="workers"><p class="empty">Loading...</p></div>
    </div>
    <div class="card">
      <h2>Compare by provider</h2>
      <div id="compare"><p class="empty">Loading...</p></div>
    </div>
  </div>
  <div id="run-detail" class="card">
    <h2>Run <button type="button" id="run-back" class="link">← all runs</button></h2>
    <div id="run-summary"><p class="empty">Select a run.</p></div>
    <div id="run-fixtures"></div>
  </div>

  <script>
    // The key is held in sessionStorage rather than the URL so it does not leak
    // into browser history, referrer headers, or reverse-proxy access logs.
    function readKey() {
      const params = new URLSearchParams(window.location.search);
      const fromUrl = params.get("api_key");
      if (fromUrl) {
        sessionStorage.setItem("atomics_api_key", fromUrl);
        params.delete("api_key");
        const rest = params.toString();
        history.replaceState(null, "", window.location.pathname + (rest ? "?" + rest : ""));
        return fromUrl;
      }
      return sessionStorage.getItem("atomics_api_key") || "";
    }

    let API_KEY = readKey();

    async function get(path) {
      const headers = API_KEY ? { "X-API-Key": API_KEY } : {};
      const res = await fetch(path, { headers });
      if (!res.ok) return null;
      return res.json();
    }

    function statusClass(s) {
      if (!s) return "pending";
      s = s.toLowerCase();
      if (s === "ok" || s === "completed" || s === "success" || s === "online") return "ok";
      if (s === "error" || s === "failed" || s === "offline") return "error";
      return "pending";
    }

    // Every value below reaches the DOM through textContent. Worker labels and
    // capabilities are caller-supplied, so string-concatenating them into
    // innerHTML would let any registered worker script the operator's browser.
    function emptyNote(container, message) {
      container.textContent = "";
      const p = document.createElement("p");
      p.className = "empty";
      p.textContent = message;
      container.appendChild(p);
    }

    function renderTable(id, rows, headers) {
      const container = document.getElementById(id);
      if (!rows || rows.length === 0) { emptyNote(container, "No data yet."); return; }
      const table = document.createElement("table");
      const headRow = document.createElement("tr");
      for (const h of headers) {
        const th = document.createElement("th");
        th.textContent = h;
        headRow.appendChild(th);
      }
      table.appendChild(headRow);
      for (const r of rows) {
        const tr = document.createElement("tr");
        for (const c of r) {
          const td = document.createElement("td");
          if (c !== null && typeof c === "object") {
            td.className = c.class;
            td.textContent = c.text;
          } else {
            td.textContent = c == null ? "-" : String(c);
          }
          tr.appendChild(td);
        }
        table.appendChild(tr);
      }
      container.textContent = "";
      container.appendChild(table);
    }

    function selectRun(runId) {
      if (location.hash !== "#run=" + runId) {
        location.hash = "run=" + runId;
      }
      loadRunDetail(runId);
    }

    function clearRun() {
      if (location.hash.startsWith("#run=")) {
        history.replaceState(null, "", location.pathname + location.search);
      }
      document.getElementById("run-detail").classList.remove("visible");
    }

    async function loadRunDetail(runId) {
      const panel = document.getElementById("run-detail");
      const summary = document.getElementById("run-summary");
      const fixtures = document.getElementById("run-fixtures");
      panel.classList.add("visible");
      emptyNote(summary, "Loading " + runId.slice(0, 8) + "…");
      fixtures.textContent = "";
      const data = await get("/api/v1/runs/" + encodeURIComponent(runId));
      if (!data || !data.run) {
        emptyNote(summary, "Run not found.");
        return;
      }
      const r = data.run;
      summary.textContent = "";
      const meta = document.createElement("p");
      meta.textContent = [
        r.provider || "-",
        r.model || "-",
        r.tier || "-",
        (r.total_tokens != null ? r.total_tokens + " tok" : ""),
        (r.total_cost_usd != null ? "$" + Number(r.total_cost_usd).toFixed(4) : ""),
      ].filter(Boolean).join(" · ");
      summary.appendChild(meta);
      const rows = (data.fixtures || []).map(f => [
        f.id,
        f.kind,
        f.suite || "-",
        f.score == null ? "-" : Number(f.score).toFixed(2),
        f.label || f.status || "-",
        f.latency_ms == null ? "-" : Math.round(f.latency_ms) + "ms",
      ]);
      renderTable("run-fixtures", rows, ["Fixture", "Kind", "Suite", "Score", "Label", "Latency"]);
    }

    async function loadRecentRuns() {
      const data = await get("/api/v1/reports/recent-runs?limit=10");
      const container = document.getElementById("recent-runs");
      const runs = data?.runs || [];
      if (!runs.length) { emptyNote(container, "No data yet."); return; }
      const table = document.createElement("table");
      const headRow = document.createElement("tr");
      for (const h of ["Run", "Provider", "Tier", "Status", "Tasks", "OK", "Tokens", "Cost"]) {
        const th = document.createElement("th");
        th.textContent = h;
        headRow.appendChild(th);
      }
      table.appendChild(headRow);
      for (const r of runs) {
        const tr = document.createElement("tr");
        const idCell = document.createElement("td");
        const btn = document.createElement("button");
        btn.type = "button";
        btn.className = "link";
        btn.textContent = String(r.run_id).slice(0, 8);
        btn.addEventListener("click", () => selectRun(r.run_id));
        idCell.appendChild(btn);
        tr.appendChild(idCell);
        const cells = [
          r.provider || r.model || "-",
          r.tier || "-",
          { class: "status " + statusClass(r.status), text: r.status || "-" },
          r.total_tasks ?? "-",
          r.successful_tasks ?? "-",
          r.total_tokens ?? "-",
          r.total_cost_usd != null ? "$" + r.total_cost_usd.toFixed(4) : "-",
        ];
        for (const c of cells) {
          const td = document.createElement("td");
          if (c !== null && typeof c === "object") {
            td.className = c.class;
            td.textContent = c.text;
          } else {
            td.textContent = c == null ? "-" : String(c);
          }
          tr.appendChild(td);
        }
        table.appendChild(tr);
      }
      container.textContent = "";
      container.appendChild(table);
    }

    async function loadDistributedJobs() {
      const data = await get("/api/v1/distributed/runs?limit=10");
      const rows = (data?.jobs || []).map(j => [
        j.job_id.slice(0, 8),
        j.mode,
        { class: "status " + statusClass(j.status), text: j.status || "-" },
        (j.assignments || []).length,
      ]);
      renderTable("distributed-jobs", rows, ["Job", "Mode", "Status", "Assignments"]);
    }

    async function loadWorkers() {
      const data = await get("/api/v1/workers");
      const rows = (data?.workers || []).map(w => [
        w.worker_id.slice(0, 8),
        w.capabilities ? w.capabilities.join(", ") : "-",
        Object.entries(w.labels || {}).map(([k, v]) => k + "=" + v).join(", ") || "-",
        { class: "status " + statusClass(w.status), text: w.status || "-" },
      ]);
      renderTable("workers", rows, ["Worker", "Capabilities", "Labels", "Status"]);
    }

    async function loadCompare() {
      const data = await get("/api/v1/compare?by=provider");
      const rows = (data?.rows || []).sort((a, b) => (b.success_rate || 0) - (a.success_rate || 0));
      const container = document.getElementById("compare");
      if (!rows || rows.length === 0) { emptyNote(container, "No comparison data yet."); return; }
      container.textContent = "";
      for (const r of rows) {
        const pct = Math.max(0, Math.min(100, Math.round((r.success_rate || 0) * 100)));
        const row = document.createElement("div");
        row.className = "row";
        const name = document.createElement("span");
        name.textContent = r.provider || r.model || "-";
        const bar = document.createElement("div");
        bar.className = "bar";
        bar.style.width = pct + "%";
        const value = document.createElement("span");
        value.textContent = pct + "%";
        row.append(name, bar, value);
        container.appendChild(row);
      }
    }

    async function refresh() {
      await Promise.all([loadRecentRuns(), loadDistributedJobs(), loadWorkers(), loadCompare()]);
    }

    document.getElementById("key-save").addEventListener("click", () => {
      API_KEY = document.getElementById("key-input").value.trim();
      sessionStorage.setItem("atomics_api_key", API_KEY);
      document.getElementById("key-warning").style.display = "none";
      refresh();
    });
    document.getElementById("run-back").addEventListener("click", clearRun);
    window.addEventListener("hashchange", () => {
      if (location.hash.startsWith("#run=")) {
        loadRunDetail(location.hash.slice(5));
      } else {
        document.getElementById("run-detail").classList.remove("visible");
      }
    });

    if (!API_KEY) {
      document.getElementById("key-warning").style.display = "block";
    }
    refresh();
    if (location.hash.startsWith("#run=")) {
      loadRunDetail(location.hash.slice(5));
    }
    setInterval(refresh, 10000);
  </script>
</body>
</html>
"""


@router.get("/dashboard", response_class=HTMLResponse)
async def dashboard_index(request: Request) -> HTMLResponse:
    """Serve the dashboard HTML under a per-response CSP nonce."""
    nonce = new_nonce()
    html = _DASHBOARD_HTML.replace("<style>", f'<style nonce="{nonce}">').replace(
        "<script>", f'<script nonce="{nonce}">'
    )
    return HTMLResponse(
        content=html,
        headers={"Content-Security-Policy": dashboard_csp(nonce)},
    )
