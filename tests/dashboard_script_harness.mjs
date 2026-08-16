/**
 * Execute the dashboard page script against a fake DOM.
 *
 * Usage: node dashboard_script_harness.mjs <script.js> <report.json>
 */
import fs from "node:fs";
import vm from "node:vm";

const [, , scriptPath, reportPath] = process.argv;
if (!scriptPath || !reportPath) {
  console.error("usage: node dashboard_script_harness.mjs <script.js> <report.json>");
  process.exit(2);
}

const innerHTMLWrites = [];
const htmlSerializations = [];
const listeners = { window: [] };

class FakeClassList {
  constructor(el) {
    this.el = el;
  }
  add(name) {
    const parts = new Set((this.el.className || "").split(/\s+/).filter(Boolean));
    parts.add(name);
    this.el.className = [...parts].join(" ");
  }
  remove(name) {
    const parts = (this.el.className || "").split(/\s+/).filter((p) => p && p !== name);
    this.el.className = parts.join(" ");
  }
}

class FakeNode {
  constructor(tag) {
    this.tagName = tag;
    this.children = [];
    this._text = "";
    this.className = "";
    this.type = "";
    this.style = {};
    this.classList = new FakeClassList(this);
    this._listeners = [];
  }
  get textContent() {
    if (this.children.length) {
      return this.children.map((c) => c.textContent).join("");
    }
    return this._text;
  }
  set textContent(value) {
    this._text = value == null ? "" : String(value);
    this.children = [];
  }
  get innerHTML() {
    return this._text;
  }
  set innerHTML(value) {
    innerHTMLWrites.push(String(value));
    this._text = String(value);
    this.children = [];
  }
  appendChild(child) {
    this.children.push(child);
    return child;
  }
  append(...nodes) {
    for (const n of nodes) this.appendChild(n);
  }
  addEventListener(type, fn) {
    this._listeners.push([type, fn]);
  }
  serialize() {
    const kids = this.children.map((c) => c.serialize()).join("");
    return `<${this.tagName} class="${this.className}">${this._text}${kids}</${this.tagName}>`;
  }
}

function el(id) {
  const node = new FakeNode("div");
  node.id = id;
  return node;
}

const ids = {
  "key-warning": el("key-warning"),
  "key-input": el("key-input"),
  "key-save": el("key-save"),
  "recent-runs": el("recent-runs"),
  "distributed-jobs": el("distributed-jobs"),
  workers: el("workers"),
  compare: el("compare"),
  trends: el("trends"),
  "api-jobs": el("api-jobs"),
  "run-detail": el("run-detail"),
  "run-summary": el("run-summary"),
  "run-fixtures": el("run-fixtures"),
  "run-back": el("run-back"),
  "job-detail": el("job-detail"),
  "job-summary": el("job-summary"),
  "job-back": el("job-back"),
};
ids["key-input"].value = "";

const store = {};
const location = {
  pathname: "/dashboard",
  search: "",
  hash: "#run=dash-run",
};
const history = {
  replaceState(_s, _t, url) {
    const parsed = new URL(url, "http://127.0.0.1");
    location.pathname = parsed.pathname;
    location.search = parsed.search;
    location.hash = parsed.hash;
  },
};

const XSS_LABEL = "<img src=x onerror=alert(1)>";
const payloads = {
  "/api/v1/reports/recent-runs?limit=10": {
    runs: [
      {
        run_id: "dash-run",
        provider: "ollama",
        model: "qwen",
        tier: "refusal",
        status: "completed",
        total_tasks: 1,
        successful_tasks: 1,
        total_tokens: 5,
        total_cost_usd: 0,
      },
    ],
  },
  "/api/v1/distributed/runs?limit=10": { jobs: [] },
  "/api/v1/workers": {
    workers: [
      {
        worker_id: "w-1",
        capabilities: ["python"],
        labels: { gpu: XSS_LABEL },
        status: "online",
      },
    ],
  },
  "/api/v1/compare?by=provider": { rows: [{ provider: "ollama", success_rate: 1 }] },
  "/api/v1/reports/trends?hours=24": {
    rows: [{ hour: "2026-08-16 15:00", total_tokens: 5 }],
  },
  "/api/v1/jobs": {
    jobs: [{ job_id: "job-1", kind: "eval-job", status: "completed" }],
  },
  "/api/v1/runs/dash-run": {
    run: {
      run_id: "dash-run",
      provider: "ollama",
      model: "qwen",
      tier: "refusal",
      total_tokens: 5,
      total_cost_usd: 0,
    },
    fixtures: [
      {
        id: "rf-01",
        kind: "evaluation",
        suite: "refusal",
        score: 0.75,
        status: "complete",
        latency_ms: 12,
      },
    ],
    result: "SECRET_RESULT",
  },
  "/api/v1/jobs/job-1": {
    job_id: "job-1",
    kind: "eval-job",
    status: "completed",
    result: "SECRET_RESULT",
    error: null,
  },
};

async function fetch(path) {
  const body = payloads[path];
  if (!body) return { ok: false, json: async () => null };
  return { ok: true, json: async () => JSON.parse(JSON.stringify(body)) };
}

const document = {
  getElementById(id) {
    return ids[id] || null;
  },
  createElement(tag) {
    return new FakeNode(tag);
  },
};

const window = {
  location,
  addEventListener(type, fn) {
    listeners.window.push([type, fn]);
  },
};

const context = {
  window,
  document,
  location,
  history,
  sessionStorage: {
    getItem: (k) => store[k] ?? null,
    setItem: (k, v) => {
      store[k] = String(v);
    },
  },
  fetch,
  setInterval() {
    return 0;
  },
  URLSearchParams,
  encodeURIComponent,
  Number,
  String,
  Math,
  Object,
  Promise,
  Boolean,
  JSON,
  console,
};

vm.createContext(context);
vm.runInContext(fs.readFileSync(scriptPath, "utf8"), context);

await new Promise((r) => setTimeout(r, 30));
location.hash = "#job=job-1";
for (const [type, fn] of listeners.window) {
  if (type === "hashchange") fn();
}
await new Promise((r) => setTimeout(r, 30));

for (const node of Object.values(ids)) {
  htmlSerializations.push(node.serialize());
}

const allText = Object.values(ids)
  .map((n) => n.textContent)
  .join("\n");

fs.writeFileSync(
  reportPath,
  JSON.stringify(
    {
      innerHTMLWrites,
      htmlSerializations,
      allText,
      runDetailVisible: (ids["run-detail"].className || "").includes("visible"),
      jobDetailVisible: (ids["job-detail"].className || "").includes("visible"),
    },
    null,
    2,
  ),
);
