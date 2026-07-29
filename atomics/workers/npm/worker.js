#!/usr/bin/env node
"use strict";

const http = require("http");
const https = require("https");
const { spawn } = require("child_process");
const { URL } = require("url");

const logger = {
  info: (msg, ...args) => console.log(`[atomics-npm-worker] ${msg}`, ...args),
  error: (msg, ...args) => console.error(`[atomics-npm-worker] ${msg}`, ...args),
};

function parseArgs(argv) {
  const args = {};
  for (let i = 2; i < argv.length; i++) {
    const key = argv[i];
    const next = argv[i + 1];
    if (key === "--coordinator" && next) {
      args.coordinator = next;
      i++;
    } else if (key === "--api-key" && next) {
      args.apiKey = next;
      i++;
    } else if (key === "--label" && next) {
      args.labels = args.labels || [];
      args.labels.push(next);
      i++;
    } else if (key === "--capability" && next) {
      args.capabilities = args.capabilities || [];
      args.capabilities.push(next);
      i++;
    } else if (key === "--worker-cmd" && next) {
      args.workerCmd = next;
      i++;
    } else if (key === "--heartbeat-interval" && next) {
      args.heartbeatInterval = parseInt(next, 10);
      i++;
    } else if (key === "--endpoint" && next) {
      args.endpoint = next;
      i++;
    }
  }
  return args;
}

function envOrArgs() {
  const args = parseArgs(process.argv);
  return {
    coordinator: process.env.ATOMICS_COORDINATOR_URL || args.coordinator || "http://127.0.0.1:8000",
    apiKey: process.env.ATOMICS_WORKER_API_KEY || args.apiKey || "",
    labels: parseLabels(process.env.ATOMICS_WORKER_LABELS || "").concat(args.labels || []),
    capabilities: parseCapabilities(process.env.ATOMICS_WORKER_CAPABILITIES || "").concat(args.capabilities || []),
    workerCmd: process.env.ATOMICS_WORKER_CMD || args.workerCmd || "node task-runner.js",
    heartbeatInterval: args.heartbeatInterval || 30,
    endpoint: process.env.ATOMICS_WORKER_ENDPOINT || args.endpoint || undefined,
  };
}

function parseLabels(raw) {
  if (!raw) return [];
  return raw.split(",").map((s) => s.trim()).filter(Boolean);
}

function parseCapabilities(raw) {
  if (!raw) return ["node"];
  return raw.split(",").map((s) => s.trim()).filter(Boolean);
}

function buildLabels(pairs) {
  const labels = {};
  for (const pair of pairs) {
    if (!pair.includes("=")) continue;
    const [k, ...rest] = pair.split("=");
    labels[k] = rest.join("=");
  }
  return labels;
}

function request(url, options = {}) {
  return new Promise((resolve, reject) => {
    const parsed = new URL(url);
    const client = parsed.protocol === "https:" ? https : http;
    const body = options.body ? JSON.stringify(options.body) : null;
    const headers = {
      "Content-Type": "application/json",
      ...(options.headers || {}),
    };
    if (body) {
      headers["Content-Length"] = Buffer.byteLength(body);
    }
    const req = client.request(
      parsed,
      {
        method: options.method || "GET",
        headers,
      },
      (res) => {
        let body = "";
        res.on("data", (chunk) => (body += chunk));
        res.on("end", () => {
          try {
            resolve({ status: res.statusCode, body: body ? JSON.parse(body) : null });
          } catch (err) {
            reject(new Error(`Invalid JSON from ${url}: ${body}`));
          }
        });
      }
    );
    req.on("error", reject);
    req.end(body);
  });
}

async function register(coordinator, apiKey, config) {
  const url = `${coordinator}/api/v1/workers/register`;
  const body = {
    labels: buildLabels(config.labels),
    capabilities: config.capabilities,
    endpoint: config.endpoint,
  };
  const res = await request(url, { method: "POST", headers: { "X-API-Key": apiKey }, body });
  if (res.status >= 400 || !res.body || !res.body.worker_id) {
    throw new Error(`Registration failed: ${res.status} ${JSON.stringify(res.body)}`);
  }
  return res.body.worker_id;
}

async function heartbeat(coordinator, apiKey, workerId) {
  const url = `${coordinator}/api/v1/workers/${workerId}/heartbeat`;
  return request(url, { method: "POST", headers: { "X-API-Key": apiKey } });
}

async function poll(coordinator, apiKey, workerId) {
  const url = `${coordinator}/api/v1/workers/${workerId}/jobs/next`;
  return request(url, { method: "GET", headers: { "X-API-Key": apiKey } });
}

async function submitResult(coordinator, apiKey, workerId, assignmentId, result) {
  const url = `${coordinator}/api/v1/workers/${workerId}/jobs/${assignmentId}/result`;
  return request(url, {
    method: "POST",
    headers: { "X-API-Key": apiKey },
    body: result,
  });
}

function executeViaBridge(cmd, taskSpec, timeoutMs) {
  return new Promise((resolve) => {
    const parts = cmd.split(" ");
    const child = spawn(parts[0], parts.slice(1), {
      stdio: ["pipe", "pipe", "pipe"],
    });
    let stdout = "";
    let stderr = "";
    let finished = false;

    const timer = setTimeout(() => {
      if (finished) return;
      finished = true;
      child.kill("SIGTERM");
      resolve({ status: "failed", error: "bridge timeout" });
    }, timeoutMs + 5000);

    child.stdout.on("data", (data) => (stdout += data));
    child.stderr.on("data", (data) => (stderr += data));
    child.on("error", (err) => {
      if (finished) return;
      finished = true;
      clearTimeout(timer);
      resolve({ status: "failed", error: `spawn error: ${err.message}` });
    });
    child.on("close", (code) => {
      if (finished) return;
      finished = true;
      clearTimeout(timer);
      if (code !== 0) {
        resolve({ status: "failed", error: stderr.slice(0, 500) || `exit code ${code}` });
        return;
      }
      try {
        const data = JSON.parse(stdout);
        resolve({
          status: "completed",
          result_json: JSON.stringify(data),
        });
      } catch (err) {
        resolve({ status: "failed", error: `invalid JSON: ${stdout.slice(0, 500)}` });
      }
    });

    child.stdin.write(JSON.stringify({ task: taskSpec.task_name || "task", prompt: taskSpec.prompt || "", timeout_ms: timeoutMs }));
    child.stdin.end();
  });
}

async function runOnce(coordinator, apiKey, workerId, config) {
  const pollRes = await poll(coordinator, apiKey, workerId);
  if (pollRes.status >= 400) {
    logger.error(`poll failed: ${pollRes.status}`);
    return false;
  }
  const assignment = pollRes.body;
  if (!assignment) return false;

  const taskSpec = assignment.task_spec || {};
  const timeoutMs = (taskSpec.timeout_seconds || 300) * 1000;
  logger.info(`executing assignment ${assignment.assignment_id}`);
  const result = await executeViaBridge(config.workerCmd, taskSpec, timeoutMs);
  await submitResult(coordinator, apiKey, workerId, assignment.assignment_id, result);
  return true;
}

async function main() {
  const config = envOrArgs();
  if (!config.apiKey) {
    logger.error("ATOMICS_WORKER_API_KEY or --api-key is required");
    process.exit(1);
  }

  const coordinator = config.coordinator.replace(/\/$/, "");
  logger.info("registering with coordinator", coordinator);
  const workerId = await register(coordinator, config.apiKey, config);
  logger.info("registered worker", workerId);

  let running = true;
  const stop = () => {
    running = false;
  };
  process.on("SIGINT", stop);
  process.on("SIGTERM", stop);

  const heartbeatMs = config.heartbeatInterval * 1000;
  let lastHeartbeat = 0;

  while (running) {
    const now = Date.now();
    if (now - lastHeartbeat >= heartbeatMs) {
      try {
        await heartbeat(coordinator, config.apiKey, workerId);
        lastHeartbeat = now;
      } catch (err) {
        logger.error("heartbeat failed", err.message);
      }
    }
    try {
      const worked = await runOnce(coordinator, config.apiKey, workerId, config);
      if (!worked) {
        await sleep(1000);
      }
    } catch (err) {
      logger.error("execute loop error", err.message);
      await sleep(1000);
    }
  }
  logger.info("shutting down");
}

function sleep(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

main().catch((err) => {
  logger.error("fatal", err);
  process.exit(1);
});
