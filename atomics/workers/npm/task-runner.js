#!/usr/bin/env node
"use strict";

// Default Node.js task runner implementing the Atomics bridge protocol.
// Reads a JSON task from stdin, produces a JSON result on stdout.
// Real deployments can replace this with their own command via --worker-cmd.

let input = "";
process.stdin.on("data", (chunk) => (input += chunk));
process.stdin.on("end", () => {
  let task;
  try {
    task = JSON.parse(input);
  } catch (err) {
    console.error(`Invalid input: ${err.message}`);
    process.exit(1);
  }

  const result = {
    status: "ok",
    result: `Node.js handled: ${task.task || "unknown"}`,
    tokens: { input: 0, output: 0 },
  };

  console.log(JSON.stringify(result));
});
