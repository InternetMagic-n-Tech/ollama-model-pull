#!/usr/bin/env node

/**
 * ollama-pull.js — Reliable Ollama model downloader with adaptive stall detection,
 * exponential backoff + connectivity/clock preflight, a hello-test, and corrupt-blob cleanup
 */

const yargs = require("yargs");
const { hideBin } = require("yargs/helpers");
const fs = require("fs");
const path = require("path");
const crypto = require("crypto");
const os = require("os");

const argv = yargs(hideBin(process.argv))
  .usage("Usage: $0 [model] [options]")
  .option("model", {
    alias: "m",
    type: "string",
    description: "Model to pull",
    default: "llama3.2:latest",
  })
  .option("host", {
    alias: "H",
    type: "string",
    description: "Ollama server URL",
    default: process.env.OLLAMA_HOST || "http://127.0.0.1:11434",
  })
  .option("band-size", {
    alias: "b",
    type: "number",
    description: "% band size to track (e.g. 10 = check every 10%)",
    default: 10,
  })
  .option("stall-factor", {
    alias: "f",
    type: "number",
    description: "How many times slower the current band can be vs the slowest good band before restarting",
    default: 10,
  })
  .option("first-band-timeout", {
    alias: "t",
    type: "number",
    description:
      "Seconds to wait before restarting when there is no baseline yet, " +
      "and max seconds with zero bytes received at any time",
    default: 300,
  })
  .option("max-failures", {
    alias: ["e", "max-errors"],
    type: "number",
    description: "Exit after this many failures in total (errors, stalls, dropped streams, failed hello tests)",
    default: 9,
  })
  .option("backoff-base", {
    type: "number",
    description: "First backoff delay in seconds; doubles on every failure",
    default: 2,
  })
  .option("backoff-max", {
    type: "number",
    description: "Maximum backoff delay in seconds",
    default: 120,
  })
  .option("offline-timeout", {
    type: "number",
    description: "Seconds to wait for the internet / Ollama server to come back before giving up",
    default: 900,
  })
  .option("finish-timeout", {
    type: "number",
    description: "Max seconds the server may stay silent while verifying/finishing before the pull is restarted",
    default: 900,
  })
  .option("ping-urls", {
    type: "string",
    description: "Comma-separated reliable URLs pinged (and whose Date header is compared to your clock) before each backoff",
    default: "https://www.cloudflare.com/cdn-cgi/trace,https://www.google.com/generate_204",
  })
  .option("verify", {
    type: "boolean",
    description: "Run a 'hello' test after the pull and clean up + re-pull corrupt blobs (use --no-verify to skip)",
    default: true,
  })
  .option("verify-timeout", {
    type: "number",
    description: "Seconds to wait for the hello test (loading a huge model can take minutes)",
    default: 900,
  })
  .option("models-dir", {
    type: "string",
    description: "Ollama models directory (used to find/delete corrupt blobs; must be local)",
    default: process.env.OLLAMA_MODELS || "/usr/share/ollama/.ollama/models",
  })
  .example("$0 qwen3:30b", "Pull qwen3:30b with default stall detection")
  .example("$0 -m mistral:7b -b 5 -f 5", "5% bands, restart if 5x slower than last band")
  .help()
  .alias("help", "h")
  .argv;

const model = argv._[0] || argv.model;
const OLLAMA_HOST = String(argv.host).replace(/\/+$/, "");
const BAND_SIZE = argv.bandSize; // % per band
const STALL_FACTOR = argv.stallFactor;
const FIRST_BAND_TIMEOUT_MS = argv.firstBandTimeout * 1000;
const MAX_FAILURES = argv.maxFailures;
const BACKOFF_BASE_MS = argv.backoffBase * 1000;
const BACKOFF_MAX_MS = argv.backoffMax * 1000;
const OFFLINE_TIMEOUT_MS = argv.offlineTimeout * 1000;
const FINISH_TIMEOUT_MS = argv.finishTimeout * 1000;
const VERIFY_TIMEOUT_MS = argv.verifyTimeout * 1000;
const MODELS_DIR = argv.modelsDir;

if (!(BAND_SIZE > 0 && BAND_SIZE <= 100)) {
  console.error("❌ --band-size must be between 0 and 100");
  process.exit(1);
}
if (!(Number.isInteger(MAX_FAILURES) && MAX_FAILURES >= 1)) {
  console.error("❌ --max-failures must be a whole number >= 1");
  process.exit(1);
}
if (!(STALL_FACTOR > 1)) {
  console.error("❌ --stall-factor must be greater than 1");
  process.exit(1);
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/** Errors that retrying cannot fix (bad model name, 4xx, etc). */
class PullError extends Error {}

const TRANSIENT_RE =
  /EOF|timeout|timed out|reset|refused|broken pipe|connection|network|unexpected|max retries|i\/o|temporar|try again|terminated|stall|partial|no such file or directory|service unavailable|bad gateway|gateway time-?out|too many requests|rate limit|busy|(?:^|[\s:(])(?:408|425|429|5\d\d)(?=$|[\s:)])/i;

function classifyError(text) {
  return TRANSIENT_RE.test(text) ? new Error(text) : new PullError(text);
}

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

function formatBytes(bytes) {
  if (!bytes) return "?";
  const gb = bytes / 1024 ** 3;
  if (gb >= 1) return `${gb.toFixed(2)} GB`;
  const mb = bytes / 1024 ** 2;
  if (mb >= 1) return `${mb.toFixed(1)} MB`;
  return `${(bytes / 1024).toFixed(1)} KB`;
}

function renderBar(completed, total, width = 30) {
  if (!total) return "[" + "░".repeat(width) + "]";
  const filled = Math.min(width, Math.round((completed / total) * width));
  return "[" + "█".repeat(filled) + "░".repeat(width - filled) + "]";
}

// ---------------------------------------------------------------------------
// Resilience: failure budget, exponential backoff, preflight checks, lock file
// ---------------------------------------------------------------------------

/** Stops the whole run immediately with a specific exit code (no more retries). */
class GiveUp extends Error {
  constructor(message, exitCode = 1) {
    super(message);
    this.exitCode = exitCode;
  }
}

/**
 * Monotonic clock in ms. Unlike Date.now() it never jumps when NTP corrects the
 * clock, so stall timers can't fire (or be skipped) because of a time change.
 */
const mono = () => Number(process.hrtime.bigint() / 1000000n);

const PING_URLS = String(argv.pingUrls).split(",").map((s) => s.trim()).filter(Boolean);
const REGISTRY_URL = "https://registry.ollama.ai/v2/";
const CLOCK_SKEW_WARN_MS = 60 * 1000;

function parseRetryAfter(value) {
  if (!value) return 0;
  const secs = Number(value);
  if (Number.isFinite(secs)) return Math.max(0, secs * 1000);
  const date = Date.parse(value);
  return Number.isNaN(date) ? 0 : Math.max(0, date - Date.now());
}

/** Pings a reliable server; returns round-trip time and our clock offset vs its Date header. */
async function pingReliable() {
  for (const url of PING_URLS) {
    const wallBefore = Date.now();
    const t0 = mono();
    try {
      const res = await fetch(url, { signal: AbortSignal.timeout(5000), redirect: "manual" });
      const rtt = mono() - t0;
      try { await res.body?.cancel(); } catch { /* ignore */ }
      const serverTime = Date.parse(res.headers.get("date") || "");
      const skew = Number.isNaN(serverTime) ? null : serverTime - (wallBefore + rtt / 2);
      return { url, rtt, skew };
    } catch {
      /* try the next one */
    }
  }
  return null;
}

/** HTTP status of the Ollama registry, or null if unreachable. Informational only. */
async function probeRegistry() {
  try {
    const res = await fetch(REGISTRY_URL, { signal: AbortSignal.timeout(5000) });
    try { await res.body?.cancel(); } catch { /* ignore */ }
    return res.status;
  } catch {
    return null;
  }
}

async function pingOllama() {
  try {
    const res = await fetch(`${OLLAMA_HOST}/api/version`, { signal: AbortSignal.timeout(5000) });
    try { await res.body?.cancel(); } catch { /* ignore */ }
    return res.ok;
  } catch {
    return false;
  }
}

/** Polls `check` until it returns something truthy, or gives up after --offline-timeout. */
async function waitFor(check, label) {
  const start = mono();
  let wait = 2000;
  let announced = false;
  while (true) {
    const result = await check();
    if (result) {
      if (announced) console.log(`   ✓ ${label} is back`);
      return result;
    }
    if (mono() - start > OFFLINE_TIMEOUT_MS) {
      throw new GiveUp(`${label} unreachable for ${argv.offlineTimeout}s, giving up.`, 3);
    }
    if (!announced) {
      console.log(`   ✗ ${label} unreachable, waiting for it to come back...`);
      announced = true;
    }
    await sleep(wait);
    wait = Math.min(30000, wait * 2);
  }
}

/** Run before every backoff: is the internet up, is our clock sane, is Ollama alive? */
async function preflight() {
  const net = await waitFor(pingReliable, "Internet connection");
  let line = `   🌐 Online via ${new URL(net.url).hostname} (${net.rtt.toFixed(0)}ms)`;
  if (net.skew !== null) line += ` · clock vs server: ${net.skew >= 0 ? "+" : ""}${(net.skew / 1000).toFixed(1)}s`;
  console.log(line);
  if (net.skew !== null && Math.abs(net.skew) > CLOCK_SKEW_WARN_MS) {
    console.log(
      `   ⚠️  Your system clock is off by ${(Math.abs(net.skew) / 1000).toFixed(0)}s. ` +
        `That can break TLS and registry auth. Try: sudo timedatectl set-ntp true`
    );
  }

  let registryDown = false;
  const reg = await probeRegistry();
  if (reg === null) {
    console.log("   ⚠️  registry.ollama.ai is unreachable from here");
    registryDown = true;
  } else if (reg >= 500) {
    console.log(`   ⚠️  registry.ollama.ai is returning HTTP ${reg} (their side, not yours)`);
    registryDown = true;
  }

  await waitFor(async () => ((await pingOllama()) ? true : null), `Ollama server at ${OLLAMA_HOST}`);
  return { registryDown };
}

/**
 * Records one failure (any kind: error, stall, dropped stream, failed hello test).
 * Exits via GiveUp on the MAX_FAILURES-th; otherwise preflights, then sleeps with
 * exponential backoff + jitter.
 */
let failureCount = 0;
async function registerFailure(reason, { retryAfterMs = 0 } = {}) {
  failureCount++;
  console.log(`\n⚠️  Failure ${failureCount}/${MAX_FAILURES}: ${reason}`);
  if (failureCount >= MAX_FAILURES) {
    throw new GiveUp(`Giving up after ${MAX_FAILURES} failures. Last: ${reason}`, 1);
  }

  const ceiling = Math.min(BACKOFF_MAX_MS, BACKOFF_BASE_MS * 2 ** (failureCount - 1));
  let delay = ceiling / 2 + Math.random() * (ceiling / 2); // "equal jitter"
  delay = Math.max(delay, Math.min(retryAfterMs, 10 * 60 * 1000)); // honour Retry-After

  const { registryDown } = await preflight();
  if (registryDown) delay = Math.max(delay, BACKOFF_MAX_MS / 2);

  console.log(`   ⏳ Backing off ${(delay / 1000).toFixed(1)}s before attempt ${failureCount + 1}...`);
  await sleep(delay); // setTimeout is monotonic too
}

// --- Single-instance lock: two pulls of one model are what causes the partial-file race.

const LOCK_FILE = path.join(
  os.tmpdir(),
  `ollama-pull-${`${OLLAMA_HOST}-${model}`.replace(/[^a-zA-Z0-9._-]/g, "_")}.lock`
);
let lockHeld = false;

const pidAlive = (pid) => {
  try {
    process.kill(pid, 0);
    return true;
  } catch (e) {
    return e.code === "EPERM"; // exists but belongs to someone else
  }
};

function acquireLock() {
  for (let i = 0; i < 3; i++) {
    try {
      const fd = fs.openSync(LOCK_FILE, "wx"); // atomic: fails if it already exists
      fs.writeSync(fd, String(process.pid));
      fs.closeSync(fd);
      lockHeld = true;
      return;
    } catch (err) {
      if (err.code !== "EEXIST") throw err;
      let pid = NaN;
      let ageMs = Infinity;
      try {
        pid = parseInt(fs.readFileSync(LOCK_FILE, "utf8"), 10);
        ageMs = Date.now() - fs.statSync(LOCK_FILE).mtimeMs;
      } catch { /* lock vanished between calls: just retry */ }
      // Empty/fresh file = another process is mid-creation, so treat it as held.
      const held = pid ? pidAlive(pid) : ageMs < 5000;
      if (held) {
        throw new GiveUp(
          `Another pull of ${model} is already running (pid ${pid || "?"}). ` +
            `Two pulls of the same model race on the partial files. If that's stale, delete ${LOCK_FILE}`,
          4
        );
      }
      try { fs.unlinkSync(LOCK_FILE); } catch { /* someone else removed it */ }
    }
  }
  throw new GiveUp(`Could not acquire lock ${LOCK_FILE}`, 4);
}

function releaseLock() {
  if (!lockHeld) return;
  lockHeld = false;
  try { fs.unlinkSync(LOCK_FILE); } catch { /* already gone */ }
}

process.on("exit", releaseLock);
for (const sig of ["SIGINT", "SIGTERM", "SIGHUP"]) {
  process.on(sig, () => {
    console.log(`\n\n🛑 Caught ${sig}, shutting down...`);
    releaseLock();
    process.exit(130);
  });
}
for (const evt of ["unhandledRejection", "uncaughtException"]) {
  process.on(evt, (err) => {
    console.error(`\n❌ ${evt}: ${err?.stack || err}\n`);
    releaseLock();
    process.exit(1);
  });
}

// ---------------------------------------------------------------------------
// One pull attempt
// ---------------------------------------------------------------------------

/**
 * Runs a single pull attempt.
 * Returns "success" | "stalled" | "dropped". Throws on errors.
 * `info.progressed` is set to true if any bytes were downloaded this attempt.
 * `baseline.ms` (slowest good band so far) persists across attempts.
 */
async function attemptPull(modelName, baseline, info) {
  const ac = new AbortController();
  const attemptStart = mono();

  const layers = {};
  let stalled = false;
  let finishing = false; // verifying / writing manifest — no byte progress expected
  let lastStatus = "";
  let renderedLines = 0;

  // Monotonic progress tracking (the key to avoiding false stalls):
  //  - per-layer `completed` never decreases
  //  - overall % is high-water-marked
  let knownTotal = 0;
  let maxCompleted = 0;
  let maxPct = 0;
  let lastProgressAt = attemptStart;
  let lastMessageAt = attemptStart; // last time the server said anything at all

  // Band tracking
  let currentBand = null; // set when the first progress data arrives
  let bandStart = attemptStart;
  let bandTainted = false; // layer set changed mid-band, so timing is unreliable
  let firstCrossing = true;
  let resumed = false;

  // Print a standalone message below the layer display and reset the redraw anchor.
  const say = (text) => {
    process.stdout.write(text);
    renderedLines = 0;
  };

  const stall = (reason) => {
    if (stalled) return;
    stalled = true;
    say(`\n\n⚠️  ${reason} Restarting...\n`);
    ac.abort();
  };

  const render = () => {
    const active = Object.entries(layers).filter(([, v]) => v.status !== "already exists");
    if (!active.length) return;
    if (renderedLines > 0) process.stdout.write(`\x1b[${renderedLines}A`);
    for (const [id, l] of active) {
      const pct = l.total ? ((l.completed / l.total) * 100).toFixed(1) : "—";
      const bar = renderBar(l.completed, l.total);
      const size = l.total ? `${formatBytes(l.completed)} / ${formatBytes(l.total)}` : "";
      process.stdout.write(`\x1b[2K  ${id}  ${bar}  ${String(pct).padStart(5)}%  ${size}\n`);
    }
    renderedLines = active.length;
  };

  const tick = () => {
    if (stalled) return;
    const now = mono();

    let total = 0;
    let completed = 0;
    for (const l of Object.values(layers)) {
      total += l.total || 0;
      completed += l.completed || 0;
    }

    if (completed > maxCompleted) {
      maxCompleted = completed;
      lastProgressAt = now;
      info.progressed = true;
    }

    // A new layer was announced: the denominator grew, so % just dropped.
    // Timing for the current band is no longer trustworthy.
    if (total > knownTotal) {
      if (knownTotal > 0 && currentBand !== null) {
        bandTainted = true;
        bandStart = now;
      }
      knownTotal = total;
    }

    // Nothing reported yet (hung connect / no manifest)
    if (knownTotal === 0) {
      if (now - attemptStart > FIRST_BAND_TIMEOUT_MS) {
        stall(`No download progress reported after ${argv.firstBandTimeout}s.`);
      }
      return;
    }

    // Download finished (or in verify/write phase): nothing left to watch.
    // Don't let the idle clock run while we wait for the next layer/status.
    if (finishing || maxCompleted >= knownTotal) {
      lastProgressAt = now;
      // ...but a server that goes completely silent here would hang us forever.
      if (now - lastMessageAt > FINISH_TIMEOUT_MS) {
        stall(`Server silent for ${argv.finishTimeout}s while verifying/finishing at ${maxPct.toFixed(1)}%.`);
      }
      return;
    }

    maxPct = Math.max(maxPct, (maxCompleted / knownTotal) * 100);

    // First real data this attempt: figure out whether we're resuming
    if (currentBand === null) {
      resumed = maxPct > 1;
      currentBand = Math.floor(maxPct / BAND_SIZE);
      bandStart = resumed ? now : attemptStart;
      return;
    }

    // No bytes at all for too long
    if (now - lastProgressAt > FIRST_BAND_TIMEOUT_MS) {
      stall(`No bytes received for ${argv.firstBandTimeout}s at ${maxPct.toFixed(1)}%.`);
      return;
    }

    const band = Math.floor(maxPct / BAND_SIZE);
    const elapsed = now - bandStart;

    if (band > currentBand) {
      const span = band - currentBand; // can be >1 if a big layer jumped us
      const perBandMs = elapsed / span;
      const label = `${currentBand * BAND_SIZE}–${band * BAND_SIZE}%`;
      const skipReason =
        firstCrossing && resumed
          ? "resumed from prior progress"
          : bandTainted
            ? "layer set changed mid-band"
            : null;

      if (skipReason) {
        say(`\n  ⏭  Band ${label} skipped (${skipReason})\n\n`);
      } else {
        const prev = baseline.ms;
        // Never let the baseline get faster.
        baseline.ms = Math.max(prev ?? 0, perBandMs);
        say(
          `\n  ✓ Band ${label} in ${(perBandMs / 1000).toFixed(1)}s ` +
            (prev === null ? "(baseline set)" : `(slowest so far: ${(prev / 1000).toFixed(1)}s)`) +
            "\n\n"
        );
      }

      firstCrossing = false;
      currentBand = band;
      bandStart = now;
      bandTainted = false;
      return;
    }

    // Still inside the same band — check continuously, not just on crossing,
    // otherwise a download that truly hangs would never be detected.
    const limit = baseline.ms !== null ? baseline.ms * STALL_FACTOR : FIRST_BAND_TIMEOUT_MS;
    if (elapsed > limit) {
      const label = `${currentBand * BAND_SIZE}–${(currentBand + 1) * BAND_SIZE}%`;
      stall(
        `Stall detected at ${maxPct.toFixed(1)}%! Band ${label} has taken ${(elapsed / 1000).toFixed(1)}s ` +
          (baseline.ms !== null
            ? `(limit ${(limit / 1000).toFixed(1)}s = ${STALL_FACTOR}x ${(baseline.ms / 1000).toFixed(1)}s).`
            : `(limit ${(limit / 1000).toFixed(0)}s, no baseline yet).`)
      );
    }
  };

  const handleMessage = (msg) => {
    if (msg.error) {
      process.stdout.write("\n");
      throw classifyError(String(msg.error));
    }

    lastMessageAt = mono();
    const { status, digest, total, completed } = msg;

    if (digest) {
      finishing = false;
      const id = digest.slice(-12);
      const prev = layers[id] || { total: 0, completed: 0 };
      layers[id] = {
        status,
        total: typeof total === "number" ? total : prev.total,
        completed: Math.max(prev.completed, typeof completed === "number" ? completed : 0),
      };
      render();
    } else if (status) {
      finishing = /^(verifying|writing|removing|success)/i.test(status);
      if (status !== lastStatus) {
        say(`\x1b[2K\r  ⏳ ${status}\n`);
        lastStatus = status;
      }
    }

    return status === "success";
  };

  // Start the watchdog BEFORE fetch so a hung connection is also caught.
  const stallTimer = setInterval(tick, 1000);
  process.stdout.write("\n");

  try {
    const res = await fetch(`${OLLAMA_HOST}/api/pull`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ name: modelName, stream: true }),
      signal: ac.signal,
    });

    if (!res.ok) {
      const text = await res.text();
      const msg = `Ollama API error ${res.status}: ${text}`;
      // 4xx will never succeed on retry (except 408/425/429); 5xx might.
      const transient = res.status >= 500 || [408, 425, 429].includes(res.status);
      const e = transient ? new Error(msg) : new PullError(msg);
      e.retryAfterMs = parseRetryAfter(res.headers.get("retry-after"));
      throw e;
    }

    const reader = res.body.getReader();
    const decoder = new TextDecoder();
    let buf = "";

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;

      // JSON lines can be split across chunks — buffer until we have full lines.
      buf += decoder.decode(value, { stream: true });
      let nl;
      while ((nl = buf.indexOf("\n")) >= 0) {
        const line = buf.slice(0, nl).trim();
        buf = buf.slice(nl + 1);
        if (!line) continue;

        let msg;
        try { msg = JSON.parse(line); } catch { continue; }
        if (handleMessage(msg)) {
          say(`\n✅ Successfully pulled ${modelName}\n\n`);
          return "success";
        }
      }
    }

    // Flush any trailing line without a newline
    if (buf.trim()) {
      let msg;
      try { msg = JSON.parse(buf.trim()); } catch { /* ignore */ }
      if (msg && handleMessage(msg)) {
        say(`\n✅ Successfully pulled ${modelName}\n\n`);
        return "success";
      }
    }

    if (stalled) return "stalled";

    // Stream ended without a "success" message — never trust pct alone;
    // a re-pull will quickly finish/verify anything that's already downloaded.
    say(`\n\n⚠️  Stream ended at ${maxPct.toFixed(1)}% without a success message. Retrying...\n`);
    return "dropped";
  } catch (err) {
    if (stalled) return "stalled";
    throw err;
  } finally {
    clearInterval(stallTimer);
    ac.abort(); // make sure the connection is closed (no-op if already done)
  }
}

// ---------------------------------------------------------------------------
// Retry loop
// ---------------------------------------------------------------------------

async function pullModel(modelName) {
  const baseline = { ms: null }; // persists across attempts — only ever increases
  let attempt = 0;

  while (true) {
    attempt++;
    console.log(`\n🦙 Pulling model: ${modelName}  (attempt ${attempt})`);
    console.log(`   Host: ${OLLAMA_HOST}`);
    console.log(
      `   Stall detection: ${BAND_SIZE}% bands, restart if ${STALL_FACTOR}x slower than slowest good band, ` +
        `first-band/idle timeout: ${argv.firstBandTimeout}s`
    );

    const info = { progressed: false };
    let result;
    let reason = "";
    let retryAfterMs = 0;

    try {
      result = await attemptPull(modelName, baseline, info);
    } catch (err) {
      if (err instanceof PullError || err instanceof GiveUp) throw err;
      result = "error";
      reason = err.cause?.code ? `${err.message} (${err.cause.code})` : err.message;
      retryAfterMs = err.retryAfterMs || 0;
    }

    if (result === "success") return;

    if (result === "stalled") reason = "Download stalled";
    else if (result === "dropped") reason = "Stream dropped without a success message";

    // Every kind of failure shares one budget (default 9), with backoff + preflight.
    await registerFailure(reason || result, { retryAfterMs });
  }
}

// ---------------------------------------------------------------------------
// Post-pull verification ("hello test") + corrupt blob cleanup
// ---------------------------------------------------------------------------

// Errors that suggest a truncated/corrupt blob. Matching this alone never deletes
// anything: the blob is re-hashed first and only removed if the hash is wrong.
const CORRUPT_RE =
  /exceeds file size|unexpected EOF|invalid file magic|bad magic|invalid gguf|read GGUF metadata|unsupported tensor|truncated|corrupt/i;

/** Loads the model and asks it to say hello. Unloads it again afterwards. */
async function runHelloTest(modelName) {
  const ac = new AbortController();
  const timer = setTimeout(() => ac.abort(), VERIFY_TIMEOUT_MS);
  try {
    const res = await fetch(`${OLLAMA_HOST}/api/generate`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        model: modelName,
        prompt: "Say hello in one short sentence.",
        stream: false,
        keep_alive: 0, // unload right after the test so it doesn't hog memory
        options: { num_predict: 32 },
      }),
      signal: ac.signal,
    });
    const text = await res.text();
    let data;
    try { data = JSON.parse(text); } catch { /* not JSON */ }

    if (!res.ok || data?.error) {
      return { ok: false, error: data?.error || `HTTP ${res.status}: ${text}` };
    }
    return { ok: true, reply: (data?.response || "").trim() };
  } catch (err) {
    if (ac.signal.aborted) {
      return { ok: false, error: `Hello test timed out after ${argv.verifyTimeout}s` };
    }
    return { ok: false, error: err.cause?.code ? `${err.message} (${err.cause.code})` : err.message };
  } finally {
    clearTimeout(timer);
  }
}

/** Streaming sha256 so multi-GB blobs don't get loaded into memory. */
function sha256File(file, label) {
  return new Promise((resolve, reject) => {
    const hash = crypto.createHash("sha256");
    const size = fs.statSync(file).size;
    let read = 0;
    let lastPrint = 0;
    const stream = fs.createReadStream(file, { highWaterMark: 8 * 1024 * 1024 });
    stream.on("data", (chunk) => {
      hash.update(chunk);
      read += chunk.length;
      const now = Date.now();
      if (now - lastPrint > 500) {
        lastPrint = now;
        process.stdout.write(`\r\x1b[2K   hashing ${label}  ${((read / size) * 100).toFixed(0)}%`);
      }
    });
    stream.on("error", reject);
    stream.on("end", () => {
      process.stdout.write("\r\x1b[2K");
      resolve(hash.digest("hex"));
    });
  });
}

/** Path of the local manifest file for a model name like "ornith-1.5:122b". */
function manifestPathFor(modelName) {
  let name = modelName;
  let tag = "latest";
  const colon = name.lastIndexOf(":");
  if (colon > name.lastIndexOf("/")) {
    tag = name.slice(colon + 1);
    name = name.slice(0, colon);
  }
  const parts = name.split("/");
  let host = "registry.ollama.ai";
  let ns = "library";
  if (parts.length >= 3) {
    host = parts[0];
    ns = parts[1];
    name = parts.slice(2).join("/");
  } else if (parts.length === 2) {
    ns = parts[0];
    name = parts[1];
  }
  return path.join(MODELS_DIR, "manifests", host, ns, name, tag);
}

/** Blob file names ("sha256-<hex>") referenced by the model's manifest. */
function manifestBlobs(modelName) {
  try {
    const m = JSON.parse(fs.readFileSync(manifestPathFor(modelName), "utf8"));
    return [m.config, ...(m.layers || [])]
      .filter((l) => l?.digest)
      .map((l) => l.digest.replace(":", "-"));
  } catch {
    return [];
  }
}

/**
 * Re-hashes the blob(s) implicated by a failed hello test and deletes any whose
 * contents don't match their filename (plus their -partial* leftovers).
 * Returns how many bad blobs were removed.
 */
async function cleanupBadBlobs(modelName, errorText) {
  // Prefer the exact path the server printed: it is the dir the server really uses.
  const named = errorText.match(/([^\s"']*\/blobs)\/(sha256-[0-9a-f]{64})/i);
  const blobsDir = named && fs.existsSync(named[1]) ? named[1] : path.join(MODELS_DIR, "blobs");

  if (!fs.existsSync(blobsDir)) {
    console.error(
      `⚠️  Can't find the blobs directory (${blobsDir}). Cleanup needs local access to the models dir; ` +
        `use --models-dir, or delete the bad blob on the server by hand.`
    );
    return 0;
  }

  const candidates = named ? [named[2].toLowerCase()] : manifestBlobs(modelName);
  if (!candidates.length) {
    console.error("⚠️  Couldn't tell which blob is bad (no path in the error, no readable manifest).");
    return 0;
  }

  console.log(
    named
      ? `\n🔍 Checking the blob named in the error: ${named[2].slice(0, 19)}…`
      : `\n🔍 Error doesn't name a blob; checking all ${candidates.length} blobs in the manifest…`
  );

  let removed = 0;
  for (const blob of candidates) {
    const file = path.join(blobsDir, blob);
    const leftovers = fs.readdirSync(blobsDir).filter((f) => f.startsWith(`${blob}-partial`));
    let bad = false;
    let missing = false;

    if (!fs.existsSync(file)) {
      console.log(`   ${blob.slice(7, 19)}  missing (will be re-downloaded)`);
      missing = true;
    } else {
      const actual = await sha256File(file, blob.slice(7, 19));
      if (actual === blob.slice(7)) {
        console.log(`   ${blob.slice(7, 19)}  ✓ hash OK`);
        continue;
      }
      bad = true;
      console.log(`   ${blob.slice(7, 19)}  ✗ hash MISMATCH (got ${actual.slice(0, 12)}…) — deleting`);
    }

    try {
      if (bad) fs.rmSync(file, { force: true });
      for (const f of leftovers) fs.rmSync(path.join(blobsDir, f), { force: true });
    } catch (err) {
      if (err.code === "EACCES" || err.code === "EPERM") {
        throw new Error(
          `No permission to delete from ${blobsDir}. Run this, then rerun the script:\n` +
            `   sudo rm -f ${path.join(blobsDir, blob)}*`
        );
      }
      throw err;
    }
    if (bad || missing || leftovers.length) removed++;
  }
  return removed;
}

// Errors that retrying can never fix: fail fast instead of burning the failure budget.
const NON_RETRYABLE_RE =
  /out of memory|requires more (system )?memory|insufficient|unknown model architecture|unsupported architecture|not supported/i;

async function main() {
  acquireLock();

  while (true) {
    await pullModel(model);

    if (!argv.verify) return;

    console.log(`🧪 Hello test: loading ${model} (up to ${argv.verifyTimeout}s)...`);
    const result = await runHelloTest(model);

    if (result.ok) {
      console.log(
        result.reply
          ? `✅ Hello test passed. Model said: ${result.reply}\n`
          : `✅ Hello test passed (model loaded and generated, no visible text).\n`
      );
      return;
    }

    console.error(`\n❌ Hello test failed: ${result.error}\n`);

    // 1. Looks like a corrupt blob: verify by hashing, delete if bad, re-pull.
    if (CORRUPT_RE.test(result.error)) {
      const removed = await cleanupBadBlobs(model, result.error);
      if (!removed) {
        throw new GiveUp(
          "No corrupt blobs found: the files hash correctly, so re-pulling won't help.\n" +
            "This is more likely an Ollama version/compatibility problem. Try updating Ollama.",
          2
        );
      }
      console.log(`\n♻️  Cleaned up ${removed} blob(s). Re-pulling...`);
      await registerFailure("Corrupt download cleaned up");
      continue;
    }

    // 2. Won't be fixed by retrying (not enough memory, unsupported model...).
    if (NON_RETRYABLE_RE.test(result.error)) {
      throw new GiveUp(
        "This doesn't look like a download problem, so nothing was deleted or retried.\n" +
          "Likely not enough RAM/VRAM for the model + context length, or an Ollama version that's too old.",
        2
      );
    }

    // 3. Server busy / 503 / timeout / runner crashed: back off and try again.
    if (TRANSIENT_RE.test(result.error)) {
      await registerFailure(`Hello test: ${result.error}`);
      continue;
    }

    // 4. Unknown: don't guess, don't delete.
    throw new GiveUp("Unrecognised hello-test error; nothing was deleted or retried.", 2);
  }
}

main().catch((err) => {
  console.error(`\n❌ Error: ${err.message}\n`);
  process.exit(err.exitCode ?? 1);
});
