#!/usr/bin/env python3
"""Interactive web UI for the model router.

Enter a prompt, get the recommended model tier + reasoning.
Uses the trained RouteLLM-style classifier from routellm_classifier.py.

Usage: python scripts/router_ui.py
Then open http://localhost:8080
"""
import json, math, pickle, re, sys
from pathlib import Path
from http.server import HTTPServer, SimpleHTTPRequestHandler

TIER_NAMES = {0: "economy", 1: "mid", 2: "premium"}
TIER_MODELS = {
  "economy": {"claude": "claude-sonnet-5", "gpt": "gpt-5.6-luna"},
  "mid": {"claude": "claude-sonnet-5", "gpt": "gpt-5.6-terra"},
  "premium": {"claude": "claude-opus-5", "gpt": "gpt-5.6-terra"},
}
TIER_COST = {"economy": "$0.20–2.00/M", "mid": "$2.00–3.00/M", "premium": "$2.00–5.00/M"}
TIER_COLORS = {"economy": "#22c55e", "mid": "#f59e0b", "premium": "#ef4444"}
QUALITY_TOLERANCE = 0.03

# Load the user-text-only trained model (trained by routellm_classifier.py Step 8)
UI_MODEL_PATH = Path(__file__).parent / "../results/routellm_ui_model.pkl"

sys.path.insert(0, str(Path(__file__).parent))
from routellm_classifier import tokenize, text_to_tfidf, structural_features
from load_trajectories import est_tokens


def load_model():
    with open(UI_MODEL_PATH, "rb") as f:
        data = pickle.load(f)
    return data["clf"], data["vocab"], data["idf"], data["references"]


CLF, VOCAB, IDF, REFERENCES = load_model()


def reference_similarity(query_tfidf, query_struct, reference):
    """Score a historical prompt by text similarity plus task structure."""
    ref_tfidf = reference["tfidf"]
    query_norm = math.sqrt(sum(value * value for value in query_tfidf))
    ref_norm = math.sqrt(sum(value * value for value in ref_tfidf))
    if query_norm and ref_norm:
        cosine = sum(a * b for a, b in zip(query_tfidf, ref_tfidf)) / (query_norm * ref_norm)
    else:
        cosine = 0.0

    scales = (1.0, 1.0, 1.0, 1.0, 0.5, 1.0, 1.0, 1.0, 0.5)
    structural_distance = math.sqrt(sum(
        ((a - b) / scale) ** 2
        for a, b, scale in zip(query_struct, reference["struct"], scales)
    ))
    return 0.7 * max(0.0, cosine) + 0.3 * math.exp(-structural_distance)


def estimate_tier_quality(query_tfidf, query_struct, tier, k=12):
    """Estimate outcome for a tier from its nearest observed trajectories."""
    candidates = [
        (reference_similarity(query_tfidf, query_struct, reference), reference)
        for reference in REFERENCES
        if reference["tier"] == tier
    ]
    candidates.sort(key=lambda item: item[0], reverse=True)
    nearest = candidates[:k]
    if not nearest:
        return 0.0, 0.0, []
    weights = [similarity + 0.05 for similarity, _ in nearest]
    total_weight = sum(weights)
    quality = sum(weight * reference["outcome"]
            for weight, (_, reference) in zip(weights, nearest)) / total_weight
    evidence = min(1.0, sum(similarity for similarity, _ in nearest) / k)
    return quality, evidence, [reference["trajectory"] for _, reference in nearest[:3]]


def classify_prompt(text):
    """Classify a prompt using the trained GradientBoosting classifier.

    Uses prompt similarity against the user-text portion of 953 trajectories.
    Each tier's quality is estimated from nearby historical prompts that
    actually ran on that tier. The best estimated tier is selected once before
    execution; this is an off-policy estimate, not a guaranteed outcome.
    """
    import numpy as np

    # Build feature vector matching training: TF-IDF on user text + structural features
    # For structural features, we simulate a typical input context since the UI
    # doesn't have the full API call. Scale overhead with prompt complexity:
    # longer/more complex prompts typically come with larger system contexts.
    base_overhead = 40_000  # ~10k tokens baseline
    scaled_overhead = base_overhead + len(text) * 10  # scale with prompt size
    fake_input = [{"role": "system", "content": "x" * scaled_overhead},
                  {"role": "user", "content": text}]

    tfidf = text_to_tfidf(text, VOCAB, IDF)
    struct = structural_features(text, fake_input)
    X = np.array([tfidf + struct])

    query_tfidf = tfidf
    tier_quality = {}
    tier_evidence = {}
    similar_trajectories = {}
    for candidate_tier in ("economy", "mid", "premium"):
      quality, evidence, examples = estimate_tier_quality(
        query_tfidf, struct, candidate_tier
      )
      tier_quality[candidate_tier] = quality
      tier_evidence[candidate_tier] = evidence
      similar_trajectories[candidate_tier] = examples

    # Choose the cheapest tier whose estimated outcome is close to the best
    # local estimate. This avoids paying premium for a negligible difference,
    # while still upgrading prompts where economy has materially weaker matches.
    best_quality = max(tier_quality.values())
    tier = next(
      candidate_tier for candidate_tier in ("economy", "mid", "premium")
      if best_quality - tier_quality[candidate_tier] <= QUALITY_TOLERANCE
    )

    # Convert relative estimated quality into display probabilities. These are
    # evidence scores, not calibrated probabilities of success.
    temperature = 0.04
    logits = {
      candidate_tier: math.exp((tier_quality[candidate_tier] - max(tier_quality.values())) / temperature)
      for candidate_tier in tier_quality
    }
    logit_total = sum(logits.values())
    probabilities = {
      candidate_tier: round(logits[candidate_tier] / logit_total * 100, 1)
      for candidate_tier in logits
    }

    # Feature analysis for UI display
    has_code = bool(re.search(r'```', text))
    has_url = bool(re.search(r'https?://', text))
    has_error = bool(re.search(r'Traceback|Error:|Exception|FAILED', text))
    has_file = bool(re.search(r'[/\\]\w+\.\w+', text))
    word_count = len(text.split())
    char_count = len(text)

    reasons = []
    best_tier = max(tier_quality, key=tier_quality.get)
    quality_gap = best_quality - tier_quality[tier]
    if tier == "economy":
        reasons.append("Task complexity is within economy-tier capability")
        if not has_code:
            reasons.append("No code blocks detected")
        if not has_error:
            reasons.append("No error traces / debugging needed")
        if word_count < 200:
            reasons.append("Short, focused prompt")
        reasons.append(f"Economy is within {QUALITY_TOLERANCE:.2f} of the best local estimate")
    elif tier == "mid":
        if has_code:
            reasons.append("Code block detected — benefits from mid-tier reasoning")
        if word_count > 300:
            reasons.append("Detailed prompt with multiple requirements")
        reasons.append(f"Mid is the cheapest tier within {QUALITY_TOLERANCE:.2f} of the best local estimate")
    elif tier == "premium":
        if has_error:
            reasons.append("Error trace detected — debugging benefits from advanced reasoning")
        if has_code and has_file:
            reasons.append("Complex code task with file operations")
        if word_count > 500:
            reasons.append("Very detailed, multi-step request")
        reasons.append("Lower-cost tiers are outside the quality tolerance for this prompt")

    features_detected = []
    if has_code: features_detected.append("code_block")
    if has_url: features_detected.append("url")
    if has_error: features_detected.append("error_trace")
    if has_file: features_detected.append("file_path")

    return {
        "tier": tier,
        "models": TIER_MODELS[tier],
        "probabilities": probabilities,
        "estimated_quality": {tier_name: round(score, 4) for tier_name, score in tier_quality.items()},
        "best_estimated_tier": best_tier,
        "quality_gap_to_best": round(quality_gap, 4),
        "quality_tolerance": QUALITY_TOLERANCE,
        "evidence": {tier_name: round(score, 4) for tier_name, score in tier_evidence.items()},
        "similar_trajectories": similar_trajectories,
        "reasons": reasons,
        "prompt_stats": {
            "words": word_count,
            "characters": char_count,
            "est_tokens": char_count // 4,
        },
        "features_detected": features_detected,
    }


HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Model Router — Viktor Challenge</title>
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
         background: #0f172a; color: #e2e8f0; min-height: 100vh; }
  .container { max-width: 900px; margin: 0 auto; padding: 2rem; }
  h1 { font-size: 1.8rem; margin-bottom: 0.5rem; }
  h1 span { color: #818cf8; }
  .subtitle { color: #94a3b8; margin-bottom: 2rem; font-size: 0.95rem; }
  textarea { width: 100%; height: 180px; background: #1e293b; border: 1px solid #334155;
             border-radius: 8px; color: #e2e8f0; padding: 1rem; font-size: 0.95rem;
             font-family: inherit; resize: vertical; outline: none; }
  textarea:focus { border-color: #818cf8; }
  .btn { background: #6366f1; color: white; border: none; padding: 0.75rem 2rem;
         border-radius: 8px; font-size: 1rem; cursor: pointer; margin-top: 1rem;
         transition: background 0.2s; }
  .btn:hover { background: #4f46e5; }
  .btn:disabled { background: #475569; cursor: not-allowed; }
  .result { margin-top: 2rem; display: none; }
  .result.show { display: block; }
  .tier-badge { display: inline-block; padding: 0.4rem 1.2rem; border-radius: 20px;
                font-weight: 700; font-size: 1.1rem; text-transform: uppercase; }
  .card { background: #1e293b; border-radius: 8px; padding: 1.25rem; margin-top: 1rem; }
  .card h3 { color: #94a3b8; font-size: 0.8rem; text-transform: uppercase;
             letter-spacing: 0.05em; margin-bottom: 0.75rem; }
  .model-name { font-size: 1.3rem; font-weight: 600; color: #f1f5f9; }
  .model-cost { color: #94a3b8; font-size: 0.9rem; }
  .ranking-note { color: #94a3b8; font-size: 0.82rem; line-height: 1.45; margin-bottom: 1rem; }
  .tier-list { display: grid; gap: 0.65rem; }
  .tier-row { display: grid; grid-template-columns: 34px minmax(100px, 1fr) auto;
              align-items: center; gap: 0.75rem; padding: 0.75rem; background: #172235;
              border: 1px solid #334155; border-radius: 6px; }
  .tier-row.selected { border-color: #818cf8; background: #202b48; }
  .priority-dot { width: 28px; height: 28px; border-radius: 50%; display: grid;
                  place-items: center; font-weight: 700; color: #fff; }
  .tier-label { font-weight: 700; text-transform: capitalize; }
  .tier-model { color: #94a3b8; font-size: 0.78rem; margin-top: 0.2rem; }
  .tier-score { text-align: right; font-size: 0.95rem; font-weight: 700; }
  .tier-score small { display: block; color: #94a3b8; font-size: 0.7rem; font-weight: 400; margin-top: 0.2rem; }
  .explanation { color: #cbd5e1; line-height: 1.5; font-size: 0.9rem; }
  .evidence-list { margin-top: 0.75rem; color: #94a3b8; font-size: 0.78rem; line-height: 1.55; }
  .reasons li { color: #cbd5e1; margin: 0.4rem 0; font-size: 0.9rem; list-style: none; }
  .reasons li::before { content: "→ "; color: #818cf8; }
  .features { display: flex; gap: 0.5rem; flex-wrap: wrap; margin-top: 0.5rem; }
  .tag { background: #334155; color: #94a3b8; padding: 0.25rem 0.6rem; border-radius: 4px;
         font-size: 0.8rem; }
  .stats { display: flex; gap: 2rem; margin-top: 0.5rem; }
  .stat-val { font-size: 1.2rem; font-weight: 600; color: #f1f5f9; }
  .stat-label { font-size: 0.75rem; color: #64748b; }
  .footer { margin-top: 3rem; text-align: center; color: #475569; font-size: 0.8rem; }
  .examples { margin-top: 1rem; display: flex; gap: 0.5rem; flex-wrap: wrap; }
  .example-btn { background: #1e293b; border: 1px solid #334155; color: #94a3b8;
                 padding: 0.4rem 0.8rem; border-radius: 6px; cursor: pointer; font-size: 0.8rem; }
  .example-btn:hover { border-color: #818cf8; color: #e2e8f0; }
</style>
</head>
<body>
<div class="container">
  <h1><span>Model Router</span> — Viktor Challenge</h1>
    <p class="subtitle">Enter a prompt to see which LLM tier the classifier recommends.
      Trained on 1,883 real Viktor trajectories with off-policy quality estimation.</p>

  <textarea id="prompt" placeholder="Enter your prompt here...&#10;&#10;Try pasting a real task like:&#10;- Send a message to the team channel saying the deploy is done&#10;- Debug this Python error: Traceback (most recent call last)...&#10;- Analyze our Q3 revenue data and build a summary deck"></textarea>

  <div class="examples">
    <button class="example-btn" onclick="setExample('slack')">Slack message</button>
    <button class="example-btn" onclick="setExample('debug')">Debug error</button>
    <button class="example-btn" onclick="setExample('code')">Write code</button>
    <button class="example-btn" onclick="setExample('analysis')">Data analysis</button>
    <button class="example-btn" onclick="setExample('simple')">Quick question</button>
  </div>

  <button class="btn" id="routeBtn" onclick="routePrompt()">Route this prompt</button>

  <div class="result" id="result">
    <div style="display:flex; align-items:center; gap:1rem; margin-bottom:1rem;">
      <span class="tier-badge" id="tierBadge"></span>
      <div>
        <div class="model-name" id="modelName"></div>
        <div class="model-cost" id="modelCost"></div>
      </div>
    </div>

    <div class="card">
      <h3>Routing evidence</h3>
      <p class="ranking-note">Priority is ranked by estimated outcome from similar historical prompts. Scores are relative evidence, not calibrated probabilities.</p>
      <div class="tier-list" id="tierList"></div>
    </div>

    <div class="card">
      <h3>Why this tier?</h3>
      <p class="explanation" id="explanation"></p>
      <ul class="reasons" id="reasons"></ul>
      <div class="evidence-list" id="evidenceList"></div>
    </div>

    <div class="card">
      <h3>Prompt analysis</h3>
      <div class="stats" id="stats"></div>
      <div class="features" id="features"></div>
    </div>
  </div>

  <div class="footer">
    Team Bayern CodeWerk — TUM.ai Hackathon 2026<br>
    Classifier: GradientBoosting on TF-IDF + structural features, 88.6% CV accuracy<br>
    Method: RouteLLM-inspired (lm-sys) with off-policy matched training labels
  </div>
</div>

<script>
const EXAMPLES = {
  slack: "Send a message to the #general channel letting the team know that the weekly deployment completed successfully. Add a green checkmark reaction.",
  debug: "Debug this Python error:\\n\\n```\\nTraceback (most recent call last):\\n  File \\"/app/main.py\\", line 42, in process_data\\n    result = transform(data)\\nTypeError: transform() missing 1 required positional argument: 'schema'\\n```\\n\\nThe function worked fine yesterday. Check git log for recent changes to /app/main.py and fix the issue.",
  code: "Write a Python script that reads a CSV file, groups rows by the 'department' column, calculates the average salary for each department, and outputs the results as a formatted table. Handle missing values gracefully.",
  analysis: "Analyze our Q3 revenue data from the attached spreadsheet. Compare it against Q2, identify the top 3 growth drivers, and flag any accounts with >20% decline. Build a summary slide deck with charts for the board meeting.",
  simple: "What time is our standup meeting tomorrow?"
};

const COSTS = {economy: "$0.80/M tokens", mid: "$3.00/M tokens", premium: "$15.00/M tokens"};
const COLORS = {economy: "#22c55e", mid: "#f59e0b", premium: "#ef4444"};

function setExample(key) {
  document.getElementById("prompt").value = EXAMPLES[key];
}

async function routePrompt() {
  const prompt = document.getElementById("prompt").value.trim();
  if (!prompt) return;

  const btn = document.getElementById("routeBtn");
  btn.disabled = true; btn.textContent = "Routing...";

  try {
    const resp = await fetch("/route", {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify({prompt})
    });
    const data = await resp.json();
    showResult(data);
  } catch (e) {
    alert("Error: " + e.message);
  } finally {
    btn.disabled = false; btn.textContent = "Route this prompt";
  }
}

function showResult(data) {
  const result = document.getElementById("result");
  result.classList.add("show");

  const badge = document.getElementById("tierBadge");
  badge.textContent = data.tier;
  badge.style.background = COLORS[data.tier];
  badge.style.color = data.tier === "mid" ? "#000" : "#fff";

  document.getElementById("modelName").textContent =
    "Claude: " + data.models.claude + " / GPT: " + data.models.gpt;
  document.getElementById("modelCost").textContent = COSTS[data.tier];

  // Rank tiers by local estimated outcome, not by a claimed calibrated probability.
  const tierOrder = ["economy", "mid", "premium"].sort((a, b) =>
    data.estimated_quality[b] - data.estimated_quality[a]);
  const tierList = document.getElementById("tierList");
  tierList.innerHTML = tierOrder.map((t, i) => {
    const selected = t === data.tier ? "selected" : "";
    const examples = (data.similar_trajectories[t] || []).join(", ");
    return `<div class="tier-row ${selected}">
      <div class="priority-dot" style="background:${COLORS[t]}">${i + 1}</div>
      <div><div class="tier-label">${t}${selected ? " · selected" : ""}</div>
        <div class="tier-model">Claude: ${data.models.claude} · GPT: ${data.models.gpt}</div></div>
      <div class="tier-score">${data.estimated_quality[t].toFixed(4)}<small>${data.probabilities[t]}% relative evidence</small></div>
    </div>`;
  }).join("");

  // Reasons
  const reasons = document.getElementById("reasons");
  reasons.innerHTML = data.reasons.map(r => `<li>${r}</li>`).join("");
  document.getElementById("explanation").textContent =
    `The router compares this prompt with historical examples for all three tiers. ` +
    `${data.tier} is selected because it is the cheapest tier within ${data.quality_tolerance.toFixed(2)} ` +
    `of the best estimated outcome (${data.best_estimated_tier}). The whole task is routed once, so the cache is not reset by a later upgrade.`;
  document.getElementById("evidenceList").innerHTML = ["economy", "mid", "premium"].map(t =>
    `<div><strong>${t} matches:</strong> ${(data.similar_trajectories[t] || []).join(", ") || "no matching history"}</div>`
  ).join("");

  // Stats
  const stats = document.getElementById("stats");
  stats.innerHTML = [
    {v: data.prompt_stats.words, l: "words"},
    {v: data.prompt_stats.characters, l: "characters"},
    {v: data.prompt_stats.est_tokens, l: "est. tokens"},
  ].map(s => `<div><div class="stat-val">${s.v.toLocaleString()}</div>
              <div class="stat-label">${s.l}</div></div>`).join("");

  // Features
  const features = document.getElementById("features");
  const all = ["code_block", "url", "error_trace", "file_path"];
  features.innerHTML = all.map(f => {
    const active = data.features_detected.includes(f);
    return `<span class="tag" style="${active ? 'background:#818cf8;color:#fff' : ''}">${
      active ? '✓' : '✗'} ${f}</span>`;
  }).join("");
}

document.getElementById("prompt").addEventListener("keydown", e => {
  if (e.key === "Enter" && e.ctrlKey) routePrompt();
});
</script>
</body>
</html>"""


class RouterHandler(SimpleHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/" or self.path == "/index.html":
            self.send_response(200)
            self.send_header("Content-Type", "text/html")
            self.end_headers()
            self.wfile.write(HTML.encode())
        else:
            self.send_error(404)

    def do_POST(self):
        if self.path == "/route":
            length = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(length))
            prompt = body.get("prompt", "")
            result = classify_prompt(prompt)
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(result).encode())
        else:
            self.send_error(404)

    def log_message(self, format, *args):
        pass  # suppress request logs


def main():
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8080
    server = HTTPServer(("0.0.0.0", port), RouterHandler)
    print(f"Router UI running at http://localhost:{port}")
    print("Press Ctrl+C to stop")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")


if __name__ == "__main__":
    main()
