#!/usr/bin/env python3
"""RouteLLM-inspired classifier router trained on our trajectory data.

Approach (adapted from lm-sys/RouteLLM paper):
- RouteLLM routes between "strong" and "weak" models using a classifier that
  predicts P(strong wins) for each prompt, then thresholds it.
- We extend this to 3 tiers (premium/mid/economy) using our off-policy matching
  to construct training labels: for each trajectory, estimate how much each tier
  would have helped via matched comparisons.

Features: TF-IDF on prompt text + structural features (code blocks, URLs, etc.)
Model: sklearn gradient boosted classifier
Labels: premium_benefit estimated from off-policy matching

Usage: python scripts/routellm_classifier.py export/
"""
import json, sys, math, re
from pathlib import Path
from collections import defaultdict, Counter
from load_trajectories import iter_requests, group_trajectories, est_tokens, first_user_text
from cost_model import trajectory_cost, logged_route, load_pricing
from outcome_signal import outcome_score

TIER_MAP = {
    "claude-opus-5": "premium", "claude-opus-4-8": "premium", "claude-opus-4-6": "premium",
    "claude-sonnet-5": "mid", "claude-sonnet-4-6": "mid",
    "gpt-5.6-terra": "mid", "gpt-5.6-luna": "mid",
    "claude-fable-5": "economy", "gpt-5.6-sol": "economy",
}


def model_family(model):
    return "claude" if model.startswith("claude") else "gpt"


def economy_model(logged):
    return "claude-sonnet-5" if model_family(logged) == "claude" else "gpt-5.6-luna"


def mid_model(logged):
    return "claude-sonnet-5" if model_family(logged) == "claude" else "gpt-5.6-terra"


def premium_model(logged):
    return "claude-opus-5" if model_family(logged) == "claude" else "gpt-5.6-terra"


# ── Text feature extraction ──────────────────────────────────────────────

def tokenize(text):
    """Simple whitespace + punctuation tokenizer."""
    return re.findall(r'\b\w+\b', text.lower())


def build_vocab(texts, max_features=500, min_df=3):
    """Build vocabulary from corpus — top features by document frequency."""
    df = Counter()
    for text in texts:
        tokens = set(tokenize(text))
        for t in tokens:
            df[t] += 1
    # Filter by min document frequency, take top N
    vocab = {word: idx for idx, (word, count) in enumerate(df.most_common(max_features))
             if count >= min_df}
    return vocab


def text_to_tfidf(text, vocab, idf):
    """Convert text to TF-IDF vector."""
    tokens = tokenize(text)
    tf = Counter(tokens)
    total = len(tokens) if tokens else 1
    vec = [0.0] * len(vocab)
    for word, idx in vocab.items():
        if word in tf:
            vec[idx] = (tf[word] / total) * idf.get(word, 1.0)
    return vec


def compute_idf(texts, vocab):
    """Compute inverse document frequency."""
    n = len(texts)
    df = Counter()
    for text in texts:
        tokens = set(tokenize(text))
        for t in tokens:
            if t in vocab:
                df[t] += 1
    return {word: math.log(n / (1 + df.get(word, 0))) for word in vocab}


def structural_features(text, first_input):
    """Non-text features from the prompt."""
    return [
        len(text) / 10_000,                                        # char count
        len(text.split()) / 1_000,                                 # word count
        1.0 if '```' in text else 0.0,                             # code block
        1.0 if re.search(r'https?://', text) else 0.0,             # URL
        1.0 if re.search(r'[/\\]\w+\.\w+', text) else 0.0,        # file path
        1.0 if 'Traceback' in text or 'Error:' in text else 0.0,   # error trace
        text.count('?') / 10,                                       # questions
        text.count('\n') / 100,                                     # newlines
        est_tokens(first_input) / 100_000,                          # input size
    ]


# ── Off-policy matching for training labels ──────────────────────────────

def compute_premium_benefit(traj_data):
    """For each trajectory, estimate benefit of premium via nearest-neighbor matching.
    Returns {key: float} where positive = premium helped."""
    # Use structural features for matching (faster than text)
    keys = list(traj_data.keys())
    vecs = {k: traj_data[k]["struct_features"] for k in keys}

    def dist(v1, v2):
        return math.sqrt(sum((a - b) ** 2 for a, b in zip(v1, v2)))

    benefits = {}
    for key_a in keys:
        tier_a = traj_data[key_a]["tier"]
        out_a = traj_data[key_a]["outcome"]
        vec_a = vecs[key_a]

        # Find 5 nearest on different tiers
        dists = []
        for key_b in keys:
            if key_a == key_b or traj_data[key_b]["tier"] == tier_a:
                continue
            d = dist(vec_a, vecs[key_b])
            dists.append((d, key_b))
        dists.sort()

        premium_outs = []
        economy_outs = []
        for _, kb in dists[:10]:
            t = traj_data[kb]["tier"]
            if t == "premium":
                premium_outs.append(traj_data[kb]["outcome"])
            elif t == "economy":
                economy_outs.append(traj_data[kb]["outcome"])

        if tier_a == "premium" and economy_outs:
            benefits[key_a] = out_a - sum(economy_outs) / len(economy_outs)
        elif tier_a == "economy" and premium_outs:
            benefits[key_a] = sum(premium_outs) / len(premium_outs) - out_a
        elif premium_outs and economy_outs:
            benefits[key_a] = (sum(premium_outs) / len(premium_outs) -
                               sum(economy_outs) / len(economy_outs))
        else:
            benefits[key_a] = 0.0

    return benefits


def main():
    export = sys.argv[1] if len(sys.argv) > 1 else "export"
    pricing = load_pricing()
    groups = group_trajectories(r for _, _, r in iter_requests(export))

    # ── Step 1: Extract text and features from all trajectories ──────────
    print("Step 1: Extracting features from prompts...")
    traj_data = {}
    user_texts = {}

    for key, calls in groups.items():
        first = calls[0]
        user_text = first_user_text(first)
        sys_text = ""
        for item in first["input"]:
            if item.get("role") == "system":
                c = item.get("content", "")
                sys_text = c if isinstance(c, str) else json.dumps(c)
                break

        # Combine system + user text for classification
        full_text = sys_text + " " + user_text
        out = outcome_score(calls)

        traj_data[key] = {
            "calls": calls,
            "tier": TIER_MAP.get(first["model"], "?"),
            "outcome": out["outcome_score"],
            "logged_model": first["model"],
            "struct_features": structural_features(full_text, first["input"]),
        }
        user_texts[key] = full_text

    # ── Step 2: Build TF-IDF vocabulary ──────────────────────────────────
    print("Step 2: Building TF-IDF vocabulary...")
    texts = list(user_texts.values())
    vocab = build_vocab(texts, max_features=300, min_df=5)
    idf = compute_idf(texts, vocab)
    print(f"  Vocabulary size: {len(vocab)}")

    # ── Step 3: Compute training labels via off-policy matching ──────────
    print("Step 3: Computing premium benefit via off-policy matching...")
    benefits = compute_premium_benefit(traj_data)

    # These are routing-policy labels, not measured model-quality boundaries.
    # The available logs cannot identify an exact causal quality threshold.
    # The bands are provisional targets used to train the classifier and must
    # be reported alongside the matching limitations.
    labels = {}
    for key, benefit in benefits.items():
        if benefit > 0.10:
            labels[key] = 2  # premium
        elif benefit > 0.05:
            labels[key] = 1  # mid
        else:
            labels[key] = 0  # economy

    label_counts = Counter(labels.values())
    print(f"  Labels: economy={label_counts[0]}, mid={label_counts[1]}, premium={label_counts[2]}")

    # ── Step 4: Build feature matrix ─────────────────────────────────────
    print("Step 4: Building feature matrix (TF-IDF + structural)...")
    keys = list(traj_data.keys())
    X = []
    y = []
    for key in keys:
        tfidf = text_to_tfidf(user_texts[key], vocab, idf)
        struct = traj_data[key]["struct_features"]
        X.append(tfidf + struct)
        y.append(labels[key])

    # ── Step 5: Train classifier ─────────────────────────────────────────
    print("Step 5: Training gradient boosted classifier...")
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.model_selection import cross_val_score
    import numpy as np

    X = np.array(X)
    y = np.array(y)

    class_counts = Counter(y)
    sample_weights = np.array([
        len(y) / (len(class_counts) * class_counts[label])
        for label in y
    ])

    # Train with cross-validation to measure quality
    clf = GradientBoostingClassifier(
        n_estimators=100,
        max_depth=3,
        learning_rate=0.1,
        random_state=42,
    )

    cv_scores = cross_val_score(clf, X, y, cv=5, scoring="accuracy")
    print(f"  Cross-validation accuracy: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")

    # Train on full data
    clf.fit(X, y, sample_weight=sample_weights)

    # Feature importance — top 20
    feature_names = list(vocab.keys()) + [
        "char_count", "word_count", "has_code_block", "has_url",
        "has_file_path", "has_error_trace", "question_marks",
        "newline_count", "input_tokens",
    ]
    importances = clf.feature_importances_
    top_features = sorted(zip(feature_names, importances), key=lambda x: -x[1])[:20]

    print(f"\n=== TOP 20 FEATURES ===")
    for name, imp in top_features:
        bar = "█" * int(imp * 200)
        print(f"  {name:<25s} {imp:.4f} {bar}")

    # ── Step 6: Predict and build routes ─────────────────────────────────
    print(f"\n=== ROUTING WITH TRAINED CLASSIFIER ===")
    predictions = clf.predict(X)
    # Also get probabilities for the strong_win_rate (RouteLLM style)
    probas = clf.predict_proba(X)

    Path("results").mkdir(exist_ok=True)
    out_f = open("results/routellm_routes.jsonl", "w")
    total_logged = total_routed = 0.0
    tier_counts = Counter()
    tier_names = {0: "economy", 1: "mid", 2: "premium"}

    for i, key in enumerate(keys):
        data = traj_data[key]
        calls = data["calls"]
        logged = calls[0]["model"]
        pred = predictions[i]
        tier = tier_names[pred]
        tier_counts[tier] += 1

        # RouteLLM-style: strong_win_rate = P(premium)
        # We have 3 classes, so combine P(premium) + P(mid) as "strength needed"
        strong_win_rate = float(probas[i][2]) if probas.shape[1] > 2 else 0.0
        mid_rate = float(probas[i][1]) if probas.shape[1] > 1 else 0.0

        if tier == "economy":
            m = economy_model(logged)
        elif tier == "mid":
            m = mid_model(logged)
        else:
            m = premium_model(logged)

        route = [m] * len(calls)
        c_logged, _ = trajectory_cost(calls, logged_route(calls), pricing)
        c_routed, _ = trajectory_cost(calls, route, pricing)
        total_logged += c_logged
        total_routed += c_routed

        out_f.write(json.dumps({
            "trajectory": key,
            "n_calls": len(calls),
            "logged_model": logged,
            "logged_tier": data["tier"],
            "routed_model": m,
            "routed_tier": tier,
            "route": route,
            "strong_win_rate": round(strong_win_rate, 4),
            "mid_rate": round(mid_rate, 4),
            "economy_rate": round(float(probas[i][0]), 4),
            "premium_benefit": round(benefits.get(key, 0.0), 4),
            "outcome_score": data["outcome"],
            "cost_logged_usd": round(c_logged, 6),
            "cost_routed_usd": round(c_routed, 6),
            "switches": 0,
        }) + "\n")
    out_f.close()

    delta = (total_routed / total_logged - 1) * 100
    print(f"  Logged cost:  ${total_logged:,.2f}")
    print(f"  Routed cost:  ${total_routed:,.2f} ({delta:+.1f}%)")
    print(f"  Tier split:   {dict(tier_counts)}")
    print(f"  wrote results/routellm_routes.jsonl")

    # ── Step 7: Threshold sweep (RouteLLM style) ─────────────────────────
    print(f"\n=== THRESHOLD SWEEP (RouteLLM-style cost–quality frontier) ===")
    print(f"  threshold = minimum P(premium) to route to premium\n")
    print(f"  {'Threshold':>10s}  {'Cost':>10s}  {'Delta':>8s}  {'Premium':>8s}  {'Mid':>6s}  {'Economy':>8s}")
    print("  " + "-" * 55)

    for thresh in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
        t_cost = 0.0
        tc = Counter()
        for i, key in enumerate(keys):
            data = traj_data[key]
            calls = data["calls"]
            logged = calls[0]["model"]
            p_premium = float(probas[i][2]) if probas.shape[1] > 2 else 0.0
            p_mid = float(probas[i][1]) if probas.shape[1] > 1 else 0.0

            if p_premium >= thresh:
                m = premium_model(logged)
                tc["premium"] += 1
            elif p_mid >= thresh:
                m = mid_model(logged)
                tc["mid"] += 1
            else:
                m = economy_model(logged)
                tc["economy"] += 1

            route = [m] * len(calls)
            c, _ = trajectory_cost(calls, route, pricing)
            t_cost += c

        d = (t_cost / total_logged - 1) * 100
        print(f"  {thresh:>10.1f}  ${t_cost:>9.2f}  {d:>+7.1f}%  "
              f"{tc.get('premium', 0):>8d}  {tc.get('mid', 0):>6d}  {tc.get('economy', 0):>8d}")

    # Save the model for reuse
    import pickle
    model_path = Path("results/routellm_model.pkl")
    with open(model_path, "wb") as f:
        pickle.dump({"clf": clf, "vocab": vocab, "idf": idf}, f)
    print(f"\n  Saved trained model to {model_path}")

    # ── Step 8: Train a user-text-only model for the UI ────────────────
    # The full model uses system+user text, but the UI only receives bare
    # user prompts. We retrain on user-text-only so the vocab and features
    # match what the UI will see at inference time.
    print(f"\n=== TRAINING USER-TEXT-ONLY MODEL (for UI) ===")
    ui_texts = {}
    for key, calls in groups.items():
        ui_texts[key] = first_user_text(calls[0])

    ui_corpus = list(ui_texts.values())
    ui_vocab = build_vocab(ui_corpus, max_features=300, min_df=3)
    ui_idf = compute_idf(ui_corpus, ui_vocab)
    print(f"  UI vocabulary size: {len(ui_vocab)}")

    X_ui = []
    y_ui = []
    ui_references = []
    for key in keys:
        tfidf = text_to_tfidf(ui_texts[key], ui_vocab, ui_idf)
        # Structural features on user text only, with original input for token count
        struct = structural_features(ui_texts[key], traj_data[key]["calls"][0]["input"])
        X_ui.append(tfidf + struct)
        y_ui.append(labels[key])
        ui_references.append({
            "trajectory": key,
            "tfidf": tfidf,
            "struct": struct,
            "tier": traj_data[key]["tier"],
            "outcome": traj_data[key]["outcome"],
        })

    X_ui = np.array(X_ui)
    y_ui = np.array(y_ui)
    ui_class_counts = Counter(y_ui)
    ui_sample_weights = np.array([
        len(y_ui) / (len(ui_class_counts) * ui_class_counts[label])
        for label in y_ui
    ])

    clf_ui = GradientBoostingClassifier(
        n_estimators=100, max_depth=3, learning_rate=0.1, random_state=42,
    )
    cv_ui = cross_val_score(clf_ui, X_ui, y_ui, cv=5, scoring="accuracy")
    print(f"  UI model CV accuracy: {cv_ui.mean():.3f} ± {cv_ui.std():.3f}")
    clf_ui.fit(X_ui, y_ui, sample_weight=ui_sample_weights)

    ui_model_path = Path("results/routellm_ui_model.pkl")
    with open(ui_model_path, "wb") as f:
        pickle.dump({
            "clf": clf_ui,
            "vocab": ui_vocab,
            "idf": ui_idf,
            "references": ui_references,
        }, f)
    print(f"  Saved UI model to {ui_model_path}")


if __name__ == "__main__":
    main()
