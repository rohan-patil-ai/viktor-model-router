# Methodology

## Goal

Choose one model tier for a complete trajectory before execution. The router is designed to reduce input-token cost without switching models mid-task, which would invalidate the shared-prefix cache.

## Data

The local challenge export contains request histories, available tools, and the model that actually served each request. It contains no final output, usage, quality label, or trajectory ID. The loader reconstructs trajectories by grouping requests with the same opening context and ordering them by input length.

The expanded evaluation uses 2,000 request rows and 1,883 reconstructed trajectories.

## Features

The router uses only information available before execution:

- TF-IDF representation of the user prompt
- Prompt length, word count, newlines, and estimated input tokens
- Code blocks, URLs, file paths, and error traces
- Historical tier and outcome of similar prompts

## Outcome signal

Because the export has no quality labels, the implementation builds a heuristic outcome score from recovered intermediate tool results:

- Error-free tool outputs: 40%
- Completion signal: 20%
- Tool-use efficiency: 20%
- Conversation brevity: 20%

This is a useful operational proxy, not ground-truth quality.

## Routing policy

For each prompt, the router finds similar historical prompts that actually ran on economy, mid, and premium tiers. It computes an inverse-distance weighted outcome estimate for each tier.

It chooses the cheapest tier whose estimated outcome is within `0.03` of the best tier estimate. This tolerance is a policy choice, not a causal quality threshold. It prevents paying for a more expensive tier when the observed local difference is small, while retaining upgrades when cheaper historical matches are materially weaker.

The full trajectory is assigned one model before any call is made. There is no fallback or mid-task upgrade, so there is no cache reset caused by routing.

## ML component

The RouteLLM-style classifier is a Gradient Boosting model trained on TF-IDF and structural features. Its labels are generated from nearest-neighbor off-policy comparisons. The UI also stores the historical prompt references and uses them directly for per-tier quality estimates.

The classifier's cross-validation score measures agreement with these constructed labels. It does not measure whether a different model would actually produce a better answer.

## Cost model

Prices are read from `scripts/pricing.json`. Token counts are estimated as serialized characters divided by four. Cached input is charged at the cached rate only when consecutive calls use the same model. Output cost is excluded because outputs and usage are absent from the export.

## Limitations

- Selection bias: model assignment in the log was not random.
- Unobserved confounding: intent, urgency, and hidden context are unavailable.
- The outcome signal is heuristic.
- Similarity estimates weaken for rare task types.
- Prices and anonymized model capability tiers are assumptions from the challenge context.
- Counterfactual results should be validated with live or judge-model reruns before production use.
