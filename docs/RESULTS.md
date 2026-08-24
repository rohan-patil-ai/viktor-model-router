# Results

![Cost comparison](results_comparison.svg)

![Initial router UI](router-ui-initial.png)

![Explainable routed result](router-ui.png)

Evaluation uses the expanded local export: 2,000 request rows reconstructed into 1,883 trajectories.

The three cost numbers answer different questions: **logged routing** is what
actually happened in the export, the **starter baseline** is the original
short-trajectory heuristic, and **our router** uses learned prompt features plus
similar historical outcomes.

| Strategy | Estimated input cost | Change vs logged |
| --- | ---: | ---: |
| Logged routing | $167.67 | baseline |
| GitHub starter baseline | $147.23 | -12.2% |
| RouteLLM-style classifier | $64.59 | -61.5% |

The classifier route distribution is 1,479 economy, 327 mid, and 77 premium trajectories.

The UI exposes the decision instead of returning only a model name: it shows
the three tier estimates, ranked priority, model candidates, quality gap to the
best tier, quality tolerance, and the historical trajectory IDs used as evidence.

The logged data's heuristic outcome averages are:

- Claude Fable 5: 0.7707, n=120
- Claude Opus 5: 0.7620, n=623
- Claude Sonnet 5: 0.7719, n=501
- GPT 5.6 Luna: 0.7118, n=42
- GPT 5.6 Sol: 0.7219, n=194
- GPT 5.6 Terra: 0.7129, n=261

Overall, 1,079 of 19,143 recovered function outputs contain an error indicator: 5.6%.

## Interpretation

The router is substantially cheaper than both logged routing and the starter baseline. The result is not a randomized quality comparison. The quality estimates are off-policy and rely on similarity matching and a heuristic outcome signal. The appropriate claim is cost reduction with evidence-backed, risk-aware routing, not guaranteed equal quality.

The chart in `docs/results_comparison.svg` is a cost comparison. It does not claim that estimated quality is measured counterfactually.
