 # Viktor Model Router

 An offline, prompt-based model router for the Viktor challenge. It compares a
 new prompt with similar historical trajectories, estimates the outcome for each
 model tier, and chooses one model before execution.

 ## Run the UI

 ```bash
 python3 scripts/router_ui.py 8080
 ```

 Open <http://localhost:8080>.

 The trained artifact is `results/routellm_ui_model.pkl`. To retrain it from the
 local challenge export:

 ```bash
 python3 scripts/routellm_classifier.py export/
 ```

 The export is challenge-only and is intentionally ignored by git.

 ## Routing method

 - TF-IDF features from the user prompt plus structural features
 - Gradient Boosting classifier trained on 1,883 reconstructed trajectories
 - Nearest historical prompts for economy, mid, and premium outcome estimates
 - Cheapest tier accepted when its estimated outcome is within `0.03` of the best
 - One whole-trajectory decision, avoiding mid-task switches and cache resets

 Token counts use the repository's `chars / 4` estimate. Prices are assumptions
 for anonymized model IDs, and counterfactual quality is estimated from matched
 trajectories rather than rerun on each model.
 # Viktor Model Router

 An offline, prompt-based model router for the Viktor challenge. It compares a
 new prompt with similar historical trajectories, estimates the outcome for each
 model tier, and chooses one model before execution.

 ## Run the UI

 ```bash
 python3 scripts/router_ui.py 8080
 ```

 Open <http://localhost:8080>.

 The trained artifact is `results/routellm_ui_model.pkl`. To retrain it from the
 local challenge export:

 ```bash
 python3 scripts/routellm_classifier.py export/
 ```

 The export is challenge-only and is intentionally ignored by git.

 ## Routing method

 - TF-IDF features from the user prompt plus structural features
- Gradient Boosting classifier trained on 1,883 reconstructed trajectories
 - Nearest historical prompts for economy, mid, and premium outcome estimates
 - Cheapest tier accepted when its estimated outcome is within `0.03` of the best
 - One whole-trajectory decision, avoiding mid-task switches and cache resets

 Token counts use the repository's `chars / 4` estimate. Prices are assumptions
 for anonymized model IDs, and counterfactual quality is estimated from matched
 trajectories rather than rerun on each model.

