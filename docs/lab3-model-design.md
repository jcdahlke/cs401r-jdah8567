# Lab 3 — Model Development: Design and Evaluation

**Tracks:** Track A (Churn Prediction) + **Track B (Offer Generation with RAG)**

---

## Part 1 — Churn Prediction Model (Track A)

The full Task 1 evaluation report is in **[`lab3-evaluation-report.md`](lab3-evaluation-report.md)**. It covers the setup, the metrics table, the baseline comparison with its confidence interval, the MLflow experiment, the confusion matrix, feature importance and the tier slices.

| Metric (test set, n = 3,000) | Value | Threshold |
|---|---|---|
| AUC-ROC | 0.7840 | reported |
| Precision @ top 10% | 0.7267 | ≥ 0.50 ✅ |
| Recall @ top 10% | 0.3303 | ≥ 0.25 ✅ |
| Lift over recency-only baseline (0.7223) | +0.0617, 95% CI [0.0416, 0.0795] | CI excludes 0 ✅ |

Final model: XGBoost 3.2.0, `max_depth=3`, `eta=0.05`, `scale_pos_weight=3.545`. Registered as `northstar-churn-models` v2, `PendingManualApproval`.

---

## Part 2 — LLM System (Track B: Offer Generation with RAG)

### 2.1 Pipeline design

Code: [`rag/offer_rag.py`](../rag/offer_rag.py). Evaluation runner: [`rag/run_eval.py`](../rag/run_eval.py).
Demo: [`notebooks/lab3-demo.ipynb`](../notebooks/lab3-demo.ipynb).

| Component | Choice | Why |
|---|---|---|
| Corpus | `northstar-policy-docs/` (return, loyalty, shipping, FAQ) | The four policy documents an offer must not contradict |
| Chunking | Split on markdown headers `#`/`##`/`###` (LangChain `MarkdownHeaderTextSplitter`, headers kept in the chunk). **No fixed size and no overlap**: 48 chunks of 164–1,489 characters | Each policy section states one rule, so one section is one retrievable unit. Fixed 500-character windows would cut the §4 tier-benefits table (745 characters) and the §5.1 retention-offer rules in half. Overlap exists to repair cuts, and header splits don't make any. Chunk IDs (`loyalty-program-terms#9`) make every offer traceable to a section |
| Embeddings | Amazon Titan Text Embeddings v2, 1024-d, normalised | Already enabled on the account with no extra model-access request (Cohere needs one); 8k-token input covers the largest chunk; normalised vectors make inner product equal cosine. All 48 chunks embed in about 10 s |
| Vector store | FAISS `IndexFlatIP`, in memory | 48 vectors don't need an approximate index or a managed service. Exact search is instant and free, whereas OpenSearch Serverless has a standing hourly cost |
| Retrieval | Top 6; the query is `"<tier> tier loyalty benefits. <request>"` | Putting the tier in the query pulls in the benefits table. In practice `loyalty-program-terms#9` (§4 Tier Benefits) ranked first for every test case |
| Re-ranking | **None** | With 48 chunks and the key chunk already ranked first, a cross-encoder re-ranker adds a model call and latency for nothing. The failures I observed (Part 2.4) happen *after* retrieval succeeded, so re-ranking wouldn't fix them |
| Generator | Claude Haiku 4.5 via the `us.` cross-region inference profile, temperature 0.2 (no `top_p`), 2–3 sentences | Short, grounded copy is well within Haiku's ability, at a fraction of Sonnet's cost per offer. The lab measured Sonnet as generator adding only about 0.1 faithfulness |
| Grounding for the judge | The customer profile is passed as one extra `contexts` entry | Otherwise the judge marks every profile fact (tier, top category) as unsupported. Measured change 1 in 2.3 |
| Tier card (final pipeline) | `tier_benefit_card()` parses the customer's own column out of the §4 benefits table at run time. It goes into the prompt as the authoritative values and into `contexts` for the judge | Fixes Failure 1, where the model read values from the wrong column of the five-column table. Because it is parsed from the policy file rather than hand-written, it can't drift the way the starter template did. Measured change 2 in 2.3 |

**Starter-template errors.** Template 3's tier-guidelines block in `prompt_templates/offer_generation_prompts.md` disagrees with POL-LOY-011 in three places:
- Gold was listed as $2,000–$9,999.99; the policy says $2,000–$4,999.99.
- Platinum was listed as $10,000+; the policy says $5,000+.
- Gold was credited with "free expedited shipping on orders over $200", which no tier has.

I corrected the file, but my pipeline doesn't inject that block: tier facts reach the model only from the retrieved policy text.

**Data-versus-policy tier gap.** The feature store's `loyalty_tier` is derived from `total_lifetime_value`, while POL-LOY-011 §2 defines tier by trailing 12-month spend. The current pipeline takes the feature-store tier as given, which is a known limitation. Failure 2 in the examples file shows the consequence: an offer told a Silver customer they were "$15 from Gold", computed from lifetime value. The intended production design is to read the tier and trailing spend from the loyalty system of record at generation time. Until then, the prompt should forbid stating distance to a tier and should tie benefits to "your current tier" rather than to a computed number.

### 2.2 Evaluation results (RAGAS, provided harness, 4 cases × 2 runs per config)

Judge: `us.anthropic.claude-haiku-4-5-20251001-v1:0` at temperature 0, with Titan v2 embeddings for answer relevancy. Per-case scores for every run are in:
- [`lab3-evidence/rag_eval.csv`](lab3-evidence/rag_eval.csv): final config, `tiercard`
- [`rag_eval_improved.csv`](lab3-evidence/rag_eval_improved.csv)
- [`rag_eval_baseline.csv`](lab3-evidence/rag_eval_baseline.csv)

No run produced a `NaN`.

**Final configuration (`tiercard`), per case:**

| Case | Tier | F run 1 | F run 2 | AR run 1 | AR run 2 | CR run 1 | CR run 2 |
|---|---|---|---|---|---|---|---|
| CUST-10000317 | Gold | 0.500 | 0.500 | 0.347 | 0.338 | 1.000 | 1.000 |
| CUST-10000904 | Bronze | 0.400 | 0.333 | 0.434 | 0.442 | 0.600 | 0.600 |
| CUST-10000122 | Platinum | 0.444 | 0.375 | 0.356 | 0.363 | 0.333 | 0.333 |
| CUST-10000588 | Silver | 0.600 | 0.600 | 0.335 | 0.335 | 0.500 | 0.500 |
| **Mean** | | **0.486** | **0.452** | **0.368** | **0.369** | **0.608** | **0.608** |

*(F = faithfulness, AR = answer relevancy, CR = context recall.)*

**All three configurations: mean [range across runs]**

| Config | Faithfulness | Answer relevancy | Context recall |
|---|---|---|---|
| `baseline` (chunks only) | 0.412 [0.372–0.453] | 0.376 [0.364–0.388] | 0.496 |
| `improved` (+ customer profile) | 0.521 [0.462–0.580] | 0.369 [0.355–0.383] | 0.608 |
| **`tiercard` (+ profile + tier card), final** | **0.469 [0.452–0.486]** | **0.369 [0.368–0.369]** | **0.608** |
| Original target (no longer graded) | 0.80 | 0.75 | 0.70 |
| Lab reference pipeline | 0.47–0.68 | 0.38–0.45 | 0.53–0.58 |

### 2.3 Two measured changes

**Change 1: customer profile as a context (`baseline` → `improved`)**

| Metric | Baseline | Improved | Change |
|---|---|---|---|
| Faithfulness | 0.412 [0.372–0.453] | 0.521 [0.462–0.580] | **+0.109** |
| Answer relevancy | 0.376 [0.364–0.388] | 0.369 [0.355–0.383] | −0.007 (noise) |
| Context recall | 0.496 | 0.608 | **+0.112** |

**Judgement: it helped, for the reason expected.** The two configs' faithfulness ranges don't overlap (baseline max 0.453 < improved min 0.462), though only just.
- Faithfulness rose because claims taken from the profile ("you shop primarily online", "your top category, Outdoor") became checkable.
- Context recall rose because the ground truths say things like "should reference Outdoor or Footwear products and acknowledge Gold status", and only the profile contains those facts.
- Answer relevancy didn't move, as expected.

**Change 2: tier card (`improved` → `tiercard`)**

| Measure | Improved | Tier card | Change |
|---|---|---|---|
| Faithfulness | 0.521 [0.462–0.580] | 0.469 [0.452–0.486] | −0.052 (inside the improved range) |
| Answer relevancy | 0.369 | 0.369 | 0 |
| Context recall | 0.608 | 0.608 | 0 |
| **Offers citing another tier's benefit** (hand-checked against §4) | **2 of 8** | **0 of 8** | **eliminated** |

(The baseline also had 2 of 8.)

**Judgement: it fixed the failure it targeted, and the faithfulness metric didn't register that.**
- **The targeted failure is gone.** Wrong-tier claims went from 2 of 8 offers to 0 of 8. The Bronze customer is no longer promised Silver's $10 birthday reward, and the Silver customer now gets its own 12-hour early access instead of Gold's 24 hours.
- **The score didn't rise, for two reasons:**
  - Faithfulness counts every unsupported claim equally, and the tier-card offers still contain invented figures (Failure 2): "$4.68 in redemption value on every $100" (wrong: 1.5× earns 150 points, which is $7.50), "an extra 500 bonus points", and "your $2,015 lifetime value goal".
  - The −0.05 change lies inside the earlier run-to-run spread.
- **I kept the tier card in the final pipeline.** Under POL-LOY-011 §5.1, a wrong-tier benefit makes the offer invalid, while invented figures are a separate problem with a separate fix (3.4).
- **Lesson:** an aggregate faithfulness score can't tell a policy-violating claim from a harmless one, so the targeted check has to be measured directly.

### 2.4 What limits each metric

- **Faithfulness (0.47 final, 0.52 before the tier card): invented claims, not missing retrieval.** The benefits table ranked first in every case. Before the tier card:
  - CUST-10000904 (Bronze) was promised Silver's $10 birthday reward in 3 of 4 runs. Baseline run 1 scored 0.000.
  - CUST-10000588 (Silver) was given Gold's 24-hour early access in improved run 2, which dropped it from 0.875 to 0.375.
  - CUST-10000122 (Platinum) scores 0.36–0.55 in every run because each offer contains a self-computed "nearly $9 in redemption value per transaction", which is in no context (and is wrong).

  The tier card removed the wrong-tier claims (2.3). What still limits the score is invented figures ("500 bonus points", "$4.68 per $100", "$2,015 lifetime value goal"), which no context supports. Judge noise is also real. CUST-10000317's improved run 1 has no unsupported fact I could find and still scored 0.50. See [`lab3-evidence/track-b-examples.md`](lab3-evidence/track-b-examples.md).
- **Answer relevancy (0.37): limited by the metric, not the offers.** RAGAS generates questions from the answer and compares them with the input "question". Every input is an instruction ("Generate a retention offer for this Silver member…") and every answer is marketing copy addressed to the customer, so the generated questions ("What benefits do Silver members get?") never resemble the instruction. All 16 scored offers sit between 0.33 and 0.46 whatever their quality: the best offer (588 run 1, F 0.875) scored 0.350.
- **Context recall (0.61): capped by the ground truths.** It is identical across runs because retrieval is deterministic and the judge runs at temperature 0. Platinum (CUST-10000122) is stuck at 0.333 because its ground truth includes grading commentary that appears nowhere in the corpus (that a heavy discount "reads as transactional" for a high-AOV customer). CUST-10000317 reaches 1.000 once the profile supplies "Outdoor/Footwear" and "Gold".

---

## Part 3 — Design Justification (Task 3)

### 3.1 Why gradient boosting for the churn model

NorthStar's churn data is 9,999 customers (6,999 training rows) with 11 tabular features on mixed scales: day counts, dollar amounts, ratios. At that size a neural network has nothing to learn that trees can't, needs far more tuning, and gives the retention team nothing to inspect. The real alternative is logistic regression. It is auditable, but the signal here is non-linear and interactive: a customer 40 days out with high 180-day frequency is a different risk from one 40 days out who has bought once, and logistic regression only sees that if I hand-build bins and interaction terms. Gradient boosting finds those thresholds itself. The final model (depth 3, eta 0.05, up to 400 rounds) keeps each tree to three splits, so gain importance, and SHAP values later, can give the retention team a reason for each flagged customer. The experiment also showed depth 6 bought nothing over depth 3 (AUC 0.7764 vs 0.7840, well inside the ±0.03 split-to-split noise), so the simpler model cost nothing. I would switch to logistic regression if a published, linear scorecard were required for audit. I would switch to a sequence model if we had raw per-transaction histories at many times this volume.

### 3.2 What the features bought

The recency-only baseline (XGBoost on `days_since_last_purchase` alone, same rows) reached AUC 0.7223. The full model reached 0.7840, a lift of **+0.0617** with a paired-bootstrap 95% CI of **[0.0416, 0.0795]**. The interval excludes zero, so the feature set does measurable work, and it beats the lab reference (+0.0464). Precision@10% is 0.7267 and recall@10% is 0.3303.

Where the lift comes from is less flattering:
- By gain, recency still leads (76.4), followed by 180-day frequency (51.7), 30-day frequency (28.7), average order value (24.3) and lifetime value (21.1).
- The three behavioural features Lab 2 built to catch still-active churners rank near the bottom: `category_diversity_score` 13.6, `online_to_store_ratio` 12.8, `avg_basket_size_6m` 12.3. Only tenure (11.2) is lower. By this measure they look decorative.
- Gain splits credit between correlated features, so this isn't proof; an ablation would settle it.

Silver is the weakest tier (AUC 0.714 on 1,139 test customers with 226 churners), which is large enough to be a real finding. Platinum's 0.878 rests on 21 churners and supports no conclusion.

### 3.3 LLM system: RAG rather than fine-tuning

The alternative is fine-tuning Haiku on retention offers. I rejected it for three technical reasons:
- **There is no training set.** No labelled corpus of correct offers exists, only four harness cases.
- **The facts change.** Thresholds, birthday rewards and the 45-day offer validity live in policy documents that get revised. Fine-tuning bakes today's numbers into weights, needs retraining after every revision, and can't show which clause a claim came from. RAG updates by re-embedding 48 chunks in about 10 seconds, and every offer carries chunk IDs a reviewer can check.
- **Faithfulness is the property we must measure.** RAG gives the judge explicit contexts to check against. A fine-tuned model has none.

The starter template shows what goes wrong when facts are written from memory: it had three wrong ones (two tier thresholds and a nonexistent Gold benefit).

**Primary production risk: an offer promising a benefit the customer's tier doesn't carry.** POL-LOY-011 §5.1 declares such an offer invalid and honouring it needs Director approval, so a hallucination becomes a customer-service incident. My evaluation shows the risk is live: the Bronze customer was promised Silver's $10 birthday reward in three of four runs even though the benefits table was the top-ranked chunk. Giving the model only its own tier's column, parsed from the policy at run time, removed wrong-tier claims entirely (0 of 8 offers). The feature-store tier also only approximates the policy tier (lifetime value vs trailing 12-month spend), which turns a model error into a data error as well.

### 3.4 What I would do differently with 10× time or data

- **Churn:** stop ranking by probability against a global top-10% budget.
  - That policy contacts almost none of the valuable at-risk customers: it catches 6 of 52 Gold churners and 5 of 21 Platinum.
  - Instead, rank by expected lost value (p(churn) × lifetime value), or use per-tier budgets.
  - Evaluate with temporal validation (train at an earlier T, test at a later T) and expected revenue retained, rather than a random split and recall@10%.
  - Run an ablation of the three behavioural features.
- **Offers:** add a deterministic guardrail between the generator and the customer.
  - Parse every dollar amount, multiplier and hour count in the offer and check it against a structured copy of the §4 table for the customer's tier, read from the system of record. Regenerate on any mismatch.
  - Grow the evaluation set from 4 cases to about 50, covering every tier and the boundary cases (final sale, Mexico shipping, near-threshold spend), with ground truths written only from the corpus. That way a 0.1 faithfulness change can be told apart from judge noise.
