# Track B — Offer Generation: Examples and Failure Cases

All examples below come from the graded evaluation runs (`rag/run_eval.py`, saved in
`docs/lab3-evidence/rag_eval.csv` and `rag_eval_baseline.csv`). Every claim in each offer was checked
by hand against `northstar-policy-docs/`.

- **Generator:** `us.anthropic.claude-haiku-4-5-20251001-v1:0`, temperature 0.2
- **Judge (RAGAS):** `us.anthropic.claude-haiku-4-5-20251001-v1:0`, temperature 0
- **Retrieval:** top 6 of 48 header-based chunks. Chunk IDs are `<file>#<chunk index>`.
- **Configs:**
  - "baseline" = retrieved chunks only in `contexts`.
  - "improved" = retrieved chunks plus the customer profile as one extra context string.
  - "tiercard" (final) = improved plus the customer's own column of the §4 benefits table, parsed from the
    policy at run time and added to the prompt and `contexts`.
- **Scores (mean faithfulness / answer relevancy / context recall):** baseline 0.412 / 0.376 / 0.496,
  improved 0.521 / 0.369 / 0.608, tiercard 0.469 / 0.369 / 0.608.
  See `docs/lab3-model-design.md` §2.2–2.3.
- **Scores:** F = faithfulness, AR = answer relevancy, CR = context recall.

Chunks that appear below:

| Chunk ID | Section |
|---|---|
| `loyalty-program-terms#2` | §2 Tiers (thresholds by trailing 12-month spend) |
| `loyalty-program-terms#3` | §2.1 Tier Upgrades |
| `loyalty-program-terms#5` | §2.3 Tier Protection |
| `loyalty-program-terms#6` | §3 Points (earn rates, 100 points = $5) |
| `loyalty-program-terms#9` | §4 Tier Benefits table |
| `loyalty-program-terms#12` | §5.1 Retention Offers (45-day validity, one per member) |
| `customer-faq#2` | FAQ: Loyalty (points expiry, tier thresholds) |
| `return-policy#2` | §1.1 Extended Window for Platinum Members |

---

## Part 1 — Five offers showing correct behavior

### Example 1 — CUST-10000588, Silver, improved run 1 (F 0.875, AR 0.350, CR 0.500)

**Profile:** churn_probability 0.70 · Silver · total_lifetime_value $1,985.00 · days_since_last_purchase 52 ·
avg_order_value $132.30 · category_diversity_score 0.375 · online_to_store_ratio 0.61 · top_categories Home, Beauty

**Offer:**
> As a Silver member, you're earning 1.25× points on every purchase—get closer to Gold tier by focusing on Home category items where you already shop, and you'll unlock Gold benefits like free return shipping and 1.5× points earn rate. Reach $2,000 in trailing 12-month spend to immediately upgrade and gain early access to sales 24 hours before other members.

**Retrieved:** `loyalty-program-terms#9`, `customer-faq#2`, `loyalty-program-terms#2`, `loyalty-program-terms#5`, `loyalty-program-terms#12`, `loyalty-program-terms#6` (+ profile)

**Why it's correct:** Silver's 1.25× rate (§4), the Gold threshold of $2,000 in *trailing 12-month* spend (§2),
immediate upgrade (§2.1), and Gold's free return shipping, 1.5× rate and 24-hour early access (§4) are all
right. It names the customer's top category (Home). It presents Gold benefits as something to unlock rather
than promising them now.

### Example 2 — CUST-10000588, Silver, baseline run 1 (F 0.625, AR 0.352, CR 0.500)

**Profile:** as in Example 1.

**Offer:**
> As a Silver member, you're earning 1.25× points on every purchase—get closer to Gold tier by focusing on Home category items, where you can accumulate rewards faster toward that $2,000 threshold. Enjoy an extra $10 birthday reward and 12-hour early access to our sales, plus free standard shipping on orders over $50 to make reaching Gold more rewarding.

**Retrieved:** `loyalty-program-terms#9`, `customer-faq#2`, `loyalty-program-terms#2`, `loyalty-program-terms#5`, `loyalty-program-terms#12`, `loyalty-program-terms#6`

**Why it's correct:** every benefit belongs to Silver: the $10 birthday reward, 12-hour early access, and free
standard shipping at $50+ (§4). It also gets the 1.25× rate and the $2,000 Gold threshold right.
**Minor issue:** "accumulate rewards faster" in Home isn't supported, because no category has a bonus earn rate.

### Example 3 — CUST-10000317, Gold, improved run 1 (F 0.500, AR 0.328, CR 1.000)

**Profile:** churn_probability 0.81 · Gold · total_lifetime_value $3,120.40 · days_since_last_purchase 47 ·
avg_order_value $186.25 · category_diversity_score 0.5 · online_to_store_ratio 0.72 · top_categories Outdoor, Footwear

**Offer:**
> As a Gold member, you're earning 1.5× points on every purchase and enjoying free standard shipping on all orders—benefits worth leveraging on your next Outdoor gear buy to build your points balance before they expire in 18 months. Since you shop primarily online, your free return shipping means you can try items risk-free, and with your high churn risk, redeeming 100 points for $5 off your next Footwear purchase could be the perfect incentive to re-engage.

**Retrieved:** `loyalty-program-terms#9`, `customer-faq#2`, `loyalty-program-terms#2`, `loyalty-program-terms#5`, `loyalty-program-terms#6`, `loyalty-program-terms#3` (+ profile)

**Why it's correct:** these facts are all right: Gold's 1.5× rate, free standard shipping at any amount, and
free return shipping (§4); points expiring 18 months after the last purchase (FAQ); 100 points = $5 (§3); and
"shops primarily online" (online_to_store_ratio 0.72). It doesn't offer the Platinum-only 60-day return window
or free expedited shipping, which are the two traps in the ground truth. This case scored context recall 1.000.
**Copy issue (not a factual one):** "with your high churn risk" exposes an internal model score to the customer.
The judge scored this case only 0.50 even though I couldn't find an unsupported fact, which shows how noisy the
judge is with only 4 cases.

### Example 4 — CUST-10000317, Gold, baseline run 1 (F 0.500, AR 0.362, CR 0.750)

**Profile:** as in Example 3.

**Offer:**
> As a Gold member, you're earning 1.5× points on every purchase and enjoying free standard shipping on all orders—benefits worth protecting as your trailing spend determines your tier status monthly. We'd love to see you back in our Outdoor category with free return shipping on your next order, plus your points are resetting with any purchase, so don't let them expire in 18 months.

**Retrieved:** `loyalty-program-terms#9`, `customer-faq#2`, `loyalty-program-terms#2`, `loyalty-program-terms#5`, `loyalty-program-terms#6`, `loyalty-program-terms#3`

**Why it's correct:** these facts are all right: Gold's benefits; tier being evaluated monthly on trailing spend
(§2, §2.2); and any purchase resetting the points-expiry clock (FAQ). It also uses the tier rules to motivate a
return visit without inventing a discount.

### Example 5 — CUST-10000904, Bronze, improved run 2 (F 0.500, AR 0.400, CR 0.600)

**Profile:** churn_probability 0.74 · Bronze · total_lifetime_value $312.80 · days_since_last_purchase 68 ·
avg_order_value $41.15 · category_diversity_score 0.125 · online_to_store_ratio 0.20 · top_categories Apparel

**Offer:**
> Your Bronze tier membership earns 1 point per $1 spent on apparel, and you'll get free standard shipping on orders over $75—perfect for stocking up on your favorite apparel styles. This 45-day offer gives you a chance to redeem accumulated points toward your next purchase and reconnect with new arrivals in the apparel category.

**Retrieved:** `loyalty-program-terms#9`, `customer-faq#2`, `loyalty-program-terms#6`, `loyalty-program-terms#2`, `return-policy#2`, `loyalty-program-terms#12` (+ profile)

**Why it's correct:** Bronze's 1× rate and $75+ free-shipping threshold (§4) are right. The "45-day" validity is
grounded: §5.1 says retention offers are valid for 45 days from issue (chunk `loyalty-program-terms#12`). It
avoids the Silver-only $10 birthday reward that the other Bronze runs got wrong (see Failure 1).

---

## Part 2 — Two failure modes

### Failure 1 — A benefit from a higher tier, even though the benefits table ranked first

**Cases:**
- **CUST-10000904 (Bronze), baseline runs 1 and 2 and improved run 1** (F 0.000 / 0.250 / 0.400):
  > "...we'd love to welcome you back with a **$10 birthday reward** when you next visit us in store."

  Bronze's birthday reward is **$5**; $10 is Silver's (§4).
- **CUST-10000588 (Silver), improved run 2** (F 0.375):
  > "...plus **24-hour early access** to our upcoming sales."

  Silver gets **12-hour** early access; 24 hours is Gold's (§4).

**Retrieved (904):** `loyalty-program-terms#9`, `customer-faq#2`, `loyalty-program-terms#6`, `loyalty-program-terms#2`, `return-policy#2`, `loyalty-program-terms#12`

**Diagnosis:** this is a reading failure, not a retrieval failure. The tier-benefits table
(`loyalty-program-terms#9`) was the top-ranked chunk in every run of every case. The model had the right row
and still read the value from the neighbouring column of a five-column markdown table. Under
POL-LOY-011 §5.1, an offer that promises a higher tier's benefit is invalid and needs Director-level approval
to honour. So in production this is the most expensive failure the pipeline can make.

**Proposed fix:**
1. **Remove the ambiguity before generation.** Parse the §4 table once and pass only the customer's own
   column as a plain list ("Bronze benefits: points 1×; birthday reward $5; free standard shipping at $75+;
   return window 30 days; no free return shipping; no early access"). The model then has no other columns to
   misread.
2. **Check after generation.** Extract every dollar amount, multiplier and hour count from the offer with a
   regex. Compare each against that tier's column, and regenerate (or block) any offer that cites a value from
   another tier. This check is deterministic, so it catches the error even when the model makes it.

**Fix 1 applied and measured (config `tiercard`, `docs/lab3-evidence/rag_eval.csv`):** wrong-tier claims went
from **2 of 8 offers to 0 of 8**.
- CUST-10000904 (Bronze) no longer mentions a birthday reward from another tier.
- CUST-10000588 (Silver) now says "early access to sales (12 hours ahead)" in both runs.

Mean faithfulness didn't rise (0.521 → 0.469, inside run-to-run noise), because the remaining unsupported
claims are Failure 2's invented figures. Fix 2 (the number check) is still proposed, not implemented.

### Failure 2 — Invented promotions and arithmetic

**Cases:**
- **CUST-10000122 (Platinum), all four runs** (F 0.364–0.545):
  > "with your $447 average order value, that's **nearly $9 in redemption value per transaction**"

  The policy has no such figure. The model computed it, and computed it wrong: at 2 points per $1, a $447
  order earns 894 points, which is about $44.70 at 100 points = $5 (§3).
- **CUST-10000588 (Silver), baseline run 2** (F 0.571):
  > "get **$10 toward your next order when you spend $150** in Home or Beauty categories this month... With
  > **just $15 in additional spend**, you'll reach Gold tier"

  "$10 when you spend $150" is a promotion that exists nowhere in the corpus. The "$15" is $2,000 − $1,985
  (total_lifetime_value), but POL-LOY-011 §2 sets tier by *trailing 12-month* spend, not lifetime value. This
  is the feature-store/policy gap the lab warns about, turned into a specific promise.

- **Still present after the tier card (config `tiercard`):**
  - CUST-10000317 run 2: "**$4.68 in redemption value on every $100** spent". It should be $7.50: 1.5×
    earns 150 points, and 100 points = $5.
  - CUST-10000904 run 2: "an **extra 500 bonus points** to help you reach Silver tier faster".
  - CUST-10000588: "bringing you closer to your **$2,015 lifetime value goal**".

  So this failure is independent of the tier fix and needs its own.

**Retrieved (122):** `loyalty-program-terms#9`, `customer-faq#2`, `return-policy#2`, `loyalty-program-terms#2`, `loyalty-program-terms#5`, `loyalty-program-terms#6`

**Diagnosis:** the prompt says "use only benefits this tier has" and "no made-up deadlines", but it doesn't ban
invented *offers* or *numbers derived* from the profile. At temperature 0.2 Haiku adds "helpful" figures, and
each one is a claim the judge can't find in any context.

**Proposed fix:**
1. **Prompt rule:** "Every number in the offer must appear verbatim in a policy chunk. Do not calculate, do
   not create promotions, and do not state distance to the next tier."
2. **Generate at temperature 0.**
3. **Run the same number check as in Failure 1.** Any figure that isn't in the retrieved chunks fails the
   offer.
4. **For the tier-distance case specifically:** read trailing 12-month spend from the system of record at
   generation time. Until then, never let the offer state how far a customer is from the next tier, because
   the feature store's `total_lifetime_value` isn't the number the policy uses.
