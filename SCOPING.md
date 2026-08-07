# Step 1 Scoping Decisions

Source: `data/accepted_2007_to_2018q4.csv/accepted_2007_to_2018Q4.csv` (2,260,701 rows, 151 columns), pulled via `scripts/download_data.py`.

## Seasoning cutoff (data-driven, not assumed)

A loan's `loan_status` only becomes final (`Fully Paid` / `Charged Off` / `Default`) once it reaches maturity, defaults, or is paid off early. Loans still within their term show as `Current` regardless of age, so including unseasoned vintages would bias the sample toward "looks fine so far."

Cutoff rule: the most recent issue month, per `term`, where **≥90% of loans have a terminal status**.

| `term` | Seasoning cutoff (issue_d ≤) | % terminal at cutoff |
|---|---|---|
| 36 months | Feb 2016 | 90.0%+ |
| 60 months | Mar 2014 | 90.0%+ |

Loans issued after these cutoffs are excluded — not because they're a different product, but because their outcome isn't known yet, and including them would silently bias the loss rate downward.

## Product-dimension checks

Three categorical fields could plausibly define "product" scope. Each was checked for whether it has enough history *within the seasoned window* to be usable as a feature or segmentation criterion, rather than excluded by assumption:

| Field | Values | Launched | Seasoned-window presence | Verdict |
|---|---|---|---|---|
| `application_type` | Individual / Joint App | Joint App: Oct 2015 | 886 / 741,079 rows (0.12%) | **Excluded** — insufficient seasoned data to model or segment on |
| `disbursement_method` | Cash / DirectPay | DirectPay: Jan 2016 | 600 / 741,079 rows (0.08%) | **Excluded** — same reason |
| `purpose` | debt_consolidation, credit_card, etc. | All major categories since 2007 | Full coverage across seasoning window | **Retained** — legitimate candidate for a feature/segmentation criterion in Step 4+ |

`application_type` and `disbursement_method` fail this check purely because they're recent LendingClub product features that haven't had time to season — not a subjective exclusion.

## Resulting scoped population

- **741,079 loans** (32.8% of the full 2,260,701-row file)
- 739,936 (99.85%) have a terminal outcome (`Fully Paid`, `Charged Off`, or the "does not meet credit policy" terminal variants)
- 1,143 residual non-terminal loans (`Current`, `Late`, `In Grace Period`) that clear the seasoning cutoff but haven't resolved — dropped as noise-level (0.15%) rather than guessed at
- 886 `Joint App` and 600 `DirectPay` rows within the seasoned window are dropped for consistency with the "single product" scope, given they're too sparse to model as their own segment

Net scoped, modelable population: loans issued on/before the term-specific cutoff, `application_type == "Individual"`, `disbursement_method == "Cash"`, with a terminal `loan_status`.

**Final count: 738,476 loans (32.7% of the raw 2,260,701-row file)** — slightly below the 741,079 seasoning-only figure once the sparse `Joint App`/`DirectPay` rows are also dropped (some rows fail both filters, so the two exclusions aren't purely additive).

Produced by `scripts/scope_data.py` → `data/scoped_accepted.csv`.
