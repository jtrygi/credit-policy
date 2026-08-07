# Step 2: Cleaning, Feature Engineering, Target, Split

Input: `data/scoped_accepted.csv` (738,476 rows, 152 columns — see SCOPING.md).
Produced by `scripts/prepare_data.py` → `data/train.csv`, `data/val.csv`, `data/test.csv`.

## Target variable

Binary `bad`:
- `bad = 1`: `Charged Off` or `Does not meet the credit policy. Status:Charged Off`
- `bad = 0`: `Fully Paid` or `Does not meet the credit policy. Status:Fully Paid`

Overall bad rate: **15.05%** (consistent across train/val/test after stratified split).

## Leakage removal (the most important decision in this step)

The raw file mixes fields known **at application time** with fields that only exist because LendingClub snapshotted loan *performance* after origination. Any post-origination field would let the model "see the answer" — e.g. `total_pymnt` near-perfectly determines `Fully Paid` vs not. 34 columns were dropped for this reason, including:

- Payment/balance tracking: `out_prncp`, `total_pymnt*`, `total_rec_*`, `recoveries`, `last_pymnt_*`, `next_pymnt_d`
- Updated credit pulls during the loan's life: `last_credit_pull_d`, `last_fico_range_high/low`
- Hardship program fields (14 columns): `hardship_flag`, `hardship_status`, `hardship_amount`, etc.
- Debt settlement fields (6 columns): `debt_settlement_flag`, `settlement_amount`, etc.

None of these are knowable at the point a lending decision is made, so none belong in a model meant to inform that decision.

## Other drops

- **Joint-application-only fields (16 columns)**: 100% missing after scoping to `application_type == "Individual"` (see SCOPING.md) — `sec_app_*`, `*_joint`.
- **Identifiers / free text**: `member_id` (100% missing), `url`, `desc`, `title`, `emp_title` (high-cardinality free text, no signal without NLP).
- **Zero-variance after scoping**: `pymnt_plan`, `policy_code`, `application_type`, `disbursement_method` — all constant once restricted to the seasoned, Individual/Cash population.
- **Redundant with `loan_amnt`**: `funded_amnt`, `funded_amnt_inv` (effectively identical for approved loans in this dataset).

Result: 152 → 87 columns retained as candidate features (before missing-value flag columns are added).

## Feature engineering

- `credit_history_months`: `issue_dt − earliest_cr_line` in months (deterministic date math — computed before the split since it uses no dataset statistics)
- `loan_to_income`: `loan_amnt / annual_inc`
- `emp_length_years`: parsed from text (`"10+ years"` → 10, `"< 1 year"` → 0, `"n years"` → n)
- `term_months`: parsed from `" 36 months"` / `" 60 months"` text to integer
- `home_ownership`: `NONE`/`ANY` (52 rows combined) folded into `OTHER` alongside the existing `OTHER` category — too sparse individually to be meaningful
- `dti`, `revol_util`, `delinq_2yrs` retained as-is — already present as clean origination-time bureau fields, satisfying the design doc's DTI/utilization/delinquency feature requirements directly

## Missing values

69 of the retained numeric columns have missing values. Handling: for each, add a `<col>_missing` indicator column, then impute with the **train-set median** (fit on train only, applied unchanged to val/test — computing statistics on val/test would leak information about their distribution into the "unseen" evaluation sets).

## Train/val/test split

70/15/15 (`train_test_split` twice: 70/30, then the 30% split 50/50), **stratified on `bad`**, `random_state=42`. Done *before* fitting the imputation medians, so no train-set statistic ever touches val or test.

| Split | Rows | Bad rate |
|---|---|---|
| train | 516,933 | 15.049% |
| val | 110,771 | 15.049% |
| test | 110,772 | 15.049% |
