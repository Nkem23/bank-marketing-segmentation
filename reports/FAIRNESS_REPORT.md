# Fairness Metrics Report: Term-Deposit Outreach Ranker (BAN6800, Module 4)

Author: Adaora Okafor Nkemdilim | 25 October 2026 | Model: Gradient Boosting without economic indicators (MLflow `term_deposit_ranker` v1)

## 1. Purpose and scope
This report checks whether the model's outreach list treats customer groups differently. Groups tested: age band, marital status and education. Gender cannot be tested because the dataset has no gender column. All results come from the stratified hold-out test set (8,236 records, 928 subscribers) and use the outreach budget from Modules 1 and 2: the top 30% of ranked customers.

## 2. Metrics and thresholds
- **Recall (equal opportunity):** the share of a group's actual subscribers who reach the outreach list. Primary objective from Module 2: each age band's recall within 10 percentage points of overall recall. This is a screening threshold, not proof of fairness.
- **Selection rate (demographic parity):** the share of a group's customers placed on the list.
- **False-positive rate and equalized-odds difference:** how often non-subscribers in a group are placed on the list, compared across groups.
- **Disparate impact ratio:** lowest group selection rate divided by the highest. Screened against the 0.80 "four-fifths" convention. This is a common screening rule, not a Nigerian legal standard, and it was added in Module 4 to meet the assignment brief.
- **Minimum evidence:** a group is judged against the 10-point rule only if it has at least 30 subscribers in the test set. Smaller groups are reported but not judged. Confidence intervals are 95% Wilson intervals.

## 3. Method
1. **Before mitigation:** one score cutoff for everyone, chosen so that 30% of customers are selected. Overall recall is 67.2%.
2. **Mitigation:** a separate score cutoff for each age band, chosen so that recall is about equal across bands while the total stays near 30%. Cutoffs were fitted on cross-validated predictions from the training data only and applied once to the test set. Overall recall after mitigation is 62.4%, with 29.5% of customers selected.
3. Marital status and education were not targeted by the mitigation. Their results after mitigation show only what happens as a side effect.

## 4. Results

### 4.1 Age band
| Group | Subscribers (test) | Selection rate: before -> after | Recall before (95% CI) | Recall after | Gap after, points | Judged? |
|---|---|---|---|---|---|---|
| 17-24 | 50 | 81.3% -> 29.0% | 96.0% (86.5%-98.9%) | 56.0% | -6.4 | yes |
| 25-34 | 346 | 40.3% -> 27.1% | 72.3% (67.3%-76.7%) | 62.1% | -0.3 | yes |
| 35-44 | 252 | 16.1% -> 32.7% | 49.2% (43.1%-55.3%) | 61.9% | -0.5 | yes |
| 45-54 | 129 | 19.2% -> 30.7% | 55.0% (46.4%-63.4%) | 66.7% | +4.3 | yes |
| 55-64 | 89 | 42.9% -> 20.1% | 77.5% (67.8%-85.0%) | 59.6% | -2.8 | yes |
| 65+ | 62 | 100.0% -> 54.2% | 100.0% (94.2%-100.0%) | 66.1% | +3.7 | yes |

Subscription rates in the test set differ widely by age, from 7.5% (45-54) to 51.7% (65+), so a model that follows who subscribes will select the youngest and oldest groups far more often. Before mitigation, five of six age bands were beyond the 10-point rule. After mitigation, none are.

### 4.2 Marital status
| Group | Subscribers (test) | Selection rate: before -> after | Recall before (95% CI) | Recall after | Gap after, points | Judged? |
|---|---|---|---|---|---|---|
| divorced | 104 | 22.8% -> 26.3% | 55.8% (46.2%-64.9%) | 55.8% | -6.6 | yes |
| married | 493 | 22.9% -> 25.2% | 61.5% (57.1%-65.7%) | 59.0% | -3.4 | yes |
| single | 330 | 48.3% -> 40.1% | 79.7% (75.0%-83.7%) | 69.7% | +7.3 | yes |
| unknown | 1 | 25.0% -> 25.0% | 0.0% (0.0%-79.3%) | 0.0% | -62.4 | no (too few subscribers) |

### 4.3 Education
| Group | Subscribers (test) | Selection rate: before -> after | Recall before (95% CI) | Recall after | Gap after, points | Judged? |
|---|---|---|---|---|---|---|
| basic.4y | 74 | 19.5% -> 15.7% | 68.9% (57.7%-78.3%) | 52.7% | -9.7 | yes |
| basic.6y | 35 | 20.5% -> 24.5% | 42.9% (28.0%-59.1%) | 48.6% | -13.8 **beyond 10 pts** | yes |
| basic.9y | 102 | 14.6% -> 12.4% | 40.2% (31.2%-49.9%) | 31.4% | -31.0 **beyond 10 pts** | yes |
| high.school | 200 | 29.5% -> 28.1% | 69.0% (62.3%-75.0%) | 63.0% | +0.6 | yes |
| professional.course | 124 | 26.6% -> 27.3% | 64.5% (55.8%-72.4%) | 59.7% | -2.7 | yes |
| university.degree | 349 | 44.6% -> 45.7% | 75.9% (71.2%-80.1%) | 74.5% | +12.1 **beyond 10 pts** | yes |
| unknown | 44 | 32.6% -> 31.1% | 77.3% (63.0%-87.2%) | 70.5% | +8.1 | yes |

### 4.4 Summary
| Attribute | Disparate impact ratio before -> after | Groups beyond 10-point recall gap before -> after |
|---|---|---|
| Age band | 0.161 -> 0.371 | 5 -> 0 |
| Marital status | 0.472 -> 0.629 | 2 -> 0 |
| Education | 0.327 -> 0.270 | 3 -> 3 |

Before mitigation, the demographic parity difference was 0.839 (age), 0.255 (marital) and 0.300 (education), and the equalized-odds difference was 0.872 (age), 0.245 (marital) and 0.371 (education).

## 5. Interpretation
- **Age gaps closed on the target measure.** All six age bands are now within 10 points of overall recall. The widest remaining gap is 6.4 points (17-24).
- **The cost was real.** Overall capture fell from 67.2% to 62.4%, a loss of 4.8 points. Neither figure meets the 70% target from Modules 1 and 2.
- **The disparate impact ratio still fails.** The age-band ratio is 0.371 against a 0.80 screen. This is expected: recall parity was the chosen objective, and subscription rates differ from 7.5% to 51.7% across age bands, so equal selection rates would mean ignoring who actually subscribes. The two criteria cannot both be met here.
- **Education got no better.** Three groups are still beyond the 10-point rule (basic.9y, basic.6y and university degree) and the ratio fell from 0.327 to 0.270. Education was not mitigated, and it overlaps with age and job, so this should be treated as an unresolved limitation, not as a result.
- **Marital status improved as a side effect** (two groups beyond the rule before, none after), which is not evidence that the mitigation targets marital status.

## 6. Alternative tested
Refitting the model without age, marital status and education gave 66.7% capture (a loss of 0.5 points, within sampling noise of about 1.5 points) and reduced the age bands beyond the rule from five to two. The 17-24 and 65+ bands were still over-selected, because other features such as job and previous-campaign outcome carry information about age. This option avoids using age to set cutoffs at decision time, but it does not meet the Module 2 objective. It was not registered or served.

## 7. Limitations
- **Small groups.** Age bands have 50 to 346 subscribers in the test set, so recall estimates carry intervals of roughly 10 points or more. "No band beyond the rule" means no detectable gap, not equality.
- **One split.** Results come from one stratified split. Differences of one or two points should not be over-read.
- **Not assessed:** gender (no column); marital "unknown" (12 records, 1 subscriber) and education "illiterate" (absent from the test set); combinations of attributes (for example age and job).
- **Different treatment by age.** The mitigated configuration uses age to set each customer's cutoff (a 17-24 customer needs a score of 0.325, a 35-44 customer 0.077). This is a protected attribute used at decision time and should be reviewed by compliance and the ethics committee before any use.
- **Data.** One Portuguese bank, retail customers, 2008-2010. Results may not transfer to Nigerian banking.

## 8. Recommendation
The age-band cutoffs close the age gaps, but they cost 4.8 points of capture and use age directly. The refit without age, marital status and education costs almost nothing but closes only part of the gap. The choice between them is not a purely statistical one, so both should go to the ethics committee and compliance team with these results. The author's recommendation, stated in the Model Card, is to take the refit option forward, noting that two age bands remain beyond the 10-point rule.

## 9. Reproducibility
Scripts: `m4/07_fairness.py` (before), `m4/08_mitigation.py` (after), `m4/10_sensitivity.py` (alternative). MLflow runs: `50_fairness_deployable`, `60_mitigation_age_band_cutoffs`, `80_sensitivity_logged`. Tables: `reports/fairness_metrics.csv`, `reports/fairness_mitigation.csv`.
