# Model Card: Term-Deposit Outreach Ranker (term_deposit_ranker v1)

Author: Adaora Nkemdilim Okafor | Course: BAN6800, Nexford University | Created: 25 October 2026
Format follows Mitchell et al. (2019). Status: academic prototype, not deployed.

## 1. Model details
- Gradient Boosting classifier (scikit-learn 1.9.1) inside a preprocessing pipeline. Registered in MLflow as `term_deposit_ranker`, version 1, alias `production`.
- Hyperparameters: 200 trees, max depth 3, learning rate 0.1, subsample 0.8, random seed 42 (tuned with 5-fold cross-validation on the full-feature model, then reused without re-tuning).
- Inputs (14): job, marital status, education, credit default, housing loan, personal loan, contact channel, month, weekday, previous-campaign outcome, age, previous contacts, previously contacted flag, days since previous contact.
- Excluded on purpose: `duration` and `campaign` (not known before outreach), raw `pdays` (999 code), and the five economic indicators (see section 7).
- Fairness configuration: score cutoffs set per age band so that each band has similar recall (`models/age_band_cutoffs.json`).
- Serving: the API (`app/main.py`) serves this registered scorer with the age-band cutoffs, as a demonstration of the candidate configuration. The alternative recommended in section 8 (refit without age, marital status and education) was evaluated in `m4/10_sensitivity.py` but is not registered or served.

## 2. Intended use
Rank customer records so that a fixed number of outreach slots goes to those most likely to subscribe to a term deposit. Decision support for relationship managers, who keep responsibility for every decision. In my corporate banking experience, outreach is usually based on what is happening in the customer's business, such as changes in turnover, transaction activity or opportunities to offer relevant products. The model could help make this process more systematic, but it should support the relationship manager's judgement rather than replace the knowledge of the customer and their business.

## 3. Out-of-scope use
- Credit, pricing or eligibility decisions.
- Automated outreach decisions with no human review.
- Use on Nigerian or any other customers without local validation (Module 1 risk register).
- Any use on data from a different period without re-checking performance and fairness.

## 4. Data
- UCI Bank Marketing dataset (Moro et al., 2012): one Portuguese bank, retail telemarketing, 2008-2010.
- After removing 12 exact duplicates: 41,176 records, 4,639 subscribers (11.27%).
- Split: stratified 80/20 with seed 42 (32,940 training rows; 8,236 test rows with 928 subscribers). The data has no customer identifier, so the same customer may appear in more than one row.

## 5. Performance (stratified hold-out test set)
| Configuration | Capture in top 30% (95% CI) | ROC-AUC |
|---|---|---|
| Random selection | 29.9% | 0.50 |
| RM-style rule (previous-campaign outcome) | 46.1% | 0.623 |
| Logistic Regression, no economic indicators | 65.0% (62.4-68.2) | 0.764 |
| **Gradient Boosting, no economic indicators (unmitigated)** | **67.2% (64.7-70.0)** | **0.777** |
| **Same model with age-band cutoffs (candidate)** | **62.4% (59.2-65.7)** | n/a |
| Gradient Boosting without age, marital status and education (alternative) | 66.7% (CI not computed) | 0.775 |
| Gradient Boosting with economic indicators (upper bound only) | 73.5% (71.1-76.4) | 0.810 |

- The 70% capture target from Modules 1 and 2 was **not met** by the model that could be deployed. The 73.5% result depends on economic indicators that mostly identify the campaign period, so it is an upper bound.
- Gradient Boosting and Logistic Regression were not statistically different in their full-feature versions (difference +1.0 point, 95% CI -0.2 to +2.4); without economic indicators the gap is 2.2 points with overlapping intervals, and no paired interval was computed.
- Accuracy is not reported as a headline: the majority class alone scores 88.7%.

## 6. Fairness analysis
Recall = share of each group's subscribers who reach the outreach list (top 30% budget). Overall recall: 67.2% before mitigation, 62.4% after.

| Age band | Subscribers (test) | Recall before | Recall after |
|---|---|---|---|
| 17-24 | 50 | 96.0% | 56.0% |
| 25-34 | 346 | 72.3% | 62.1% |
| 35-44 | 252 | 49.2% | 61.9% |
| 45-54 | 129 | 55.0% | 66.7% |
| 55-64 | 89 | 77.5% | 59.6% |
| 65+ | 62 | 100.0% | 66.1% |

- Bands beyond the 10-point recall-gap rule: 5 before, 0 after. Because bands have 50-346 subscribers, recall estimates carry intervals of roughly 10 points, so "0" means no detectable gap, not proven equality.
- Disparate impact ratio (age bands): 0.161 before, 0.371 after, against the 0.80 screening convention. It fails because subscription rates differ widely by age (about 7% to 52%) and recall parity, not equal selection, was the chosen objective (Module 2).
- Marital status: 2 groups beyond the gap before, 0 after. Education was not mitigated: 3 groups beyond the gap before (basic.6y -24 points, basic.9y -27, "unknown" +10) and 3 after (basic.6y -14, basic.9y -31, university degree +12), and its ratio worsened from 0.327 to 0.270.
- Not assessed: gender (no column). Marital "unknown" (12 rows) and education "illiterate" were too small to judge.
- Alternative tested: refitting without age, marital status and education cost 0.5 points of capture (66.7%) and reduced the age-band breaches from 5 to 2, because other features still stand in for age.

## 7. Limitations and risks
- Economic indicators and month encode the campaign period. Without the indicators capture falls from 73.5% to 67.2%; without month as well, to 59.4%. Counterfactual checks show month alone can move any customer onto the list, so month-based "advice" is not actionable.
- Time-ordered stress test: test-period subscription rate was 30.8% against 6.4% in training, and capture fell to 48-53% (AUC 0.65-0.70). Expect drift.
- Single bank, single period, retail customers; relationship-manager corporate banking differs (Modules 1 and 2).
- Sensitivity: age noise of about 1 year changes nothing; about 5 years costs 1.7 points; replacing "unknown" values with the most common category costs 0.75 points.
- The API applies a per-customer cutoff that approximates the 30% budget; a real deployment ranks a whole list. The API has no authentication, and the model file is a pickle, so load only files produced by the bank itself.

## 8. Ethical considerations
- Age-band cutoffs treat customers differently by age (a 17-24 customer needs a score of 0.325, a 35-44 customer 0.077). This improves equal recall but uses a protected attribute at decision time, and it cost 4.8 points of capture. In my view, the cutoffs should not be adopted automatically just because they reduce the recall gaps. Capture falls from 67.2% to 62.4%, and neither version meets the 70% target. I also tested a model refitted without age, marital status and education. It captured 66.7%, a loss of 0.5 points that is within sampling noise, and it cut the age-band breaches from five to two. Of the three options, I would take this one to the committee, because it reduces reliance on these characteristics at almost no cost to capture. It is not a full solution: the 17-24 and 65+ bands are still beyond the 10-point rule, and other features such as job and previous-campaign outcome can still stand in for age. The committee and the compliance team should decide whether the remaining gap is acceptable before any use.

- Outreach is not harmless: being ranked low means less attention, being ranked high means more contact. From my relationship-banking experience, customers have different needs, and a lower ranking should not mean that a customer is ignored. A customer whose turnover has declined may need attention even if the model gives them a low score, while repeated contact with another customer may not be useful. I would use the ranking to guide outreach, with room for relationship managers to consider the customer's circumstances and record the reasons for overriding the recommendation.
- No direct identifiers, but attribute combinations can still reveal patterns. Raw and processed data are not committed to GitHub.

## 9. Recommended acceptance criteria and monitoring (proposed, confirm before use)
- Capture in the top 30% of at least 60% on a recent labelled sample (the low end of the validated range).
- Age-band recall within 10 points of overall recall for every band with at least 30 subscribers, reported with confidence intervals.
- Monthly: subscription base rate, share of customers by contact channel, month and previous-campaign status, and selection rate by age band, marital status and education.
- Review triggered by a base-rate shift of more than 5 points, a failed data-quality check, or any band breaching the gap rule.

## 10. References
Mitchell, M., Wu, S., Zaldivar, A., Barnes, P., Vasserman, L., Hutchinson, B., Spitzer, E., Raji, I. D., & Gebru, T. (2019). Model cards for model reporting. Proceedings of the Conference on Fairness, Accountability, and Transparency, 220-229. https://doi.org/10.1145/3287560.3287596

Moro, S., Rita, P., & Cortez, P. (2012). Bank marketing [Data set]. UCI Machine Learning Repository. https://archive.ics.uci.edu/dataset/222/bank%2Bmarketing

Hardt, M., Price, E., & Srebro, N. (2016). Equality of opportunity in supervised learning. Advances in Neural Information Processing Systems, 29, 3315-3323. https://arxiv.org/abs/1610.02413
