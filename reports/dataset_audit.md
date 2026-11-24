# Dataset audit for the DL extension

## Existing AML data (kept, labelled as SILVER)
`data/annotations/sample.jsonl` — 250 diabetes candidate pairs (train 150 / dev 50 / test 50, document-level split).
Labels are `rule_heuristic_regex_v1` pseudo-labels (`PROVISIONAL_RULE_PSEUDO_GOLD`, `human_reviewed=false`).
Distribution: 4 causal / 85 association / 161 none overall; **0 causal in dev and test**. Not human ground truth; used only for the silver comparison.

## Candidate public datasets
| dataset | what it actually labels | fit for CAUSAL / ASSOCIATION / NONE? | access in this build |
|---|---|---|---|
| **SemEval-2010 Task 8** | 9 directed semantic relations + Other between marked nominals; includes `Cause-Effect(e1,e2)` and `(e2,e1)` | **Yes for causal + direction**; no association class | **Downloaded and parsed** (GitHub mirror of the original files: 8,000 train / 2,717 test; label counts checked: Cause-Effect 1,003 train / 328 test) |
| Corr2Cause (Jin et al.) | Whether a *causal-graph hypothesis* follows from a set of correlation statements (formal inference over synthetic text) | **No** — a different task; its binary valid/invalid labels are not relation labels, so they were *not* forced into the 3-class scheme | Hugging Face only (blocked here); assessment from the published description |
| BioRED (Luo et al., 2022) | Human-annotated biomedical entity relations (e.g. association, positive/negative correlation, bind…) | Partially: domain-relevant, but its relation types are association-style, **not an explicit causal label**; mapping "positive correlation" → causal would be unjustified | NCBI FTP (blocked here); assessment from the published description, **not verified by download** |

Only SemEval was verified by actually reading the files. Statements about Corr2Cause and BioRED come from their publications and should be re-checked before citing as facts about the data.

## Chosen dataset and mapping
SemEval-2010 Task 8 → `NONE`, `CAUSE_E1_E2`, `CAUSE_E2_E1` (see `src/preprocessing/semeval.py`).
Split: official test file = test (17 sentences also present in train removed); dev = 10 % of train, stratified by label, identical sentences kept together, seed 42.
Result (`data/processed/semeval_quality_report.json`): train 7,199 / dev 801 / test 2,700; zero normalised-sentence overlap train↔dev and train↔test; zero missing markers.
Imbalance: ~87 % NONE → inverse-frequency class weights.

## Limitations
* General-English nominals, not biomedical abstracts → domain shift.
* SemEval has no document ids; leakage control is at the sentence level (strongest unit available).
* **Licence:** the mirror carries no explicit licence statement; the original SemEval-2010 data was released for research use. Cite Hendrickx et al. (2010) and check the terms before redistributing the files. The raw files are downloaded at run time rather than redistributed by this project's code.
* No association labels can be learned from it; association edges continue to come from the rules extractor.
