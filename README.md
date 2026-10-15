# "Oh No, the Robots Are Here": figures and tables

This repository rebuilds every figure and table in

> Major, S., Tomašević, A., Vranić, A., Cvetković, D., Vudragović, D., Stupovski, B. & Mitrović Dankulov, M. "Oh No, the Robots Are Here": AI identity labels, social uptake and sanctioning in mixed human–AI communities.

The code reads the dataset deposited on Zenodo, [doi:10.5281/zenodo.22964197](https://doi.org/10.5281/zenodo.22964197). That record contains the de-identified platform activity, agent memory, agent interviews, browser telemetry, experiment configuration, annotations and the analysis-ready tables from a 21-day field experiment. In the experiment, 36 student participants and 72 language-model agents discussed popular culture in two YSocial communities. AI accounts were labelled in one community and unlabelled in the other.

## Quick start

Requires Python 3.10 or later. Figure 1 also needs `rsvg-convert` from librsvg (`apt install librsvg2-bin`, `brew install librsvg`); without it, Figure 1 is written as SVG only.

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python -m repro.download   # fetches 07_annotations and 08_analysis_ready (~3 MB) into data/
python -m repro            # writes every figure and table into output/
```

`requirements.txt` gives minimum versions. `requirements-lock.txt` pins the exact versions the code was last tested with (Python 3.13).

`python -m repro.download --all` fetches the complete record (about 130 MB), and `--fit` adds the browser telemetry and experiment configuration the model refits need (about 70 MB). The newest published version of the record is used unless `--record` names one. Every unpacked file is checked against the record's `SHA256SUMS`. To use a copy of the record you already have, set `YSOCIAL_DATA_DIR` to its unpacked directory.

## What each script produces

Each script runs on its own, for example `python -m repro.fig2_reply_target_choice`. Inputs are in `data/08_analysis_ready/results_tables/` unless noted otherwise.

| Script | Output in `output/` | Manuscript item | Inputs |
|---|---|---|---|
| `repro/fig1_study_design.py` | `figures/fig1_study_design.{svg,pdf,png}` | Fig. 1: study design. A drawn schematic with no data input. | — |
| `repro/fig2_reply_target_choice.py` | `figures/fig2_reply_target_choice.{pdf,png}` | Fig. 2 (RQ1): observed and standardized AI-target selection; hidden-minus-visible contrasts | `rq1_primary_estimand.csv`, `rq1_model_estimands.csv`, `rq3_group_summary.csv`, `rq1_choice_set_attrition.csv` |
| `repro/fig3_direct_thread_entry.py` | `figures/fig3_direct_thread_entry.{pdf,png}` | Fig. 3 (additional RQ1 analysis): direct thread entry after feed-card renders | `rq1b_primary_estimands.csv` |
| `repro/fig4_prior_contact.py` | `figures/fig4_prior_contact.{pdf,png}` | Fig. 4 (RQ2): change in selecting the same AI account after one prior reply | `rq2_primary_estimands.csv` |
| | `figures/figS1_prior_contact_support.{pdf,png}` | Supplementary Fig. S1: prior-contact frequencies | `rq2_history_support.csv` |
| | `figures/figS2_prior_contact_sensitivity.{pdf,png}` | Supplementary Fig. S2: doubled-prior and one-to-two-reply projections | `rq2_doubled_prior_estimands.csv`, `rq2_one_to_two_estimands.csv` |
| `repro/figS3_daily_activity.py` | `figures/figS3_daily_activity.{pdf,png}` | Supplementary Fig. S3: daily contributions and active accounts | `design_daily_posts_by_type.csv`, `design_daily_active_authors.csv` |
| `repro/tables.py` | `tables/table1.csv`, `table2.csv`, `tableS1.csv` … `tableS23.csv`, and all of them in `tables/tables.md` | Tables 1–2 and Supplementary Tables S1–S14, S16–S21 and S23, with the same rows, rounding and units as the manuscript | the table-specific result files listed under each table in `tables.md`; Table S12 reads `07_annotations/human_validation/llm_majority_metrics.csv` |
| `repro/download.py` | `data/` | — | Zenodo record |

The tables, grouped by what they report:

| Table | Content |
|---|---|
| 1 | Community composition, activity and analysed replies |
| 2 | Reception: criticism, toxicity and other-participant downvotes by target authorship (RQ3) |
| S1–S3 | RQ1 standardized probabilities, rolling-origin comparison, sampler diagnostics, prior sensitivity |
| S4–S5 | Direct thread entry: estimates, sensitivities and coverage; browser-telemetry quality criteria |
| S6 | Endpoint-specific sample flow |
| S7, S8, S13, S23 | Toxicity by reply dyad; reactions by actor and target; downvote windows; toxicity thresholds |
| S9 | RQ2 contact estimates, denominators, predictive comparison, diagnostics and deletion refits |
| S10, S11, S16 | RSS feeds, AI persona attributes, shared AI browsing and memory settings |
| S12 | Human agreement and language-model performance on the validation samples |
| S14 | Support in the reply-target candidate sets |
| S17–S19 | Criticism model checks, parameters and sensitivities; overlap of criticism, toxicity and AI-identity references with downvotes |
| S20–S21 | Participation and repeated contact; counts behind the AI-identity examples |

## Scope

The figures and tables are built from `08_analysis_ready/results_tables/` in the Zenodo record. Those files hold the posterior summaries, diagnostics and counts produced by the study's analysis pipeline. The Bayesian models are not refitted by `repro`; the code that fitted them is in [`analysis/`](analysis/README.md), with a table of which file fits which model and the commands that refit each model from record version 1.1.0 or later. The canonical input tables for the models are in the same record under `08_analysis_ready/canonical/`, and the other model inputs under `08_analysis_ready/model_inputs/`, `07_annotations/llm_annotations/criticism_labels.csv` and `05_browser_telemetry/`.

Rebuilt tables were checked number by number against the manuscript, and the figures against the manuscript figures. `pytest` checks a set of headline values. A few printed items cannot be rebuilt from the deposited summaries:

- **Table S14:** the pooled "Total" median (IQR) of candidate-set size.
- **Table S23:** the footnote's category shares among toxic replies. Both need row-level inputs that are not in the record.
- **Tables S15 and S22** contain definitions and settings text, with no numbers to rebuild.
- **Table S21:** the quoted examples are an editorial selection from `02_platform_activity/posts.csv`. `tableS21.csv` gives the counts reported with them.

## Licence and citation

Code: MIT (see `LICENSE`). Data: CC BY 4.0, as stated in the Zenodo record. When using the data or code, please cite both the article and the Zenodo record (see `CITATION.cff`).

The study was funded by the Science Fund of the Republic of Serbia, Prizma programme, grant 7416 (CTRUST).
